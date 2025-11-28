# Versión “desde cero” del modelo de remodelación siguiendo el Apéndice 3
# Se estructura por familias: cada bloque tiene un binario de upgrade (o uso)
# y fija la base cuando no hay upgrade. El X_input que alimenta al XGB se
# construye siempre coherente con las dummies/ordinales elegidas.

from typing import Dict, Any, List
import pandas as pd
import numpy as np
import gurobipy as gp

from .xgb_predictor import XGBBundle
from .costs import CostTables
from .features import MODIFIABLE

# ------------------ helpers de encoding/base ------------------

_ORD_MAP = {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4, "No aplica": -1, "NA": -1}
_UTIL_MAP = {"ELO": 0, "NoSeWa": 1, "NoSewr": 2, "AllPub": 3}


def _na2noaplica(s: str) -> str:
    s = str(s).strip()
    return "No aplica" if s in {"NA", "No aplica"} else s


def _is_noaplica(v) -> bool:
    s = str(v).strip()
    if s in {"No aplica", "NA", "", "None"}:
        return True
    try:
        n = int(pd.to_numeric(v, errors="coerce"))
        return n == -1
    except Exception:
        return False


def _coerce_base_value(col: str, val):
    col = str(col)
    if col == "Utilities":
        s = str(val).strip()
        if s in _UTIL_MAP:
            return float(_UTIL_MAP[s])
        try:
            v = int(pd.to_numeric(val, errors="coerce"))
            return float(v if v in (0, 1, 2, 3) else 3)
        except Exception:
            return 3.0
    if col in _ORD_MAP:
        try:
            v = int(pd.to_numeric(val, errors="coerce"))
            if v in (-1, 0, 1, 2, 3, 4):
                return float(v)
        except Exception:
            pass
        return float(_ORD_MAP.get(str(val).strip(), 0))
    try:
        return float(pd.to_numeric(val, errors="coerce"))
    except Exception:
        return 0.0


def build_base_input_row(bundle: XGBBundle, base_row: pd.Series) -> pd.DataFrame:
    cols = list(bundle.feature_names_in())
    row = {}
    for c in cols:
        if "_" in c and c.count("_") >= 1:
            root, cat = c.split("_", 1)
            root = root.replace("_", " ").strip()
            cat = cat.strip()
            base_val = base_row.get(root, None)
            if root == "Central Air":
                base_cat = "Y" if str(base_val).strip().upper() in {"Y", "YES", "1", "TRUE"} else "N"
            else:
                base_cat = _na2noaplica(base_val)
            row[c] = 1.0 if _na2noaplica(base_cat) == _na2noaplica(cat) else 0.0
        else:
            row[c] = float(_coerce_base_value(c, base_row.get(c, 0)))
    return pd.DataFrame([row], columns=cols, dtype=float)


def _put_var_obj(df: pd.DataFrame, col: str, var: gp.Var):
    if col in df.columns:
        if df[col].dtype != "O":
            df[col] = df[col].astype("object")
        df.loc[0, col] = var


def _base_guard_pick_one(m: gp.Model, bins: Dict[str, gp.Var], base_key: str, name: str, z: gp.Var | None = None) -> gp.Var:
    """Garantiza factibilidad: si z=0 fija la categoría base, si z=1 permite cambios."""
    if z is None:
        z = m.addVar(vtype=gp.GRB.BINARY, name=f"upg_{name}")
    if base_key not in bins:
        # fallback: el primero
        base_key = next(iter(bins.keys()))
    for k, v in bins.items():
        if k == base_key:
            m.addConstr(v >= 1 - z, name=f"{name}_stay_base")
        else:
            m.addConstr(v <= z, name=f"{name}_allow_{k}")
    return z


# ------------------ modelo principal ------------------


def build_mip_embed(base_row: pd.Series, budget: float, ct: CostTables, bundle: XGBBundle,
                    base_price=None) -> gp.Model:
    m = gp.Model("remodel_embed_v2")

    # presupuesto
    try:
        budget_usd = float(budget)
        if not np.isfinite(budget_usd) or budget_usd < 0:
            budget_usd = 0.0
    except Exception:
        budget_usd = 0.0
    m._budget_usd = float(budget_usd)

    # base numérica y precio base
    base_X = build_base_input_row(bundle, base_row)
    if base_price is None:
        try:
            base_price = float(bundle.predict(base_X).iloc[0])
        except Exception:
            base_price = 0.0

    feature_order = bundle.feature_names_in()
    # Usamos dtype objeto para poder inyectar Vars sin warnings de dtype
    X_input = base_X.copy().astype(object)

    lin_cost = gp.LinExpr(0.0)

    # ========== UTILITIES ==========
    util_levels = ["ELO", "NoSeWa", "NoSewr", "AllPub"]
    util_base = str(base_row.get("Utilities", "AllPub")).strip()
    util_bin = {u: m.addVar(vtype=gp.GRB.BINARY, name=f"util_is_{u}") for u in util_levels}
    m.addConstr(gp.quicksum(util_bin.values()) == 1, name="UTIL_pick_one")
    # valor ordinal 0..3 (como en el XGB)
    util_ord = m.addVar(lb=0, ub=3, vtype=gp.GRB.INTEGER, name="Utilities_ord")
    m.addConstr(util_ord == gp.quicksum(i * util_bin[u] for i, u in enumerate(util_levels)), name="UTIL_link_ord")
    _put_var_obj(X_input, "Utilities", util_ord)
    # no bajar: si costo menor al base, bloquea
    z_util = _base_guard_pick_one(m, util_bin, util_base, "UTIL")
    lin_cost += gp.quicksum((ct.util_cost(u) if hasattr(ct, "util_cost") else 0.0) * util_bin[u] for u in util_levels if u != util_base)

    # ========== ROOF STYLE / MATL ==========
    roof_styles = ["Flat", "Gable", "Gambrel", "Hip", "Mansard", "Shed"]
    roof_matls = ["ClyTile", "CompShg", "Membran", "Metal", "Roll", "Tar&Grv", "WdShake", "WdShngl"]
    A = {
        "Gable":   {"CompShg":1,"Metal":1,"ClyTile":1,"WdShngl":1,"Membran":0,"Roll":1,"Tar&Grv":1,"WdShake":1},
        "Hip":     {"CompShg":1,"Metal":1,"ClyTile":1,"WdShngl":1,"Membran":1,"Roll":1,"Tar&Grv":1,"WdShake":1},
        "Flat":    {"CompShg":0,"Metal":1,"ClyTile":0,"WdShngl":0,"Membran":1,"Roll":1,"Tar&Grv":1,"WdShake":0},
        "Mansard": {"CompShg":1,"Metal":1,"ClyTile":1,"WdShngl":1,"Membran":1,"Roll":1,"Tar&Grv":1,"WdShake":1},
        "Shed":    {"CompShg":1,"Metal":1,"ClyTile":1,"WdShngl":1,"Membran":1,"Roll":1,"Tar&Grv":1,"WdShake":1},
        "Gambrel": {"CompShg":1,"Metal":1,"ClyTile":1,"WdShngl":1,"Membran":1,"Roll":1,"Tar&Grv":1,"WdShake":1},
    }
    roof_style_base = str(base_row.get("Roof Style", "Gable")).strip()
    roof_matl_base = str(base_row.get("Roof Matl", "CompShg")).strip()
    xs = {s: m.addVar(vtype=gp.GRB.BINARY, name=f"roof_style_is_{s}") for s in roof_styles}
    ym = {t: m.addVar(vtype=gp.GRB.BINARY, name=f"roof_matl_is_{t}") for t in roof_matls}
    m.addConstr(gp.quicksum(xs.values()) == 1, name="ROOF_style_pick")
    m.addConstr(gp.quicksum(ym.values()) == 1, name="ROOF_matl_pick")
    for s in roof_styles:
        _put_var_obj(X_input, f"Roof Style_{s}", xs[s])
    for t in roof_matls:
        _put_var_obj(X_input, f"Roof Matl_{t}", ym[t])
    # estilo fijo a la base (decisión de negocio)
    for s, v in xs.items():
        if s == roof_style_base:
            v.LB = v.UB = 1.0
        else:
            v.UB = 0.0
    # compatibilidad
    for s in roof_styles:
        for t in roof_matls:
            if A.get(s, {}).get(t, 0) == 0:
                m.addConstr(xs[s] + ym[t] <= 1, name=f"ROOF_incompat_{s}_{t}")
    # no bajar costo
    cost_mat_base = ct.get_roof_matl_cost(roof_matl_base) if hasattr(ct, "get_roof_matl_cost") else 0.0
    for t in roof_matls:
        if (ct.get_roof_matl_cost(t) if hasattr(ct, "get_roof_matl_cost") else 0.0) < cost_mat_base:
            ym[t].UB = 0.0
    lin_cost += getattr(ct, "roof_demo_cost", 0.0) + gp.quicksum((ct.get_roof_matl_cost(t) if hasattr(ct, "get_roof_matl_cost") else 0.0) * ym[t] for t in roof_matls if t != roof_matl_base)

    # ========== EXTERIOR 1/2 + QUAL/COND ==========
    EX_MATS = [
        "AsbShng","AsphShn","BrkComm","BrkFace","CBlock","CemntBd","HdBoard","ImStucc",
        "MetalSd","Other","Plywood","PreCast","Stone","Stucco","VinylSd","Wd Sdng","WdShngl",
    ]
    ex1 = {nm: m.addVar(vtype=gp.GRB.BINARY, name=f"ex1_is_{nm}") for nm in EX_MATS}
    ex2 = {nm: m.addVar(vtype=gp.GRB.BINARY, name=f"ex2_is_{nm}") for nm in EX_MATS}
    m.addConstr(gp.quicksum(ex1.values()) == 1, name="EXT_ex1_pick_one")
    has2 = 0 if _is_noaplica(base_row.get("Exterior 2nd")) else 1
    if has2:
        m.addConstr(gp.quicksum(ex2.values()) == 1, name="EXT_ex2_pick_one")
    else:
        for v in ex2.values():
            v.UB = 0.0
    ex1_base = str(base_row.get("Exterior 1st", "Wd Sdng")).strip()
    ex2_base = str(base_row.get("Exterior 2nd", "Wd Sdng")).strip()
    for nm in EX_MATS:
        _put_var_obj(X_input, f"Exterior 1st_{nm}", ex1[nm])
        _put_var_obj(X_input, f"Exterior 2nd_{nm}", ex2[nm])
    # no bajar costo y guarda factibilidad
    cost_ex1_base = ct.ext_mat_cost(ex1_base)
    cost_ex2_base = ct.ext_mat_cost(ex2_base)
    z_ext1 = _base_guard_pick_one(m, ex1, ex1_base, "EXT1")
    if has2:
        z_ext2 = _base_guard_pick_one(m, ex2, ex2_base, "EXT2", z_ext1)  # comparte upgrade
    for nm in EX_MATS:
        if ct.ext_mat_cost(nm) < cost_ex1_base:
            ex1[nm].UB = 0.0
        if has2 and ct.ext_mat_cost(nm) < cost_ex2_base:
            ex2[nm].UB = 0.0
    lin_cost += gp.quicksum(ct.ext_mat_cost(nm) * ex1[nm] for nm in EX_MATS if nm != ex1_base)
    if has2:
        lin_cost += gp.quicksum(ct.ext_mat_cost(nm) * ex2[nm] for nm in EX_MATS if nm != ex2_base)

    # calidades exterior
    QUALS = ["Po","Fa","TA","Gd","Ex"]
    exq = {nm: m.addVar(vtype=gp.GRB.BINARY, name=f"ext_qual_is_{nm}") for nm in QUALS}
    exc = {nm: m.addVar(vtype=gp.GRB.BINARY, name=f"ext_cond_is_{nm}") for nm in QUALS}
    m.addConstr(gp.quicksum(exq.values()) == 1, name="EXT_EQ_onehot")
    m.addConstr(gp.quicksum(exc.values()) == 1, name="EXT_EC_onehot")
    v_exq = m.addVar(lb=0, ub=4, vtype=gp.GRB.INTEGER, name="Exter_Qual_ord")
    v_exc = m.addVar(lb=0, ub=4, vtype=gp.GRB.INTEGER, name="Exter_Cond_ord")
    m.addConstr(v_exq == gp.quicksum(i * exq[q] for i, q in enumerate(QUALS)), name="EXT_EQ_link")
    m.addConstr(v_exc == gp.quicksum(i * exc[q] for i, q in enumerate(QUALS)), name="EXT_EC_link")
    _put_var_obj(X_input, "Exter Qual", v_exq)
    _put_var_obj(X_input, "Exter Cond", v_exc)
    exq_base = str(base_row.get("Exter Qual", "TA")).strip()
    exc_base = str(base_row.get("Exter Cond", "TA")).strip()
    v_exq.LB = _ORD_MAP.get(exq_base, 0); v_exq.UB = 4
    v_exc.LB = _ORD_MAP.get(exc_base, 0); v_exc.UB = 4
    z_exq = _base_guard_pick_one(m, exq, exq_base, "EXT_EQ")
    z_exc = _base_guard_pick_one(m, exc, exc_base, "EXT_EC", z_exq)
    for nm, v in exq.items():
        if _ORD_MAP.get(nm, 0) < _ORD_MAP.get(exq_base, 0):
            v.UB = 0.0
    for nm, v in exc.items():
        if _ORD_MAP.get(nm, 0) < _ORD_MAP.get(exc_base, 0):
            v.UB = 0.0
    lin_cost += gp.quicksum(ct.exter_qual_cost(nm) * exq[nm] for nm in QUALS if nm != exq_base)
    lin_cost += gp.quicksum(ct.exter_cond_cost(nm) * exc[nm] for nm in QUALS if nm != exc_base)

    # ========== MAS VNR ==========
    mvt = {nm: m.addVar(vtype=gp.GRB.BINARY, name=f"mvt_is_{nm}") for nm in ["BrkCmn","BrkFace","CBlock","Stone","No aplica"]}
    m.addConstr(gp.quicksum(mvt.values()) == 1, name="MVT_pick_one")
    for nm, v in mvt.items():
        _put_var_obj(X_input, f"Mas Vnr Type_{nm}", v)
    mva_base = float(_coerce_base_value("Mas Vnr Area", base_row.get("Mas Vnr Area", 0)))
    mva = m.addVar(lb=mva_base, ub=mva_base, name="Mas_Vnr_Area")
    _put_var_obj(X_input, "Mas Vnr Area", mva)
    # fija tipo a base si el costo base es el menor; si no, bloquea los más baratos y cobra costo * área
    base_type = str(base_row.get("Mas Vnr Type", "No aplica")).strip()
    for nm, v in mvt.items():
        if nm == base_type:
            v.LB = v.UB = 1.0
        else:
            v.UB = 0.0
    if hasattr(ct, "mas_vnr_cost"):
        lin_cost += gp.quicksum(ct.mas_vnr_cost(nm) * mva * mvt[nm] for nm in mvt)

    # ========== ELECTRICAL / CENTRAL AIR / UTILITIES ya arriba ==========
    elect_types = ["SBrkr","FuseA","FuseF","FuseP","Mix"]
    ebase = str(base_row.get("Electrical", "SBrkr")).strip()
    elect = {nm: m.addVar(vtype=gp.GRB.BINARY, name=f"elect_is_{nm}") for nm in elect_types}
    m.addConstr(gp.quicksum(elect.values()) == 1, name="ELEC_pick_one")
    for nm, v in elect.items():
        _put_var_obj(X_input, f"Electrical_{nm}", v)
        if ct.electrical_cost(nm) < ct.electrical_cost(ebase):
            v.UB = 0.0
    lin_cost += gp.quicksum(ct.electrical_cost(nm) * elect[nm] for nm in elect if nm != ebase)

    ca_yes = m.addVar(vtype=gp.GRB.BINARY, name="central_air_yes")
    ca_no = m.addVar(vtype=gp.GRB.BINARY, name="central_air_no")
    m.addConstr(ca_yes + ca_no == 1, name="CA_pick_one")
    _put_var_obj(X_input, "Central Air_Y", ca_yes)
    _put_var_obj(X_input, "Central Air_N", ca_no)
    if str(base_row.get("Central Air", "N")).strip().upper() in {"Y", "YES", "1"}:
        ca_yes.LB = ca_yes.UB = 1.0
    else:
        ca_no.LB = ca_no.UB = 1.0
        lin_cost += ct.central_air_install * ca_yes

    # ========== HEATING tipo + QC ==========
    heat_types = ["Floor","GasA","GasW","Grav","OthW","Wall"]
    heat_base = str(base_row.get("Heating", "GasA")).strip()
    heat = {nm: m.addVar(vtype=gp.GRB.BINARY, name=f"heat_is_{nm}") for nm in heat_types}
    m.addConstr(gp.quicksum(heat.values()) == 1, name="HEAT_pick_one_type")
    z_heat = _base_guard_pick_one(m, heat, heat_base, "HEAT_type")
    for nm, v in heat.items():
        _put_var_obj(X_input, f"Heating_{nm}", v)
        if ct.heating_type_cost(nm) < ct.heating_type_cost(heat_base):
            v.UB = 0.0
    lin_cost += gp.quicksum(ct.heating_type_cost(nm) * heat[nm] for nm in heat if nm != heat_base)
    # Heating QC
    hqc = {nm: m.addVar(vtype=gp.GRB.BINARY, name=f"heat_qc_is_{nm}") for nm in QUALS}
    m.addConstr(gp.quicksum(hqc.values()) == 1, name="HEAT_qc_onehot")
    v_hqc = m.addVar(lb=0, ub=4, vtype=gp.GRB.INTEGER, name="Heating_QC_ord")
    m.addConstr(v_hqc == gp.quicksum(i * hqc[q] for i, q in enumerate(QUALS)), name="HEAT_qc_link")
    _put_var_obj(X_input, "Heating QC", v_hqc)
    hqc_base = str(base_row.get("Heating QC", "TA")).strip()
    v_hqc.LB = _ORD_MAP.get(hqc_base, 0); v_hqc.UB = 4
    z_hqc = _base_guard_pick_one(m, hqc, hqc_base, "HEAT_qc")
    for nm, v in hqc.items():
        if _ORD_MAP.get(nm, 0) < _ORD_MAP.get(hqc_base, 0):
            v.UB = 0.0
    lin_cost += gp.quicksum(ct.heating_qc_cost(nm) * hqc[nm] for nm in QUALS if nm != hqc_base)

    # ========== KITCHEN QUAL ==========
    kq = {nm: m.addVar(vtype=gp.GRB.BINARY, name=f"kit_is_{nm}") for nm in QUALS}
    m.addConstr(gp.quicksum(kq.values()) == 1, name="KIT_onehot")
    v_kq = m.addVar(lb=0, ub=4, vtype=gp.GRB.INTEGER, name="Kitchen_Qual_ord")
    m.addConstr(v_kq == gp.quicksum(i * kq[q] for i, q in enumerate(QUALS)), name="KIT_link_int")
    _put_var_obj(X_input, "Kitchen Qual", v_kq)
    kq_base = str(base_row.get("Kitchen Qual", "TA")).strip()
    v_kq.LB = _ORD_MAP.get(kq_base, 0); v_kq.UB = 4
    z_kq = _base_guard_pick_one(m, kq, kq_base, "KIT")
    for nm, v in kq.items():
        if _ORD_MAP.get(nm, 0) < _ORD_MAP.get(kq_base, 0):
            v.UB = 0.0
    lin_cost += gp.quicksum(ct.kitchen_level_cost(nm) * kq[nm] for nm in QUALS if nm != kq_base)

    # ========== BASEMENT COND / FIN TYPES ==========
    bcond = {nm: m.addVar(vtype=gp.GRB.BINARY, name=f"bsmt_cond_is_{nm}") for nm in QUALS + ["No aplica"]}
    m.addConstr(gp.quicksum(bcond.values()) == 1, name="BSC_pick_one")
    v_bcond = m.addVar(lb=-1, ub=4, vtype=gp.GRB.INTEGER, name="Bsmt_Cond_ord")
    m.addConstr(v_bcond == gp.quicksum((-1 if nm=="No aplica" else _ORD_MAP.get(nm,0)) * bcond[nm] for nm in bcond), name="BSC_link")
    _put_var_obj(X_input, "Bsmt Cond", v_bcond)
    bcond_base = str(base_row.get("Bsmt Cond", "TA")).strip()
    v_bcond.LB = _ORD_MAP.get(bcond_base, 0) if not _is_noaplica(bcond_base) else -1
    z_bcond = _base_guard_pick_one(m, bcond, bcond_base if bcond_base in bcond else "No aplica", "BSC")
    for nm, v in bcond.items():
        if nm == "No aplica" and not _is_noaplica(bcond_base):
            v.UB = 0.0
        if nm != "No aplica" and _ORD_MAP.get(nm,0) < _ORD_MAP.get(bcond_base,0):
            v.UB = 0.0
    lin_cost += gp.quicksum(ct.bsmt_cond_cost(nm) * bcond[nm] for nm in QUALS if nm != bcond_base)

    # ========== FIREPLACE QU ==========
    fpq = {nm: m.addVar(vtype=gp.GRB.BINARY, name=f"fireplace_is_{nm}") for nm in QUALS + ["No aplica"]}
    m.addConstr(gp.quicksum(fpq.values()) == 1, name="FQ_pick_one")
    v_fpq = m.addVar(lb=-1, ub=4, vtype=gp.GRB.INTEGER, name="Fireplace_Qu_ord")
    m.addConstr(v_fpq == gp.quicksum((-1 if nm=="No aplica" else _ORD_MAP.get(nm,0)) * fpq[nm] for nm in fpq), name="FQ_link")
    _put_var_obj(X_input, "Fireplace Qu", v_fpq)
    fp_base = str(base_row.get("Fireplace Qu", "No aplica")).strip()
    v_fpq.LB = _ORD_MAP.get(fp_base, -1) if not _is_noaplica(fp_base) else -1
    z_fp = _base_guard_pick_one(m, fpq, fp_base if fp_base in fpq else "No aplica", "FQ")
    for nm, v in fpq.items():
        if nm == "No aplica" and not _is_noaplica(fp_base):
            v.UB = 0.0
        if nm != "No aplica" and _ORD_MAP.get(nm,0) < _ORD_MAP.get(fp_base,0):
            v.UB = 0.0
    lin_cost += gp.quicksum(ct.fireplace_cost(nm) * fpq[nm] for nm in QUALS if nm != fp_base)

    # ========== FENCE ==========
    fences = ["GdPrv","MnPrv","GdWo","MnWw","No aplica"]
    fn = {nm: m.addVar(vtype=gp.GRB.BINARY, name=f"fence_is_{nm}") for nm in fences}
    m.addConstr(gp.quicksum(fn.values()) == 1, name="Fence_pick_one")
    for nm, v in fn.items():
        _put_var_obj(X_input, f"Fence_{nm}", v)
    fence_base = str(base_row.get("Fence", "No aplica")).strip()
    z_fence = _base_guard_pick_one(m, fn, fence_base, "FENCE")
    for nm, v in fn.items():
        if ct.fence_category_cost(nm) < ct.fence_category_cost(fence_base):
            v.UB = 0.0
    lin_cost += gp.quicksum(ct.fence_category_cost(nm) * fn[nm] for nm in fences if nm != fence_base)

    # ========== PAVED DRIVE ==========
    paved = ["Y","P","N"]
    pdv = {nm: m.addVar(vtype=gp.GRB.BINARY, name=f"paved_drive_is_{nm}") for nm in paved}
    m.addConstr(gp.quicksum(pdv.values()) == 1, name="PAVED_pick_one")
    for nm, v in pdv.items():
        _put_var_obj(X_input, f"Paved Drive_{nm}", v)
    pv_base = str(base_row.get("Paved Drive", "Y")).strip()
    z_paved = _base_guard_pick_one(m, pdv, pv_base, "PAVED")
    for nm, v in pdv.items():
        if ct.paved_drive_cost(nm) < ct.paved_drive_cost(pv_base):
            v.UB = 0.0
    lin_cost += gp.quicksum(ct.paved_drive_cost(nm) * pdv[nm] for nm in paved if nm != pv_base)

    # ========== GARAGE QUAL / COND ==========
    gq_cats = QUALS + ["No aplica"]
    gq = {nm: m.addVar(vtype=gp.GRB.BINARY, name=f"garage_qual_is_{nm}") for nm in gq_cats}
    gc = {nm: m.addVar(vtype=gp.GRB.BINARY, name=f"garage_cond_is_{nm}") for nm in gq_cats}
    m.addConstr(gp.quicksum(gq.values()) == 1, name="GQ_pick_one")
    m.addConstr(gp.quicksum(gc.values()) == 1, name="GC_pick_one")
    v_gq = m.addVar(lb=-1, ub=4, vtype=gp.GRB.INTEGER, name="Garage_Qual_ord")
    v_gc = m.addVar(lb=-1, ub=4, vtype=gp.GRB.INTEGER, name="Garage_Cond_ord")
    m.addConstr(v_gq == gp.quicksum((-1 if nm=="No aplica" else _ORD_MAP.get(nm,0)) * gq[nm] for nm in gq), name="GQ_link")
    m.addConstr(v_gc == gp.quicksum((-1 if nm=="No aplica" else _ORD_MAP.get(nm,0)) * gc[nm] for nm in gc), name="GC_link")
    _put_var_obj(X_input, "Garage Qual", v_gq)
    _put_var_obj(X_input, "Garage Cond", v_gc)
    gq_base = str(base_row.get("Garage Qual", "TA")).strip()
    gc_base = str(base_row.get("Garage Cond", "TA")).strip()
    v_gq.LB = _ORD_MAP.get(gq_base, 0) if not _is_noaplica(gq_base) else -1
    v_gc.LB = _ORD_MAP.get(gc_base, 0) if not _is_noaplica(gc_base) else -1
    z_gq = _base_guard_pick_one(m, gq, gq_base if gq_base in gq else "No aplica", "GQ")
    z_gc = _base_guard_pick_one(m, gc, gc_base if gc_base in gc else "No aplica", "GC", z_gq)
    for nm, v in gq.items():
        if nm == "No aplica" and not _is_noaplica(gq_base):
            v.UB = 0.0
        if nm != "No aplica" and _ORD_MAP.get(nm,0) < _ORD_MAP.get(gq_base,0):
            v.UB = 0.0
    for nm, v in gc.items():
        if nm == "No aplica" and not _is_noaplica(gc_base):
            v.UB = 0.0
        if nm != "No aplica" and _ORD_MAP.get(nm,0) < _ORD_MAP.get(gc_base,0):
            v.UB = 0.0
    lin_cost += gp.quicksum(ct.garage_qc_costs.get(nm,0.0) * gq[nm] for nm in gq if nm != gq_base and nm!="No aplica")
    lin_cost += gp.quicksum(ct.garage_qc_costs.get(nm,0.0) * gc[nm] for nm in gc if nm != gc_base and nm!="No aplica")

    # ========== GARAGE FINISH ==========
    gf = {nm: m.addVar(vtype=gp.GRB.BINARY, name=f"garage_finish_is_{nm}") for nm in ["Fin","RFn","Unf","No aplica"]}
    m.addConstr(gp.quicksum(gf.values()) == 1, name="GaFin_pick_one")
    for nm, v in gf.items():
        _put_var_obj(X_input, f"Garage Finish_{nm}", v)
    gf_base = str(base_row.get("Garage Finish", "Unf")).strip()
    z_gf = _base_guard_pick_one(m, gf, gf_base if gf_base in gf else "No aplica", "GaFin")
    for nm, v in gf.items():
        if nm == "No aplica" and not _is_noaplica(gf_base):
            v.UB = 0.0
    lin_cost += gp.quicksum(ct.garage_finish_cost(nm) * gf[nm] for nm in gf if nm != gf_base and nm != "No aplica")

    # ========== POOL QC ==========
    pq = {nm: m.addVar(vtype=gp.GRB.BINARY, name=f"poolqc_is_{nm}") for nm in QUALS + ["No aplica"]}
    m.addConstr(gp.quicksum(pq.values()) == 1, name="PoolQC_pick_one")
    v_pq = m.addVar(lb=-1, ub=4, vtype=gp.GRB.INTEGER, name="Pool_QC_ord")
    m.addConstr(v_pq == gp.quicksum((-1 if nm=="No aplica" else _ORD_MAP.get(nm,0)) * pq[nm] for nm in pq), name="PoolQC_link_ord")
    _put_var_obj(X_input, "Pool QC", v_pq)
    pq_base = str(base_row.get("Pool QC", "No aplica")).strip()
    v_pq.LB = _ORD_MAP.get(pq_base, -1) if not _is_noaplica(pq_base) else -1
    z_pq = _base_guard_pick_one(m, pq, pq_base if pq_base in pq else "No aplica", "PoolQC")
    for nm, v in pq.items():
        if nm == "No aplica" and not _is_noaplica(pq_base):
            v.UB = 0.0
        if nm != "No aplica" and _ORD_MAP.get(nm,0) < _ORD_MAP.get(pq_base,0):
            v.UB = 0.0
    lin_cost += gp.quicksum(getattr(ct, "poolqc_costs", {}).get(nm, 0.0) * pq[nm] for nm in QUALS if nm != pq_base)

    # ========== AMPLIACIONES (z10/20/30) ==========
    COMPONENTES = ["GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea"]
    z = {c: {10: m.addVar(vtype=gp.GRB.BINARY, name=f"z10_{c}"),
             20: m.addVar(vtype=gp.GRB.BINARY, name=f"z20_{c}"),
             30: m.addVar(vtype=gp.GRB.BINARY, name=f"z30_{c}")} for c in COMPONENTES}

    def _comp_cost_per_ft2(comp: str, base_row: pd.Series, ct: CostTables) -> float:
        comp = comp.replace("GarageArea", "Garage Area").replace("WoodDeckSF", "Wood Deck SF").replace("OpenPorchSF","Open Porch SF").replace("EnclosedPorch","Enclosed Porch").replace("3SsnPorch","3Ssn Porch").replace("ScreenPorch","Screen Porch").replace("PoolArea","Pool Area")
        if comp == "Pool Area":
            return float(getattr(ct, "pool_area_cost", 0.0))
        if comp == "Wood Deck SF":
            return float(getattr(ct, "wooddeck_cost", 0.0))
        if comp == "Open Porch SF":
            return float(getattr(ct, "openporch_cost", 0.0))
        if comp == "Enclosed Porch":
            return float(getattr(ct, "enclosedporch_cost", 0.0))
        if comp == "3Ssn Porch":
            return float(getattr(ct, "threessnporch_cost", 0.0))
        if comp == "Screen Porch":
            return float(getattr(ct, "screenporch_cost", 0.0))
        if comp == "Garage Area":
            try:
                foundation = str(base_row.get("Foundation", "No aplica")).strip()
                fcost = float(ct.foundation_cost(foundation)) if hasattr(ct, "foundation_cost") else 0.0
            except Exception:
                fcost = 0.0
            return float(getattr(ct, "construction_cost", 0.0) + fcost)
        return 0.0

    for c in COMPONENTES:
        m.addConstr(z[c][10] + z[c][20] + z[c][30] <= 1, name=f"AMPL_one_scale_{c}")
        area_col = c.replace("GarageArea", "Garage Area").replace("WoodDeckSF", "Wood Deck SF").replace("OpenPorchSF","Open Porch SF").replace("EnclosedPorch","Enclosed Porch").replace("3SsnPorch","3Ssn Porch").replace("ScreenPorch","Screen Porch").replace("PoolArea","Pool Area")
        base_area = float(_coerce_base_value(area_col, base_row.get(area_col, 0)))
        var_area = m.addVar(lb=0.0, name=f"x_{area_col}")
        _put_var_obj(X_input, area_col, var_area)
        # incrementos de 10/20/30% del área base
        m.addConstr(var_area == base_area + 0.1*base_area*z[c][10] + 0.2*base_area*z[c][20] + 0.3*base_area*z[c][30], name=f"AMPL_area_{c}")
        # costo por tramo: delta_area * costo_unitario
        unit_cost = _comp_cost_per_ft2(c, base_row, ct)
        lin_cost += unit_cost * (0.1*base_area*z[c][10] + 0.2*base_area*z[c][20] + 0.3*base_area*z[c][30])

    # ========== OBJETIVO ==========
    # embed XGB
    try:
        booster_order = list(bundle.booster_feature_order())
    except Exception:
        booster_order = list(feature_order)
    x_list = []
    missing = []
    for c in booster_order:
        if c in X_input.columns:
            val = X_input.loc[0, c]
            if isinstance(val, gp.Var):
                x_list.append(val)
            else:
                # si viene NaN u otro no finito, reemplaza por 0
                try:
                    v_val = float(val)
                    if not np.isfinite(v_val):
                        v_val = 0.0
                except Exception:
                    v_val = 0.0
                v = m.addVar(lb=v_val, ub=v_val, name=f"const_{c}")
                X_input.loc[0, c] = v
                x_list.append(v)
        else:
            missing.append(c)
    if missing:
        raise RuntimeError(f"Faltan columnas en X_input: {missing[:10]}")

    # embed manual de árboles (reuso de gurobi_model actual)
    # y_log acotado para evitar no acotamiento del PWL/objetivo
    y_log = m.addVar(lb=8.0, ub=16.0, name="y_log")
    bundle.attach_to_gurobi(m, x_list, y_log)
    if bundle.is_log_target():
        y_price = m.addVar(lb=0.0, name="y_price")
        grid = np.linspace(8.0, 16.0, 161)
        m.addGenConstrPWL(y_log, y_price, grid.tolist(), np.expm1(grid).tolist(), name="exp_expm1")
    else:
        y_price = y_log

    m._y_log_var = y_log
    m._y_price_var = y_price
    m._X_input = X_input
    m._X_base_numeric = base_X
    m._feat_order = booster_order
    m._x_vars = {c: X_input.loc[0, c] for c in booster_order}
    m._x_cols = booster_order
    m._base_price_val = base_price

    # costo y presupuesto
    cost_model = m.addVar(lb=0.0, name="cost_model")
    m.addConstr(cost_model == lin_cost, name="COST_eq")
    m.addConstr(cost_model <= budget_usd, name="BUDGET")
    m._cost_var = cost_model
    m._lin_cost_expr = lin_cost
    m._budget = budget_usd

    # objetivo: maximizar y_price - cost_model - base_price
    m.setObjective(y_price - cost_model - base_price, gp.GRB.MAXIMIZE)
    # guard-rail: no aceptar soluciones que bajen el precio vs base
    m.addConstr(y_price >= base_price, name="NO_WORSE_THAN_BASE")
    m.update()
    return m
