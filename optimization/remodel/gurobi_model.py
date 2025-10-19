#optimization/remodel/gurobi_model.py
from typing import Dict, Any
import pandas as pd
import gurobipy as gp
import numpy as np

from sklearn.compose import ColumnTransformer

# >>> SHIMS antes de gurobi_ml
from .compat_sklearn import ensure_check_feature_names
ensure_check_feature_names()
from .compat_xgboost import patch_get_booster
patch_get_booster()

try:
    from gurobi_ml.sklearn.pipeline import add_pipeline_constr as _add_sklearn
except Exception:
    from gurobi_ml.sklearn import add_predictor_constr as _add_sklearn

from .features import MODIFIABLE, IMMUTABLE
from .costs import CostTables
from .xgb_predictor import XGBBundle

def _vtype(code: str):
    return gp.GRB.CONTINUOUS if code == "C" else (gp.GRB.BINARY if code == "B" else gp.GRB.INTEGER)

# --- helper: getVarByName seguro (hace m.update() si hace falta) ---
def _get_by_name_safe(m: gp.Model, *names: str):
    """
    Intenta devolver la primera variable cuyo nombre coincide con alguno
    de los pasados. Si el índice de nombres aún no existe, hace m.update()
    y vuelve a intentar.
    """
    for nm in names:
        try:
            v = m.getVarByName(nm)
        except gp.GurobiError:
            m.update()  # construye el índice de nombres
            v = m.getVarByName(nm)
        if v is not None:
            return v
    return None


def add_ohe_quality_block(
    m, X_input, base_row, feature_name: str,
    cats=("No aplica","Po","Fa","TA","Gd","Ex"),
    forbid_build_when_na=True,
    upgrade_only=True,
):
    """
    Crea binarios one-hot para 'feature_name' con categorías 'cats',
    los inyecta a X_input y aplica reglas de política:
      - if forbid_build_when_na: si la base es 'No aplica', se fija a 'No aplica'
        (no permitimos crear el ítem).
      - if upgrade_only: prohíbe 'degradar' calidad (usando orden ordinal
        No aplica < Po < Fa < TA < Gd < Ex). Si la base es 'No aplica',
        ya está fijo por la regla anterior.
    Devuelve: dict {cat: var_bin}, cat_base, cat_pick
    """
    # 1) crear binarios (deben existir en features.py como MODIFIABLE)
    z = {}
    for cat in cats:
        varname = f"{feature_name}_{cat}"
                # buscar variables ya creadas (prefiere "x_..."); si no existen, créalas
        v = _get_by_name_safe(m, f"x_{varname}", varname)
        if v is None:
            v = m.addVar(vtype=gp.GRB.BINARY, name=f"x_{varname}")

        if v is None:
            v = _get_by_name_safe(m, f"x_{varname}", varname)
            if v is None:
                v = m.addVar(vtype=gp.GRB.BINARY, name=f"x_{varname}")

        if v is None:
            # si no existe, créalo (fallback). Mejor si siempre viene de features.py.
            v = m.addVar(vtype=gp.GRB.BINARY, name=f"x_{varname}")
        z[cat] = v

    # 2) one-hot
    m.addConstr(gp.quicksum(z.values()) == 1, name=f"OHE_{feature_name}_onehot")

    # 3) inyectar a X_input (si el pipeline tiene esas columnas)
    for cat, v in z.items():
        col = f"{feature_name}_{cat}"
        if col in X_input.columns:
            if X_input[col].dtype != "O":
                X_input[col] = X_input[col].astype("object")
            X_input.loc[0, col] = v

    # 4) política: no construir si base es "No aplica"
    base_txt = str(base_row.get(feature_name, "No aplica")).strip()
    if forbid_build_when_na and base_txt == "No aplica":
        # fijar 'No aplica' = 1, resto 0
        for cat, v in z.items():
            if cat == "No aplica":
                v.LB = 1.0
                v.UB = 1.0
            else:
                v.UB = 0.0
        return z, base_txt, "No aplica"

    # 5) upgrade-only (si aplica)
    if upgrade_only:
        ORD = {"No aplica": -1, "Po":0, "Fa":1, "TA":2, "Gd":3, "Ex":4}
        base_ord = ORD.get(base_txt, 0)  # si no mapea, trata como Po

        # crea una variable entera ordinal ligada al one-hot (para comparar >=)
        q_int = m.addVar(lb=-1, ub=4, vtype=gp.GRB.INTEGER, name=f"{feature_name}_ord")
        m.addConstr(
            q_int
            == (-1)*z["No aplica"] + 0*z["Po"] + 1*z["Fa"] + 2*z["TA"] + 3*z["Gd"] + 4*z["Ex"],
            name=f"{feature_name}_ord_link"
        )
        # no empeorar (mantener o subir)
        m.addConstr(q_int >= base_ord, name=f"{feature_name}_upgrade_only")

    # 6) devolver también cuál queda escogida (útil para reporte)
    pick_var = None
    for cat, v in z.items():
        if v.X > 0.5 if m.Status in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT) else False:
            pick_var = cat
            break

    return z, base_txt, pick_var


def build_mip_embed(base_row: pd.Series, budget: float, ct: CostTables, bundle: XGBBundle) -> gp.Model:
    m = gp.Model("remodel_embed")

    # -------------------
    # 1) Variables de decisión
    # -------------------
    x: dict[str, gp.Var] = {}
    for f in MODIFIABLE:
        x[f.name] = m.addVar(lb=f.lb, ub=f.ub, vtype=_vtype(f.vartype), name=f"x_{f.name}")

    lin_cost = gp.LinExpr(ct.project_fixed)

    def pos(expr):
        v = m.addVar(lb=0.0, name=f"pos_{len(m.getVars())}")
        m.addConstr(v >= expr)
        return v

    # >>> MUY IMPORTANTE: construir el índice de nombres antes de cualquier getVarByName
    m.update()

    # -------------------
    # 2) Armar X_input (fila 1xN en el ORDEN que espera el modelo)
    # -------------------
    feature_order = bundle.feature_names_in()
    modif = {f.name for f in MODIFIABLE}

    # Base: var si es modificable, si no el valor de la casa base
    row_vals: Dict[str, Any] = {}
    for fname in feature_order:
        row_vals[fname] = x[fname] if fname in modif else base_row.get(fname, 0)

    # ... arriba: X_input = pd.DataFrame([row_vals], columns=feature_order)
    X_input = pd.DataFrame([row_vals], columns=feature_order, dtype=object)
    # (mantén el _align_ohe_dtypes(...) tal como ya lo tienes)


    # ==== OHE de calidades con "No aplica" explícito ====
    for fname in ["Fireplace Qu","Bsmt Qual","Bsmt Cond","Garage Qual","Garage Cond","Pool QC"]:
        add_ohe_quality_block(
            m, X_input, base_row,
            feature_name=fname,
            cats=("No aplica","Po","Fa","TA","Gd","Ex"),
            forbid_build_when_na=True,   # <— no construir si base es No aplica
            upgrade_only=True            # <— no degradar calidades existentes
        )



    # ========= KITCHEN QUAL (simple, sin 'eligible') =========
    KITCH_LEVELS = ["Po","Fa","TA","Gd","Ex"]
    ORD = {"Po":0,"Fa":1,"TA":2,"Gd":3,"Ex":4}

    kq_base_txt = str(base_row.get("Kitchen Qual","TA")).strip()
    kq_base = ORD.get(kq_base_txt, 2)

    # one-hot
    kit_bins = {nm: m.addVar(vtype=gp.GRB.BINARY, name=f"kit_is_{nm}") for nm in KITCH_LEVELS}
    m.addConstr(gp.quicksum(kit_bins.values()) == 1, name="KIT_onehot")

    # no-empeorar
    for nm, v in kit_bins.items():
        if ORD[nm] < kq_base:
            v.UB = 0.0

    # (opcional) link al entero si tu XGB lo usa ordinal
    if "Kitchen Qual" in x:
        m.addConstr(
            x["Kitchen Qual"] ==
            0*kit_bins["Po"] + 1*kit_bins["Fa"] + 2*kit_bins["TA"] + 3*kit_bins["Gd"] + 4*kit_bins["Ex"],
            name="KIT_link_int"
        )

    kit_cost = gp.LinExpr(0.0)
    for nm, vb in kit_bins.items():
        if ORD[nm] > kq_base:
            kit_cost += ct.kitchen_level_cost(nm) * vb
    lin_cost += kit_cost

    # ========= FIN =========


    # ================== (EXTERIOR) ==================

    # === Exter Qual / Exter Cond → upgrade-only + costo por nivel ===
    def _q_to_ord(v):
        mapping = {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4}
        try:
            return int(v)
        except Exception:
            return mapping.get(str(v), 2)

    if "Exter Qual" in x:
        exq_base = _q_to_ord(base_row.get("Exter Qual", "TA"))
        m.addConstr(x["Exter Qual"] >= exq_base, name="R_EXQ_upgrade_only")
        lin_cost += pos(x["Exter Qual"] - exq_base) * ct.exter_qual_upgrade_per_level

    if "Exter Cond" in x:
        exc_base = _q_to_ord(base_row.get("Exter Cond", "TA"))
        m.addConstr(x["Exter Cond"] >= exc_base, name="R_EXC_upgrade_only")
        lin_cost += pos(x["Exter Cond"] - exc_base) * ct.exter_cond_upgrade_per_level

    # Materiales disponibles (usa mismos nombres que en costs.py y en tu CSV)
    EXT_MATS = [
        "AsbShng","AsphShn","BrkComm","BrkFace","CBlock","CemntBd","HdBoard","ImStucc",
        "MetalSd","Other","Plywood","PreCast","Stone","Stucco","VinylSd","Wd Sdng","WdShngl",
    ]

    # Binarios de decisión (de features.py)
    ex1 = {nm: x[f"ex1_is_{nm}"] for nm in EXT_MATS if f"ex1_is_{nm}" in x}
    ex2 = {nm: x[f"ex2_is_{nm}"] for nm in EXT_MATS if f"ex2_is_{nm}" in x}

    # ¿Existe Exterior2nd en la casa base?
    ex2_base_name = str(base_row.get("Exterior 2nd", "None"))
    Ilas2 = 0 if (ex2_base_name in ["None", "nan", "NaN", "NoneType", "0"] or pd.isna(base_row.get("Exterior 2nd"))) else 1

    # (E1) selección única / activación por Ilas2
    if ex1:
        m.addConstr(gp.quicksum(ex1.values()) == 1, name="EXT_ex1_pick_one")
    if ex2:
        m.addConstr(gp.quicksum(ex2.values()) == Ilas2, name="EXT_ex2_pick_ilas2")

    # (E2) inyectar dummies a X_input (tu modelo trae OHE de exteriores)
    for nm in EXT_MATS:
        col1 = f"Exterior 1st_{nm}"
        if col1 in X_input.columns and nm in ex1:
            X_input.loc[0, col1] = ex1[nm]
        col2 = f"Exterior 2nd_{nm}"
        if col2 in X_input.columns and nm in ex2:
            X_input.loc[0, col2] = ex2[nm]

    # (E3) ELEGIBILIDAD (solo si calidad o condición ≤ TA(=2))
    def _q_to_ord(v):
        M = {"Po":0,"Fa":1,"TA":2,"Gd":3,"Ex":4}
        try:
            return int(v)
        except Exception:
            return M.get(str(v), 2)

    exq_base = _q_to_ord(base_row.get("Exter Qual", 2))
    exc_base = _q_to_ord(base_row.get("Exter Cond", 2))
    eligible = 1 if (exq_base <= 2 or exc_base <= 2) else 0

    # material base (nombres CSV)
    ex1_base_name = str(base_row.get("Exterior 1st"))
    ex2_base_name = str(base_row.get("Exterior 2nd"))

    # Si NO elegible -> obligar a mantener materiales base
    if not eligible:
        # frente 1
        for nm in EXT_MATS:
            if nm in ex1:
                ex1[nm].UB = 1 if nm == ex1_base_name else 0
        # frente 2 (solo si existe)
        if Ilas2 == 1:
            for nm in EXT_MATS:
                if nm in ex2:
                    ex2[nm].UB = 1 if nm == ex2_base_name else 0

    # (E4) Costos de cambio de material (solo si cambias)
    area_ext = ct.exterior_area_proxy(base_row)

    # frente 1: suma de “material distinto al base”
    for nm in EXT_MATS:
        if nm in ex1 and nm != ex1_base_name:
            # demolición + reconstrucción con material nm
            lin_cost += (ct.exterior_demo_face1 * area_ext) * ex1[nm]
            lin_cost += (ct.ext_mat_cost(nm) * area_ext) * ex1[nm]

    # frente 2 (si existe)
    if Ilas2 == 1:
        for nm in EXT_MATS:
            if nm in ex2 and nm != ex2_base_name:
                lin_cost += (ct.exterior_demo_face2 * area_ext) * ex2[nm]
                lin_cost += (ct.ext_mat_cost(nm) * area_ext) * ex2[nm]

    
    # ================== FIN (EXTERIOR) ==================

    # ================== (MAS VNR: tipo + área) ==================
    MV_TYPES = ["BrkCmn", "BrkFace", "CBlock", "No aplica", "Stone"]  # <-- sin "None"

    mvt = {nm: x[f"mvt_is_{nm}"] for nm in MV_TYPES if f"mvt_is_{nm}" in x}

    if mvt:
        m.addConstr(gp.quicksum(mvt.values()) == 1, name="MVT_pick_one")

    def _put_var(df: pd.DataFrame, col: str, var: gp.Var):
        if col in df.columns:
            if df[col].dtype != "O":
                df[col] = df[col].astype("object")
            df.loc[0, col] = var

    for nm in MV_TYPES:
        col = f"Mas Vnr Type_{nm}"
        if col in X_input.columns and nm in mvt:
            _put_var(X_input, col, mvt[nm])

    mvt_base = str(base_row.get("Mas Vnr Type", "No aplica")).strip()
    try:
        mv_area_base = float(pd.to_numeric(base_row.get("Mas Vnr Area"), errors="coerce") or 0.0)
    except Exception:
        mv_area_base = 0.0

    base_cost = ct.mas_vnr_cost(mvt_base)
    mv_area = x["Mas Vnr Area"] if "Mas Vnr Area" in x else mv_area_base

    # política: si base es "No aplica" o área=0 → no se puede agregar
    no_base_veneer = (mv_area_base <= 1e-9) or (mvt_base == "No aplica")
    if no_base_veneer:
        if "No aplica" in mvt:
            m.addConstr(mvt["No aplica"] == 1, name="MVT_stay_noaplica")
        for nm in MV_TYPES:
            if nm != "No aplica" and nm in mvt:
                mvt[nm].UB = 0
        if isinstance(mv_area, gp.Var):
            mv_area.LB = 0.0
            mv_area.UB = 0.0
    else:
        for nm in MV_TYPES:
            if nm in mvt and ct.mas_vnr_cost(nm) < base_cost:
                mvt[nm].UB = 0
        if isinstance(mv_area, gp.Var):
            m.addConstr(mv_area >= mv_area_base, name="MVT_area_no_decrease")

    area_term = mv_area if isinstance(mv_area, gp.Var) else float(mv_area)
    for nm in MV_TYPES:
        if nm in mvt and nm != mvt_base:
            lin_cost += ct.mas_vnr_cost(nm) * area_term * mvt[nm]
    # ================== FIN (MAS VNR) ==================



    # -------------------
    # 4) ROOF: estilo/material con dummies "horneadas" y compatibilidad
    # -------------------
    style_names = ["Flat", "Gable", "Gambrel", "Hip", "Mansard", "Shed"]
    matl_names  = ["ClyTile", "CompShg", "Membran", "Metal", "Roll", "Tar&Grv", "WdShake", "WdShngl"]

    # binarios provenientes de features.py
    s_bin = {nm: x[f"roof_style_is_{nm}"] for nm in style_names if f"roof_style_is_{nm}" in x}
    m_bin = {nm: x[f"roof_matl_is_{nm}"]  for nm in matl_names  if f"roof_matl_is_{nm}"  in x}

    # (exactamente uno)
    if s_bin:
        m.addConstr(gp.quicksum(s_bin.values()) == 1, name="ROOF_pick_one_style")
    if m_bin:
        m.addConstr(gp.quicksum(m_bin.values()) == 1, name="ROOF_pick_one_matl")

    # helper para asignar gp.Var en DataFrame sin warning
    def _put_var(df: pd.DataFrame, col: str, var: gp.Var):
        if col in df.columns:
            if df[col].dtype != "O":  # 'O' = object
                df[col] = df[col].astype("object")
            df.loc[0, col] = var

    # Inyectar dummies en X_input si el modelo las tiene
    for nm in style_names:
        col = f"Roof Style_{nm}"
        if col in X_input.columns and nm in s_bin:
            _put_var(X_input, col, s_bin[nm])

    for nm in matl_names:
        col = f"Roof Matl_{nm}"
        if col in X_input.columns and nm in m_bin:
            _put_var(X_input, col, m_bin[nm])


    # Compatibilidad (0 = prohibido) usando labels del dataset
    for sn, forbids in {
        "Gable":   ["Membran"],
        "Hip":     ["Membran"],
        "Flat":    ["WdShngl", "ClyTile", "CompShg"],   # techo plano: sin tejas ni asfalto tradicional
        "Mansard": ["Membran"],
        "Shed":    ["ClyTile"],                         # ej: evitar teja pesada en shed muy inclinado
    }.items():
        for mn in forbids:
            if (f"roof_style_is_{sn}" in x) and (f"roof_matl_is_{mn}" in x):
                m.addConstr(x[f"roof_style_is_{sn}"] + x[f"roof_matl_is_{mn}"] <= 1,
                            name=f"ROOF_compat_{sn}_{mn}")


# ================== FIN (ROOF) ==================

    # ================== (CENTRAL AIR) ==================
    base_air_raw = str(base_row.get("Central Air", "N")).strip()
    base_is_Y = base_air_raw in {"Y", "Yes", "1", "True"}

    col_Y = "Central Air_Y"
    col_N = "Central Air_N"

    has_Y = col_Y in X_input.columns
    has_N = col_N in X_input.columns

    if has_Y or has_N:
        # 1 binaria de decisión
        air_yes = m.addVar(vtype=gp.GRB.BINARY, name="central_air_yes")

        # Si la base ya tiene aire, no se puede quitar
        if base_is_Y:
            air_yes.LB = 1.0
            air_yes.UB = 1.0
        else:
            # costo al agregar
            lin_cost += ct.central_air_install * air_yes

        # Proxy para el dummy "_N" (en vez de meter 1 - air_yes directo en el DF)
        air_no = None
        if has_N:
            air_no = m.addVar(vtype=gp.GRB.BINARY, name="central_air_no")
            m.addConstr(air_yes + air_no == 1, name="CentralAir_onehot")

        # Inyectar a X_input
        if has_Y:
            if X_input[col_Y].dtype != "O":
                X_input[col_Y] = X_input[col_Y].astype("object")
            X_input.loc[0, col_Y] = air_yes

        if has_N:
            if X_input[col_N].dtype != "O":
                X_input[col_N] = X_input[col_N].astype("object")
            X_input.loc[0, col_N] = air_no


    # ================== FIN (CENTRAL AIR) ==================


    # ================== (ELECTRICAL) ==================
    ELECT_TYPES = ["SBrkr", "FuseA", "FuseF", "FuseP", "Mix"]

    # binarios de decisión (de features.py)
    e_bin = {nm: x[f"elect_is_{nm}"] for nm in ELECT_TYPES if f"elect_is_{nm}" in x}

    # (El1) elegir exactamente uno
    if e_bin:
        m.addConstr(gp.quicksum(e_bin.values()) == 1, name="ELEC_pick_one")

    # (El2) inyectar dummies a X_input (pipeline entrenado con OHE de Electrical)
    for nm, vb in e_bin.items():
        col = f"Electrical_{nm}"
        if col in X_input.columns:
            _put_var(X_input, col, vb)

    # (El3) upgrade-only por costo: solo permitir tipos con costo >= costo del base
    elec_base_name = str(base_row.get("Electrical", "SBrkr"))
    base_cost = ct.electrical_cost(elec_base_name)
    for nm, vb in e_bin.items():
        if ct.electrical_cost(nm) < base_cost:
            vb.UB = 0  # no puedes elegir algo "más barato/peor" que el base

    # (El4) costo: si cambias, pagas demolición pequeña + costo del nuevo tipo
    # Detectar la base para evitar cobrar si te quedas igual
    for nm, vb in e_bin.items():
        if nm != elec_base_name:
            lin_cost += ct.electrical_demo_small * vb
            lin_cost += ct.electrical_cost(nm) * vb
    # ================== FIN (ELECTRICAL) ==================

    # ================== (HEATING + HEATINGQC) ==================
    HEAT_TYPES = ["Floor","GasA","GasW","Grav","OthW","Wall"]
    heat_bin = {nm: x[f"heat_is_{nm}"] for nm in HEAT_TYPES if f"heat_is_{nm}" in x}

    # --- Parámetro de política ---
    qc_threshold = 2  # 2=TA; si quieres exigir EX usa 4

    # Base (tipo y QC)
    heat_base = str(base_row.get("Heating", "GasA")).strip()
    q_map = {"Po":0,"Fa":1,"TA":2,"Gd":3,"Ex":4}
    try:
        qc_base = int(pd.to_numeric(base_row.get("Heating QC"), errors="coerce"))
        if qc_base not in (0,1,2,3,4):
            qc_base = q_map.get(str(base_row.get("Heating QC")).strip(), 2)
    except Exception:
        qc_base = q_map.get(str(base_row.get("Heating QC")).strip(), 2)

    # (H1) elegir exactamente un tipo
    if heat_bin:
        m.addConstr(gp.quicksum(heat_bin.values()) == 1, name="HEAT_pick_one_type")

    # Inyectar dummies Heating_* al X_input si el XGB las tiene
    def _put_var(df, col, var):
        if col in df.columns:
            if df[col].dtype != "O":
                df[col] = df[col].astype("object")
            df.loc[0, col] = var

    for nm, vb in heat_bin.items():
        _put_var(X_input, f"Heating_{nm}", vb)

    # Binarias de caminos (deben existir en MODIFIABLE)
    upg_type = x.get("heat_upg_type")
    upg_qc   = x.get("heat_upg_qc")

    # Elegibilidad: solo si QC_base <= TA
    eligible = 1 if qc_base <= 2 else 0
    if (upg_type is not None) and (upg_qc is not None):
        # Exclusión + gating por elegibilidad
        m.addConstr(upg_type + upg_qc <= eligible, name="HEAT_paths_exclusive")

    # No empeorar calidad (del PDF)
    if "Heating QC" in x:
        m.addConstr(x["Heating QC"] >= qc_base, name="HEAT_qc_upgrade_only")

    # “Cualquier intervención” = OR(upg_type, upg_qc)
    any_rebuild = m.addVar(vtype=gp.GRB.BINARY, name="HEAT_any_rebuild")
    if (upg_type is not None) and (upg_qc is not None):
        m.addConstr(any_rebuild >= upg_type, name="HEAT_any_ge_type")
        m.addConstr(any_rebuild >= upg_qc,   name="HEAT_any_ge_qc")
        m.addConstr(any_rebuild <= upg_type + upg_qc, name="HEAT_any_le_sum")
    else:
        m.addConstr(any_rebuild == 0, name="HEAT_any_no_vars")

    # Piso de calidad si hay intervención
    if "Heating QC" in x:
        m.addConstr(x["Heating QC"] >= qc_threshold * any_rebuild,
                    name="HEAT_qc_min_if_any_rebuild")

    # (H3) upgrade-only de tipo por costo: prohibir tipos más baratos que el base
    base_type_cost = ct.heating_type_cost(heat_base)
    for nm, vb in heat_bin.items():
        if ct.heating_type_cost(nm) < base_type_cost:
            vb.UB = 0  # no bajar de categoría/costo

    # Si NO es elegible: fijar tipo y calidad a base
    if eligible == 0:
        for nm, vb in heat_bin.items():
            vb.UB = 1 if nm == heat_base else 0
        if "Heating QC" in x:
            x["Heating QC"].LB = qc_base
            x["Heating QC"].UB = qc_base

    # (H4) ChangeType = 1 - I_base
    change_type = None
    if heat_bin:
        change_type = m.addVar(vtype=gp.GRB.BINARY, name="heat_change_type")
        if heat_base in heat_bin:
            m.addConstr(change_type == 1 - heat_bin[heat_base], name="HEAT_change_def")
        else:
            m.addConstr(change_type >= 0, name="HEAT_change_def_guard")

    # (H5) Dummies de QC (para costos y, si aplica, inyectar al XGB)
    qc_bins = {}
    if "Heating QC" in x:
        for lvl, nm in enumerate(["Po","Fa","TA","Gd","Ex"]):
            qb = m.addVar(vtype=gp.GRB.BINARY, name=f"heat_qc_is_{nm}")
            qc_bins[nm] = qb
        m.addConstr(gp.quicksum(qc_bins.values()) == 1, name="HEAT_qc_onehot")
        m.addConstr(x["Heating QC"] == 0*qc_bins["Po"] + 1*qc_bins["Fa"] + 2*qc_bins["TA"]
                                    + 3*qc_bins["Gd"] + 4*qc_bins["Ex"], name="HEAT_qc_link")
        for nm, qb in qc_bins.items():
            _put_var(X_input, f"Heating QC_{nm}", qb)

    # (H6) Costos (tal como el PDF)
    cost_heat = gp.LinExpr(0.0)

    # Reconstruir mismo tipo: C_base * (UpgType - ChangeType)
    if (upg_type is not None) and (change_type is not None):
        cost_heat += base_type_cost * (upg_type - change_type)

    # Cambiar a tipo más caro
    for nm, vb in heat_bin.items():
        if nm != heat_base:
            cost_heat += ct.heating_type_cost(nm) * vb

    # Cambiar calidad
    if qc_bins:
        for nm, qb in qc_bins.items():
            if q_map[nm] != qc_base:
                cost_heat += ct.heating_qc_cost(nm) * qb

    # Añadir a la función de costo
    lin_cost += cost_heat
    # ================== FIN HEATING ==================

    # ================== FIREPLACE QU ==================
    if "Fireplace Qu" in x:
        fp_levels_cost = ["Po","Fa","TA","Gd","Ex"]   # niveles con costo (sin "No aplica")
        fp_all = fp_levels_cost + ["No aplica"]

        fp_base_txt = str(base_row.get("Fireplace Qu", "No aplica")).strip()
        has_fp = 0 if fp_base_txt in ["No aplica","NA"] else 1

        if has_fp == 0:
            # --- Sin chimenea en base: NO se puede construir una ---
            # Fijar el ordinal del predictor a un valor neutro (TA=2) para el XGB
            x["Fireplace Qu"].LB = 2
            x["Fireplace Qu"].UB = 2
            # No creamos binarios ni el vínculo FP_link_int
            # (y costo = 0 por definición)
        else:
            # --- Con chimenea en base: crear one-hot y linkear al ordinal ---
            fp_bins = {nm: m.addVar(vtype=gp.GRB.BINARY, name=f"fp_is_{nm}") for nm in fp_levels_cost}
            m.addConstr(gp.quicksum(fp_bins.values()) == 1, name="FP_onehot")

            # vínculo al entero 0..4
            m.addConstr(
                x["Fireplace Qu"] ==
                0*fp_bins["Po"] + 1*fp_bins["Fa"] + 2*fp_bins["TA"] + 3*fp_bins["Gd"] + 4*fp_bins["Ex"],
                name="FP_link_int"
            )

            # habilitar según política
            for nm in fp_levels_cost:
                fp_bins[nm].UB = 0  # parte bloqueado

            if fp_base_txt == "Po":
                fp_bins["Po"].UB = 1
                fp_bins["Fa"].UB = 1
            elif fp_base_txt == "TA":
                fp_bins["TA"].UB = 1
                fp_bins["Gd"].UB = 1
                fp_bins["Ex"].UB = 1
            elif fp_base_txt in {"Fa","Gd","Ex"}:
                fp_bins[fp_base_txt].UB = 1   # mantener
            else:
                # fallback defensivo
                fp_bins["TA"].UB = 1

            # costos si cambias a otro nivel
            def _fp_cost(nm: str) -> float:
                try:
                    return float(ct.fireplace_cost(nm))
                except Exception:
                    return 0.0

            for nm in fp_levels_cost:
                if nm != fp_base_txt:
                    lin_cost += _fp_cost(nm) * fp_bins[nm]
    # ================== FIN FIREPLACE QU ==================



    # ================== BSMT (Fin1, Fin2, Unf, Total) ==================
    # Variables existentes (deben estar en MODIFIABLE)
    b1_var = x.get("BsmtFin SF 1")
    b2_var = x.get("BsmtFin SF 2")
    bu_var = x.get("Bsmt Unf SF")
    if (b1_var is None) or (b2_var is None) or (bu_var is None):
        raise RuntimeError(
            "Faltan variables de sótano en MODIFIABLE: "
            "asegúrate de tener 'BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF'."
        )

    # Bases (constantes de la casa)
    def _num(v): 
        import pandas as pd
        try: return float(pd.to_numeric(v, errors="coerce") or 0.0)
        except: return 0.0

    b1_base = _num(base_row.get("BsmtFin SF 1"))
    b2_base = _num(base_row.get("BsmtFin SF 2"))
    bu_base = _num(base_row.get("Bsmt Unf SF"))
    tb_base = _num(base_row.get("Total Bsmt SF"))
    if tb_base <= 0.0:
        tb_base = b1_base + b2_base + bu_base

    # Transferencias SOLO desde Unf → Fin1/Fin2 (no hay demoliciones)
    tr1 = m.addVar(lb=0.0, name="bsmt_tr1")  # pies² que pasan de Unf a Fin1
    tr2 = m.addVar(lb=0.0, name="bsmt_tr2")  # pies² que pasan de Unf a Fin2

    # Enlaces sin demoler:
    #   Fin1 = Fin1_base + tr1
    #   Fin2 = Fin2_base + tr2
    #   Unf  = Unf_base  - tr1 - tr2
    m.addConstr(b1_var == b1_base + tr1, name="BSMT_link_fin1")
    m.addConstr(b2_var == b2_base + tr2, name="BSMT_link_fin2")
    m.addConstr(bu_var == bu_base - tr1 - tr2, name="BSMT_link_unf")

    # No se puede terminar más de lo que había sin terminar
    m.addConstr(tr1 + tr2 <= bu_base + 1e-6, name="BSMT_no_more_than_unf_base")

    # Conservación del total (seguridad)
    m.addConstr(b1_var + b2_var + bu_var == tb_base, name="BSMT_conservation_sum")

    # Costos: solo cobramos lo que efectivamente se terminó
    lin_cost += ct.finish_basement_per_f2 * (tr1 + tr2)
    # ================== FIN BSMT ==================

 
    # ================== (BSMT COND: ordinal upgrade-only) ==================
    BC_LEVELS = ["Po","Fa","TA","Gd","Ex"]
    BC_ORD = {"Po":0,"Fa":1,"TA":2,"Gd":3,"Ex":4}

    bc_base_txt = str(base_row.get("Bsmt Cond", "TA")).strip()
    bc_base = BC_ORD.get(bc_base_txt, 2)

    # binarios one-hot de estado final
    bc_bin = {nm: m.addVar(vtype=gp.GRB.BINARY, name=f"bsmtcond_is_{nm}") for nm in BC_LEVELS}
    m.addConstr(gp.quicksum(bc_bin.values()) == 1, name="BSMTCOND_onehot")

    # no empeorar: niveles por debajo de la base quedan prohibidos
    for nm, vb in bc_bin.items():
        if BC_ORD[nm] < bc_base:
            vb.UB = 0.0

    # si tu XGB usa la columna ordinal "Bsmt Cond", enlázala
    if "Bsmt Cond" in x:
        m.addConstr(
            x["Bsmt Cond"] ==
            0*bc_bin["Po"] + 1*bc_bin["Fa"] + 2*bc_bin["TA"] + 3*bc_bin["Gd"] + 4*bc_bin["Ex"],
            name="BSMTCOND_link_int"
        )

    # si tu pipeline trae OHE de Bsmt Cond, inyecta dummies
    for nm in BC_LEVELS:
        col = f"Bsmt Cond_{nm}"
        if col in X_input.columns:
            if X_input[col].dtype != "O":
                X_input[col] = X_input[col].astype("object")
            X_input.loc[0, col] = bc_bin[nm]

    # costos: solo si subes por sobre la base y únicamente los niveles finales
    for nm, vb in bc_bin.items():
        if BC_ORD[nm] > bc_base:
            # el PDF cobra el costo del nivel elegido (no incremental por salto)
            lin_cost += ct.bsmt_cond_cost(nm) * vb

    # si la base ya es Gd/Ex, implícitamente quedas fijado a la base por el "no empeorar"
    # ================== FIN (BSMT COND) ==================

    # ================== (BSMT FIN TYPE1 / TYPE2) ==================
    BS_TYPES = ["GLQ","ALQ","BLQ","Rec","LwQ","Unf","NA"]

    # dummies de decisión desde features.py
    b1 = {nm: x.get(f"b1_is_{nm}") for nm in BS_TYPES}
    b2 = {nm: x.get(f"b2_is_{nm}") for nm in BS_TYPES}

    # helper para poner gp.Var en DataFrame
    def _put_var(df, col, var):
        if col in df.columns:
            if df[col].dtype != "O":
                df[col] = df[col].astype("object")
            df.loc[0, col] = var

    # base
    b1_base = str(base_row.get("BsmtFin Type 1", "NA")).strip()
    b2_base = str(base_row.get("BsmtFin Type 2", "NA")).strip()
    has_b2  = 0 if b2_base in ["NA", "None", "nan", "NaN"] else 1

    # (T1) selección única
    if all(v is not None for v in b1.values()):
        m.addConstr(gp.quicksum(b1.values()) == 1, name="B1_pick_one")

    # (T2) selección según existencia
    if all(v is not None for v in b2.values()):
        m.addConstr(gp.quicksum(b2.values()) == has_b2, name="B2_pick_hasB2")

    # Inyectar a X_input si tu XGB trae OHE "BsmtFin Type 1_*" y "BsmtFin Type 2_*"
    for nm, vb in b1.items():
        if vb is not None:
            _put_var(X_input, f"BsmtFin Type 1_{nm}", vb)
    for nm, vb in b2.items():
        if vb is not None:
            _put_var(X_input, f"BsmtFin Type 2_{nm}", vb)

    # Conjunto “Rec o peor”
    BAD = {"Rec","LwQ","Unf"}

    # Flags UpgB1 / UpgB2 (activas sólo si base ∈ BAD)
    upgB1 = m.addVar(vtype=gp.GRB.BINARY, name="B1_upg_flag")
    upgB2 = m.addVar(vtype=gp.GRB.BINARY, name="B2_upg_flag")

    is_bad1 = 1 if b1_base in BAD else 0
    is_bad2 = 1 if (has_b2 == 1 and b2_base in BAD) else 0

    # Fijo estas flags a la constante (equivalen a las activaciones del PDF)
    upgB1.LB = upgB1.UB = is_bad1
    upgB2.LB = upgB2.UB = is_bad2

    # Máscaras para “distinto de la base” M = 1 - Base
    M1 = {}
    M2 = {}
    for nm in BS_TYPES:
        M1[nm] = 0 if nm == b1_base else 1
        M2[nm] = 0 if nm == b2_base else 1

    # “Sólo puedes CAMBIAR si upg=1”  (sum_{b≠base} b1_is_b <= upgB1)
    if all(v is not None for v in b1.values()):
        m.addConstr(gp.quicksum(M1[nm]*b1[nm] for nm in BS_TYPES) <= upgB1, name="B1_change_if_upg")
    if has_b2 and all(v is not None for v in b2.values()):
        m.addConstr(gp.quicksum(M2[nm]*b2[nm] for nm in BS_TYPES) <= upgB2, name="B2_change_if_upg")

    # Dominio permitido (si NO upg → forzar base; si upg → prohibir categorías más baratas que la base)
    def _apply_allowed(bvars, b_base, upg_flag):
        if not bvars:
            return
        base_cost = ct.bsmt_type_cost(b_base)
        if upg_flag == 0:
            # sólo base
            for nm, vb in bvars.items():
                if vb is None: continue
                if nm == b_base:
                    vb.LB = 1.0; vb.UB = 1.0
                else:
                    vb.UB = 0.0
        else:
            # permitir mantener base o mejorar (>= costo base); nunca bajar
            for nm, vb in bvars.items():
                if vb is None: continue
                if ct.bsmt_type_cost(nm) < base_cost:
                    vb.UB = 0.0

    # Casos especiales NA
    if b1_base == "NA":
        # se mantiene NA sí o sí
        for nm, vb in b1.items():
            if vb is None: continue
            if nm == "NA": vb.LB = vb.UB = 1.0
            else: vb.UB = 0.0
    else:
        _apply_allowed(b1, b1_base, is_bad1)

    if has_b2 == 0:
        # Type2 no existe → todos 0
        for nm, vb in b2.items():
            if vb is None: continue
            vb.UB = 0.0
    elif b2_base == "NA":
        for nm, vb in b2.items():
            if vb is None: continue
            if nm == "NA": vb.LB = vb.UB = 1.0
            else: vb.UB = 0.0
    else:
        _apply_allowed(b2, b2_base, is_bad2)

    # Costos (sólo si cambias; usa máscara 1{b ≠ base})
    cost_b1 = gp.LinExpr(0.0)
    cost_b2 = gp.LinExpr(0.0)
    for nm, vb in b1.items():
        if vb is not None and M1[nm] == 1:
            cost_b1 += ct.bsmt_type_cost(nm) * vb
    for nm, vb in b2.items():
        if vb is not None and M2[nm] == 1:
            cost_b2 += ct.bsmt_type_cost(nm) * vb

    lin_cost += cost_b1 + cost_b2
    # ================== FIN (BSMT FIN TYPE1/2) ==================



    # -------------------
    # 5) Enlazar predictor (pre -> XGB) y pasar de log a precio
    # -------------------
    y_log = m.addVar(lb=-gp.GRB.INFINITY, name="y_log")
    _add_sklearn(m, bundle.pipe_for_gurobi(), X_input, [y_log])

    if bundle.is_log_target():
        y_price = m.addVar(lb=0.0, name="y_price")
        grid = np.linspace(10.0, 14.0, 81)
        m.addGenConstrPWL(y_log, y_price, grid.tolist(), np.expm1(grid).tolist(), name="exp_expm1")
    else:
        y_price = y_log

    # -------------------
    # 6) Costos lineales (numéricos + cocina + utilities + roof)
    # -------------------
    def _num_base(col: str) -> float:
        try:
            return float(pd.to_numeric(base_row.get(col), errors="coerce"))
        except Exception:
            return 0.0

    base_vals = {
        "Bedroom AbvGr": _num_base("Bedroom AbvGr"),
        "Full Bath": _num_base("Full Bath"),
        "Wood Deck SF": _num_base("Wood Deck SF"),
        "Garage Cars": _num_base("Garage Cars"),
        "Total Bsmt SF": _num_base("Total Bsmt SF"),
    }



    if "Bedroom AbvGr" in x:
        lin_cost += pos(x["Bedroom AbvGr"] - base_vals["Bedroom AbvGr"]) * ct.add_bedroom
    if "Full Bath" in x:
        lin_cost += pos(x["Full Bath"] - base_vals["Full Bath"]) * ct.add_bathroom
    if "Wood Deck SF" in x:
        lin_cost += pos(x["Wood Deck SF"] - base_vals["Wood Deck SF"]) * ct.deck_per_m2
    if "Garage Cars" in x:
        lin_cost += pos(x["Garage Cars"] - base_vals["Garage Cars"]) * ct.garage_per_car


    # Utilities (upgrade-only; costo solo si cambias)
    if "Utilities" in x:
        util_names = {0: "ELO", 1: "NoSeWa", 2: "NoSewr", 3: "AllPub"}
        util_to_ord = {"ELO": 0, "NoSeWa": 1, "NoSewr": 2, "AllPub": 3}

        u_new = x["Utilities"]
        u_base_name = str(base_row.get("Utilities"))
        try:
            u_base_ord = int(pd.to_numeric(base_row.get("Utilities"), errors="coerce"))
            if u_base_ord not in (0, 1, 2, 3):
                u_base_ord = util_to_ord.get(u_base_name, 0)
        except Exception:
            u_base_ord = util_to_ord.get(u_base_name, 0)

        u_bin = {
            0: m.addVar(vtype=gp.GRB.BINARY, name="util_ELO"),
            1: m.addVar(vtype=gp.GRB.BINARY, name="util_NoSeWa"),
            2: m.addVar(vtype=gp.GRB.BINARY, name="util_NoSewr"),
            3: m.addVar(vtype=gp.GRB.BINARY, name="util_AllPub"),
        }
        m.addConstr(gp.quicksum(u_bin.values()) == 1, name="UTIL_one_hot")
        m.addConstr(u_new == gp.quicksum(k * u_bin[k] for k in u_bin), name="UTIL_link")
        m.addConstr(u_new >= u_base_ord, name="UTIL_upgrade_only")  # solo mejorar

        for k, vb in u_bin.items():
            if k != u_base_ord:
                lin_cost += ct.util_cost(util_names[k]) * vb

    # Roof: costos (fuera del bloque de utilities)
    if s_bin or m_bin:
        base_style = str(base_row.get("Roof Style", "Gable"))
        base_matl  = str(base_row.get("Roof Matl",  "CompShg"))

        for sn, vb in s_bin.items():
            if sn != base_style:
                lin_cost += ct.roof_style_cost(sn) * vb  # costo fijo por cambiar estilo

        roof_area = float(pd.to_numeric(base_row.get("Gr Liv Area"), errors="coerce") or 0.0)
        for mn, vb in m_bin.items():
            if mn != base_matl:
                lin_cost += ct.roof_matl_cost(mn) * roof_area * vb  # $/ft2 * área

    # -------------------
    # 7) Presupuesto y objetivo
    # -------------------
    total_cost = lin_cost
    m.addConstr(total_cost <= budget, name="budget")
    m.setObjective(y_price - total_cost, gp.GRB.MAXIMIZE)

    # -------------------
    # 8) Resto de restricciones (R1..R8)
    # -------------------
    if "1st Flr SF" in x and "2nd Flr SF" in x:
        m.addConstr(x["1st Flr SF"] >= x["2nd Flr SF"], name="R1_floor1_ge_floor2")

    if "Gr Liv Area" in x and "Lot Area" in base_row:
        m.addConstr(x["Gr Liv Area"] <= float(base_row["Lot Area"]), name="R2_grliv_le_lot")

    if "1st Flr SF" in x and "Total Bsmt SF" in x:
        m.addConstr(x["1st Flr SF"] >= x["Total Bsmt SF"], name="R3_floor1_ge_bsmt")

    def _val_or_var(col):
        return x[col] if col in x else float(base_row[col])

    need = all(c in base_row for c in ["Full Bath", "Bedroom AbvGr"])
    if need and ("Half Bath" in base_row or "Half Bath" in x):
        fullb = _val_or_var("Full Bath")
        halfb = _val_or_var("Half Bath") if ("Half Bath" in x or "Half Bath" in base_row) else 0.0
        beds  = _val_or_var("Bedroom AbvGr")
        m.addConstr(fullb + halfb <= beds, name="R4_baths_le_bedrooms")

    if ("Full Bath" in x) or ("Full Bath" in base_row):
        m.addConstr(_val_or_var("Full Bath") >= 1, name="R5_min_fullbath")
    if ("Bedroom AbvGr" in x) or ("Bedroom AbvGr" in base_row):
        m.addConstr(_val_or_var("Bedroom AbvGr") >= 1, name="R5_min_bedrooms")
    if ("Kitchen AbvGr" in x) or ("Kitchen AbvGr" in base_row):
        m.addConstr(_val_or_var("Kitchen AbvGr") >= 1, name="R5_min_kitchen")

    lowqual_names = ["Low Qual Fin SF", "LowQualFinSF"]
    lowqual_col = next((c for c in lowqual_names if c in X_input.columns), None)
    cols_needed = ["Gr Liv Area", "1st Flr SF", "2nd Flr SF"]
    if all(c in X_input.columns for c in cols_needed):
        lhs = _val_or_var("Gr Liv Area")
        rhs = _val_or_var("1st Flr SF") + _val_or_var("2nd Flr SF")
        if lowqual_col is not None:
            rhs += _val_or_var(lowqual_col)
        m.addConstr(lhs == rhs, name="R7_gr_liv_equality")

    r8_ok = all(c in X_input.columns for c in ["TotRms AbvGrd", "Bedroom AbvGr", "Kitchen AbvGr"])
    if r8_ok:
        other = m.addVar(lb=0, ub=15, vtype=gp.GRB.INTEGER, name="R8_other_rooms")
        m.addConstr(_val_or_var("TotRms AbvGrd") == _val_or_var("Bedroom AbvGr") + _val_or_var("Kitchen AbvGr") + other,
                    name="R8_rooms_balance")

    return m
