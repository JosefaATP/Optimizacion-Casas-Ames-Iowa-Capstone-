#optimization/remodel/gurobi_model.py
from typing import Dict, Any
import pandas as pd
import gurobipy as gp
import numpy as np

from sklearn.compose import ColumnTransformer  # puede no usarse directo

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

# ======================== helpers base ========================

def _vtype(code: str):
    return gp.GRB.CONTINUOUS if code == "C" else (gp.GRB.BINARY if code == "B" else gp.GRB.INTEGER)

# getVarByName seguro
def _get_by_name_safe(m: gp.Model, *names: str):
    for nm in names:
        try:
            v = m.getVarByName(nm)
        except gp.GurobiError:
            m.update()
            v = m.getVarByName(nm)
        if v is not None:
            return v
    return None

# coerciones y mapeos ordinales
_ORD_MAP = {"Po":0, "Fa":1, "TA":2, "Gd":3, "Ex":4, "No aplica":-1, "NA":-1}
_UTIL_MAP = {"ELO":0, "NoSeWa":1, "NoSewr":2, "AllPub":3}

_ORDINAL_MINUS1 = {"Fireplace Qu", "Pool QC"}
_ORDINAL_0TO4   = {"Kitchen Qual", "Exter Qual", "Exter Cond", "Heating QC",
                   "Bsmt Qual", "Bsmt Cond", "Garage Qual", "Garage Cond"}

def _as_int_or_map(val, default=0, allow_minus1=True):
    try:
        v = int(pd.to_numeric(val, errors="coerce"))
        if allow_minus1:
            return v if v in (-1,0,1,2,3,4) else default
        else:
            return v if v in (0,1,2,3,4) else default
    except Exception:
        s = str(val).strip()
        if s in _ORD_MAP:
            v = _ORD_MAP[s]
            if (not allow_minus1) and v == -1:
                return default
            return v
        return default

def _na2noaplica(s: str) -> str:
    s = str(s).strip()
    return "No aplica" if s in {"NA", "No aplica"} else s

def _is_noaplica(v) -> bool:
    try:
        n = int(pd.to_numeric(v, errors="coerce"))
        if n == -1:
            return True
    except Exception:
        pass
    s = str(v).strip()
    return s in {"No aplica", "NA", "nan", "None", ""}


def _coerce_base_value(col: str, val):
    col = str(col)
    # ordinales
    if col in _ORDINAL_MINUS1:
        return float(_as_int_or_map(val, default=-1, allow_minus1=True))
    if col in _ORDINAL_0TO4:
        return float(_as_int_or_map(val, default=2, allow_minus1=False))
    # utilities
    if col == "Utilities":
        s = str(val).strip()
        if s in _UTIL_MAP:
            return float(_UTIL_MAP[s])
        try:
            v = int(pd.to_numeric(val, errors="coerce"))
            return float(v if v in (0,1,2,3) else 3)
        except Exception:
            return 3.0
    # booleanos Y/N simples
    if col in {"Central Air"}:
        s = str(val).strip().upper()
        return 1.0 if s in {"Y","YES","1","TRUE"} else 0.0
    # generico: numerico si se puede, si no -> 0.0
    s = str(val).strip()
    if s in {"No aplica","NA","nan","None",""}:
        return 0.0
    try:
        return float(pd.to_numeric(val, errors="coerce"))
    except Exception:
        return 0.0

def _norm_cat(s: str) -> str:
    s = str(s).strip()
    return "No aplica" if s in {"NA", "No aplica"} else s

# inyeccion generica de gp.Var al DataFrame del pipeline
def _put_var_obj(df: pd.DataFrame, col: str, var: gp.Var):
    if col in df.columns:
        if df[col].dtype != "O":
            df[col] = df[col].astype("object")
        df.loc[0, col] = var

# ordinal base helper
def _qual_base_ord(base_row, col: str) -> int:
    MAP = {"No aplica": -1, "Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4}
    val = str(base_row.get(col, "No aplica")).strip()
    try:
        nv = int(pd.to_numeric(val, errors="coerce"))
        if nv in (-1,0,1,2,3,4):
            return nv
    except Exception:
        pass
    return MAP.get(val, -1)

# fila de entrada numerica exacta para el regressor
def build_base_input_row(bundle: XGBBundle, base_row: pd.Series) -> pd.DataFrame:
    cols = list(bundle.feature_names_in())
    row = {}
    for c in cols:
        if "_" in c and (c.count("_") >= 1):
            root, cat = c.split("_", 1)
            root = root.replace("_", " ").strip()
            cat  = cat.strip()
            base_val = base_row.get(root, None)
            if root == "Central Air":
                base_cat = "Y" if str(base_val).strip().upper() in {"Y","YES","1","TRUE"} else "N"
            else:
                base_cat = _norm_cat(base_val)
            row[c] = 1.0 if _norm_cat(base_cat) == _norm_cat(cat) else 0.0
        else:
            row[c] = float(_coerce_base_value(c, base_row.get(c, 0)))
    return pd.DataFrame([row], columns=cols, dtype=float)

FORBID_BUILD_WHEN_NA = {"Fireplace Qu","Bsmt Cond","Garage Qual","Garage Cond","Pool QC"}

def apply_quality_policy_ordinal(m, x: dict, base_row: pd.Series, col: str):
    q = x.get(col)
    if q is None:
        return

    # => NA robusto: acepta -1 numérico o "No aplica"/"NA" string
    raw = base_row.get(col, "No aplica")
    base_is_na = _is_noaplica(raw)  # usa tu helper robusto
    base_ord = _qual_base_ord(base_row, col)  # -1..4

    # Si tu política prohíbe “construir desde NA”, fíjalo en -1
    if base_is_na and col in FORBID_BUILD_WHEN_NA:
        q.LB = -1.0
        q.UB = -1.0
        return

    # Si existe (>=0), solo upgrade-only (no bajar)
    if base_ord >= 0:
        m.addConstr(q >= base_ord, name=f"{col.replace(' ','_')}_upgrade_only")


def _is_noaplica(val) -> bool:
    s = str(val).strip()
    if s in {"No aplica", "NA", "", "None"}:
        return True
    try:
        v = int(pd.to_numeric(val, errors="coerce"))
        return v == -1
    except Exception:
        return False

# ======================== modelo principal ========================

def build_mip_embed(base_row: pd.Series, budget: float, ct: CostTables, bundle: XGBBundle,
                    base_price=None, fix_to_base: bool = False) -> gp.Model:
    m = gp.Model("remodel_embed")

    # presupuesto
    try:
        budget_usd = float(budget)
        if not np.isfinite(budget_usd) or budget_usd <= 0:
            print(f"[WARN] Presupuesto invalido o no positivo ({budget}), fallback 40000")
            budget_usd = 40000.0
    except Exception:
        print(f"[WARN] Presupuesto no numerico ({type(budget)}={budget}), fallback 40000")
        budget_usd = 40000.0
    m._budget_usd = float(budget_usd)
    m._budget = float(budget_usd)

    # base_X numerico y base_price consistente
    base_X = build_base_input_row(bundle, base_row)

    # --- DEBUG: base_X no debe traer NaN/Inf
    try:
        _bad_cols = []
        for c in base_X.columns:
            v = base_X.iloc[0][c]
            try:
                vf = float(v)
            except Exception:
                # si por algún motivo no es float (no debería en base_X), márcalo
                _bad_cols.append((c, v))
                continue
            if not np.isfinite(vf):
                _bad_cols.append((c, v))
        if _bad_cols:
            print("[DEBUG] base_X trae valores no finitos en:")
            for c, v in _bad_cols[:25]:
                print(f"   - {c}: {v} | raw base={base_row.get(c, None)}")
            # Si prefieres, puedes sanear:
            base_X = base_X.fillna(0.0)
    except Exception as e:
        print(f"[DEBUG] chequeo base_X falló: {e}")

    if base_price is None:
        try:
            base_price = float(bundle.predict(base_X).iloc[0])
        except Exception:
            try:
                raw_df = pd.DataFrame([base_row])
                base_price = float(bundle.predict(raw_df).iloc[0])
            except Exception:
                base_price = 0.0

    # 1) variables de decision
    x: Dict[str, gp.Var] = {}
    for f in MODIFIABLE:
        x[f.name] = m.addVar(lb=f.lb, ub=f.ub, vtype=_vtype(f.vartype), name=f"x_{f.name}")

    lin_cost = gp.LinExpr(0.0)
    m._lin_cost_expr = lin_cost

    # construir indice de nombres
    m.update()

    # 2) armar X_input consistente con el pipeline
    feature_order = bundle.feature_names_in()
    modif = {f.name for f in MODIFIABLE}

    base_X = build_base_input_row(bundle, base_row)

    row_vals: Dict[str, Any] = {}
    for fname in feature_order:
        if fname in modif:
            row_vals[fname] = x[fname]
            continue
        try:
            if fname in base_X.columns:
                v = base_X.iloc[0].loc[fname]
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    row_vals[fname] = float(v)
                    continue
        except Exception:
            pass
        raw_val = base_row.get(fname, None)
        if str(fname) == "Central Air":
            row_vals[fname] = "Y" if str(raw_val).strip().upper() in {"Y","YES","1","TRUE"} else "N"
            continue
        try:
            num = pd.to_numeric(raw_val, errors="coerce")
            if not pd.isna(num):
                row_vals[fname] = float(num)
                continue
        except Exception:
            pass
        if fname in _ORDINAL_MINUS1 or fname in _ORDINAL_0TO4 or fname == "Utilities":
            row_vals[fname] = _coerce_base_value(fname, raw_val)
            continue
        row_vals[fname] = _norm_cat(raw_val)

    X_input = pd.DataFrame([row_vals], columns=feature_order, dtype=object)
    m._X_input = X_input
    m._X_base_numeric = base_X

    # chequeo de encoding para columnas no modificables
    try:
        mismatches = []
        for fname in feature_order:
            if fname in modif:
                continue
            try:
                v_base = float(base_X.iloc[0].loc[fname])
            except Exception:
                v_base = float(_coerce_base_value(fname, base_row.get(fname, 0)))
            try:
                v_in = X_input.iloc[0].loc[fname]
                v_in_f = float(v_in)
            except Exception:
                v_in_f = None
            if v_in_f is None or (not np.isfinite(v_in_f)) or abs(v_base - v_in_f) > 1e-8:
                mismatches.append((fname, v_base, v_in))
        if mismatches:
            print(f"[WARN] Encoding mismatch en {len(mismatches)} columnas no-modificables. Auto-fix a base.")
            for fname, v_base, _ in mismatches:
                X_input.iloc[0, X_input.columns.get_loc(fname)] = float(v_base)
            m._X_input = X_input
            m._encoding_ok = False
        else:
            m._encoding_ok = True
    except Exception as e:
        print(f"[WARN] Encoding check fallo: {e}")
        m._encoding_ok = False

    # === calidad general: propaga upgrades de calidades específicas a Overall Qual ===
    try:
        base_overall = None
        if "Overall Qual" in base_X.columns:
            base_overall = float(pd.to_numeric(base_X.iloc[0].get("Overall Qual"), errors="coerce"))
        if base_overall is None or not np.isfinite(base_overall):
            base_overall = float(pd.to_numeric(base_row.get("Overall Qual", 5), errors="coerce") or 5.0)
        if not np.isfinite(base_overall):
            base_overall = 5.0

        QUAL_IMPACT_COLS = [
            "Kitchen Qual", "Exter Qual", "Exter Cond", "Heating QC",
            "Fireplace Qu", "Bsmt Cond", "Garage Qual", "Garage Cond", "Pool QC",
        ]
        improv_terms = []
        for col in QUAL_IMPACT_COLS:
            q_var = x.get(col)
            if q_var is None:
                continue
            base_ord = _qual_base_ord(base_row, col)
            if base_ord < 0:
                continue
            try:
                b = float(base_ord)
            except Exception:
                b = 0.0
            # normaliza el salto a [0,1] dividiendo por 4 (Po..Ex) y trunca en 0
            improv_terms.append((q_var - b) / 4.0)

        if improv_terms and "Overall Qual" in X_input.columns:
            n_cols = len(improv_terms)
            # Overall Qual debe ser INTEGER redondeado hacia ABAJO (floor)
            overall_qual_cont = m.addVar(lb=1.0, ub=10.0, vtype=gp.GRB.CONTINUOUS, name="Overall_Qual_float")
            overall_var = m.addVar(lb=1.0, ub=10.0, vtype=gp.GRB.INTEGER, name="Overall_Qual_calc")
            
            avg_delta = (1.0 / n_cols) * gp.quicksum(improv_terms)
            max_boost = 2.0  # si todas las calidades pasan de Po->Ex, suma hasta +2 en OverallQual
            
            # Valor continuo antes de redondear
            m.addConstr(
                overall_qual_cont == float(base_overall) + max_boost * avg_delta,
                name="OverallQual_from_upgrades_float",
            )
            
            # Redondea hacia abajo (floor): overall_var >= overall_qual_cont - 1 AND overall_var <= floor(overall_qual_cont)
            # En Gurobi, hacemos: overall_var >= overall_qual_cont - 1 (siempre se cumple si es floor)
            #                     overall_var <= overall_qual_cont (fuerza a ser <= el valor continuo)
            m.addConstr(overall_var <= overall_qual_cont, name="OverallQual_floor_ub")
            m.addConstr(overall_var >= overall_qual_cont - 1.0, name="OverallQual_floor_lb")
            
            X_input.loc[0, "Overall Qual"] = overall_var
            m._overall_qual_var = overall_var
    except Exception as e:
        print(f"[WARN] no se pudo actualizar Overall Qual con mejoras: {e}")

    # fijar todo a base si se pide
    if fix_to_base:
        for fname in (f.name for f in MODIFIABLE):
            var = x.get(fname)
            if var is None:
                continue
            try:
                base_val = float(base_X.iloc[0].loc[fname])
            except Exception:
                base_val = _coerce_base_value(fname, base_row.get(fname, 0))
            try:
                if not np.isfinite(base_val):
                    base_val = 0.0
            except Exception:
                base_val = 0.0
            var.LB = base_val
            var.UB = base_val

    # politicas de ordinales upgrade-only
    for col in ["Kitchen Qual","Exter Qual","Exter Cond","Heating QC",
                "Fireplace Qu","Bsmt Cond","Garage Qual","Garage Cond","Pool QC"]:
        apply_quality_policy_ordinal(m, x, base_row, col)

    # ========= KITCHEN QUAL =========
    KITCH_LEVELS = ["Po","Fa","TA","Gd","Ex"]
    ORD = {"Po":0,"Fa":1,"TA":2,"Gd":3,"Ex":4}

    kq_base_txt = str(base_row.get("Kitchen Qual","TA")).strip()
    kq_base = ORD.get(kq_base_txt, 2)

    kit_bins = {nm: m.addVar(vtype=gp.GRB.BINARY, name=f"kit_is_{nm}") for nm in KITCH_LEVELS}
    m.addConstr(gp.quicksum(kit_bins.values()) == 1, name="KIT_onehot")

    for nm, v in kit_bins.items():
        if ORD[nm] < kq_base:
            v.UB = 0.0

    if "Kitchen Qual" in x:
        m.addConstr(
            x["Kitchen Qual"] == 0*kit_bins["Po"] + 1*kit_bins["Fa"] + 2*kit_bins["TA"] + 3*kit_bins["Gd"] + 4*kit_bins["Ex"],
            name="KIT_link_int"
        )

    kit_cost = gp.LinExpr(0.0)
    for nm, vb in kit_bins.items():
        if ORD[nm] > kq_base:
            kit_cost += ct.kitchen_level_cost(nm) * vb
    lin_cost += kit_cost

    # ================== EXTERIOR ==================
    EXT_MATS = [
        "AsbShng","AsphShn","BrkComm","BrkFace","CBlock","CemntBd","HdBoard","ImStucc",
        "MetalSd","Other","Plywood","PreCast","Stone","Stucco","VinylSd","Wd Sdng","WdShngl",
    ]

    ex1 = {nm: x.get(f"ex1_is_{nm}") for nm in EXT_MATS if f"ex1_is_{nm}" in x}
    ex2 = {nm: x.get(f"ex2_is_{nm}") for nm in EXT_MATS if f"ex2_is_{nm}" in x}

    ex2_base_name_raw = str(base_row.get("Exterior 2nd", "None"))
    Ilas2 = 0 if (ex2_base_name_raw in ["None","nan","NaN","NoneType","0"] or pd.isna(base_row.get("Exterior 2nd"))) else 1

    if ex1:
        m.addConstr(gp.quicksum(ex1.values()) == 1, name="EXT_ex1_pick_one")
    if ex2:
        m.addConstr(gp.quicksum(ex2.values()) == Ilas2, name="EXT_ex2_pick_ilas2")

    for nm in EXT_MATS:
        col1 = f"Exterior 1st_{nm}"
        if col1 in X_input.columns and nm in ex1:
            _put_var_obj(X_input, col1, ex1[nm])
        col2 = f"Exterior 2nd_{nm}"
        if col2 in X_input.columns and nm in ex2:
            _put_var_obj(X_input, col2, ex2[nm])

    ex1_base_name = str(base_row.get("Exterior 1st", "None")).strip()
    ex2_base_name = str(base_row.get("Exterior 2nd", "None")).strip()

    def _q_to_ord(v):
        M = {"Po":0,"Fa":1,"TA":2,"Gd":3,"Ex":4}
        try:
            return int(pd.to_numeric(v, errors="coerce"))
        except Exception:
            return M.get(str(v).strip(), 2)

    exq_base_ord = _q_to_ord(base_row.get("Exter Qual", "TA"))
    exc_base_ord = _q_to_ord(base_row.get("Exter Cond", "TA"))
    eligible = 1 if (exq_base_ord <= 2 or exc_base_ord <= 2) else 0

    if eligible == 0:
        if ex1_base_name in ex1:
            for nm, v in ex1.items():
                v.UB = 1.0 if nm == ex1_base_name else 0.0
        if Ilas2 == 1 and ex2:
            if ex2_base_name in ex2:
                for nm, v in ex2.items():
                    v.UB = 1.0 if nm == ex2_base_name else 0.0

    # COSTO ABSOLUTO para materiales (no incremental)
    for nm, vb in ex1.items():
        lin_cost += ct.ext_mat_cost(nm) * vb
    if Ilas2 == 1:
        for nm, vb in ex2.items():
            lin_cost += ct.ext_mat_cost(nm) * vb

    EQ_LEVELS = ["Po","Fa","TA","Gd","Ex"]
    ORD = {"Po":0,"Fa":1,"TA":2,"Gd":3,"Ex":4}

    eq_bin = {nm: m.addVar(vtype=gp.GRB.BINARY, name=f"exterqual_is_{nm}") for nm in EQ_LEVELS}
    ec_bin = {nm: m.addVar(vtype=gp.GRB.BINARY, name=f"extercond_is_{nm}") for nm in EQ_LEVELS}
    m.addConstr(gp.quicksum(eq_bin.values()) == 1, name="EXT_EQ_onehot")
    m.addConstr(gp.quicksum(ec_bin.values()) == 1, name="EXT_EC_onehot")

    for nm in EQ_LEVELS:
        if ORD[nm] < exq_base_ord: eq_bin[nm].UB = 0.0
        if ORD[nm] < exc_base_ord: ec_bin[nm].UB = 0.0

    if eligible == 0:
        for nm, vb in eq_bin.items():
            if ORD[nm] == exq_base_ord: vb.LB = vb.UB = 1.0
            else: vb.UB = 0.0
        for nm, vb in ec_bin.items():
            if ORD[nm] == exc_base_ord: vb.LB = vb.UB = 1.0
            else: vb.UB = 0.0

    if "Exter Qual" in x:
        m.addConstr(x["Exter Qual"] == 0*eq_bin["Po"] + 1*eq_bin["Fa"] + 2*eq_bin["TA"] + 3*eq_bin["Gd"] + 4*eq_bin["Ex"],
                    name="EXT_EQ_link_int")
    if "Exter Cond" in x:
        m.addConstr(x["Exter Cond"] == 0*ec_bin["Po"] + 1*ec_bin["Fa"] + 2*ec_bin["TA"] + 3*ec_bin["Gd"] + 4*ec_bin["Ex"],
                    name="EXT_EC_link_int")

    for nm in EQ_LEVELS:
        col = f"Exter Qual_{nm}"
        if col in X_input.columns:
            _put_var_obj(X_input, col, eq_bin[nm])
    for nm in EQ_LEVELS:
        col = f"Exter Cond_{nm}"
        if col in X_input.columns:
            _put_var_obj(X_input, col, ec_bin[nm])

    for nm, vb in eq_bin.items():
        if ORD[nm] > exq_base_ord:
            lin_cost += (ct.exter_qual_cost(nm) - ct.exter_qual_cost(EQ_LEVELS[exq_base_ord])) * vb
    for nm, vb in ec_bin.items():
        if ORD[nm] > exc_base_ord:
            lin_cost += (ct.exter_cond_cost(nm) - ct.exter_cond_cost(EQ_LEVELS[exc_base_ord])) * vb


    # FIXED: Add path exclusion constraint - Material change XOR Quality upgrade (per specification)
    # Specification: se pueden seguir dos caminos implies paths are mutually exclusive
    if eligible == 1:
        UpgMat_ext = m.addVar(vtype=gp.GRB.BINARY, name="ext_upg_material")
        UpgQC_ext = m.addVar(vtype=gp.GRB.BINARY, name="ext_upg_qc")
        m.addConstr(UpgMat_ext + UpgQC_ext <= 1, name="EXT_exclusive_paths")

    # ================== MAS VNR (robusto, sin “bajar” a No aplica) ==================
    MV_CANDIDATES = ["BrkCmn", "BrkFace", "CBlock", "Stone", "No aplica", "NA", "None"]

    def _norm(s: str) -> str:
        return str(s).strip().lower().replace(" ", "").replace("_", "")

    NA_KEYS = {"noaplica", "na", "none"}

    def _gv(name: str):
        v = m.getVarByName(name)
        if v is not None:
            return v
        return x.get(name)

    def _pick_mvt_var(nm: str):
        # prueba con y sin prefijo x_
        for cand in (f"x_mvt_is_{nm}", f"mvt_is_{nm}"):
            v = _gv(cand)
            if v is not None:
                return v
        return None

    # 1) Construye el mapa tipo->var binaria (acepta varios alias)
    mvt_raw = {}
    for nm in MV_CANDIDATES:
        v = _pick_mvt_var(nm)
        if v is not None:
            mvt_raw[nm] = v

    if mvt_raw:
        m.addConstr(gp.quicksum(mvt_raw.values()) == 1, name="MVT_pick_one")

    # 2) Ata TODAS las columnas de X_input que empiezan con "Mas Vnr Type_"
    #    al binario correcto (coincidencia por sufijo, robusta a espacios/case)
    for col in [c for c in X_input.columns if c.startswith("Mas Vnr Type_")]:
        suf = col.split("Mas Vnr Type_", 1)[1]
        # busca var por nombre exacto o por normalización
        v = mvt_raw.get(suf)
        if v is None:
            # intenta por normalización
            target = _norm(suf)
            for nm, vv in mvt_raw.items():
                if _norm(nm) == target:
                    v = vv
                    break
        if v is not None:
            _put_var_obj(X_input, col, v)

    # 3) Lee base y costos
    mvt_base_txt = str(base_row.get("Mas Vnr Type", "No aplica")).strip()
    base_key     = _norm(mvt_base_txt)
    try:
        mv_area_base = float(pd.to_numeric(base_row.get("Mas Vnr Area"), errors="coerce") or 0.0)
    except Exception:
        mv_area_base = 0.0

    def _cost(nm: str) -> float:
        try:
            c = float(pd.to_numeric(ct.mas_vnr_cost(nm), errors="coerce"))
            return c if pd.notna(c) else 0.0
        except Exception:
            return 0.0

    base_cost = _cost(mvt_base_txt)
    mv_area   = _gv("x_Mas Vnr Area") or _gv("Mas Vnr Area") or mv_area_base

    # 4) Política:
    #    - Si NO había veneer (No aplica/NA/None o área≈0): quedarse en “No aplica” y área=0.
    #    - Si SÍ había veneer: prohibido “No aplica” y prohibidos tipos con costo < base.
    had_veneer = (mv_area_base > 1e-9) and (base_key not in NA_KEYS)

    if not had_veneer:
        # forzar No aplica = 1, demás = 0, área 0
        for nm, v in mvt_raw.items():
            if _norm(nm) in NA_KEYS:
                m.addConstr(v == 1, name="MVT_stay_noaplica")
            else:
                m.addConstr(v == 0, name=f"MVT_forbid_{nm}")
        if isinstance(mv_area, gp.Var):
            mv_area.LB = 0.0
            mv_area.UB = 0.0
    else:
        # prohíbe No aplica
        for nm, v in mvt_raw.items():
            if _norm(nm) in NA_KEYS:
                m.addConstr(v == 0, name="MVT_forbid_noaplica")
        # prohíbe opciones más baratas que la base
        for nm, v in mvt_raw.items():
            if _norm(nm) not in NA_KEYS and _cost(nm) < base_cost:
                m.addConstr(v == 0, name=f"MVT_forbid_cheaper_{nm}")
        # área no puede bajar y (por ahora) la fijamos a la base para evitar upgrades gratis
        if isinstance(mv_area, gp.Var):
            m.addConstr(mv_area >= mv_area_base, name="MVT_area_no_decrease")
            mv_area.LB = mv_area_base
            mv_area.UB = mv_area_base

    # 5) Costo lineal: solo si CAMBIA el tipo (área * costo_nuevo)
    try:
        ub_candidate = float(mv_area_base if mv_area_base > 0 else 0.0)
    except Exception:
        ub_candidate = 0.0
    UB_DEFAULT = 2000.0
    mv_area_ub = max(ub_candidate, UB_DEFAULT)

    if isinstance(mv_area, gp.Var):
        for nm, v in mvt_raw.items():
            if _norm(nm) not in NA_KEYS and nm != mvt_base_txt and v is not None:
                p = m.addVar(lb=0.0, ub=mv_area_ub, name=f"mv_area_mvt_{nm}")
                m.addConstr(p <= mv_area,                      name=f"MVT_lin_p_le_a_{nm}")
                m.addConstr(p <= mv_area_ub * v,               name=f"MVT_lin_p_le_UBb_{nm}")
                m.addConstr(p >= mv_area - mv_area_ub * (1-v), name=f"MVT_lin_p_ge_a_minus_UB1b_{nm}")
                lin_cost += (_cost(nm) - base_cost) * p
    else:
        area_term = float(mv_area)
        for nm, v in mvt_raw.items():
            if _norm(nm) not in NA_KEYS and nm != mvt_base_txt and v is not None:
                lin_cost += (_cost(nm) - base_cost) * area_term * v

    # 6) Log de sanidad
    try:
        msg_allowed = ", ".join([nm for nm,v in mvt_raw.items() if v.UB > 0.5 or hasattr(v, "LB") and v.LB > 0.5])
        print(f"[CHK-MVT] Base={mvt_base_txt} area_base={mv_area_base} | allowed={msg_allowed}")
    except Exception:
        pass

    # ================== ROOF ==================
    style_names = ["Flat", "Gable", "Gambrel", "Hip", "Mansard", "Shed"]
    matl_names  = ["ClyTile", "CompShg", "Membran", "Metal", "Roll", "Tar&Grv", "WdShake", "WdShngl"]

    s_bin = {nm: x.get(f"roof_style_is_{nm}") for nm in style_names if f"roof_style_is_{nm}" in x}
    m_bin = {nm: x.get(f"roof_matl_is_{nm}")  for nm in matl_names  if f"roof_matl_is_{nm}"  in x}

    base_style = str(base_row.get("RoofStyle", base_row.get("Roof Style", "")))
    base_mat   = str(base_row.get("RoofMatl",  base_row.get("Roof Matl",  "")))

    # estilo fijo si existe
    if s_bin and base_style in s_bin:
        s_bin[base_style].LB = s_bin[base_style].UB = 1.0
        for nm, var in s_bin.items():
            if nm != base_style:
                var.UB = 0.0
        m.addConstr(gp.quicksum(s_bin.values()) == 1, name="ROOF_style_fixed_pick_one")

    # materials universo completo
    all_m_bin = {}
    for nm in matl_names:
        var_name = f"roof_matl_is_{nm}"
        if var_name in x:
            all_m_bin[nm] = x[var_name]
        else:
            v = m.addVar(vtype=gp.GRB.BINARY, name=var_name)
            if base_mat and nm == base_mat:
                v.LB = v.UB = 1.0
            else:
                v.LB = v.UB = 0.0
            all_m_bin[nm] = v
    m.addConstr(gp.quicksum(all_m_bin.values()) == 1, name="ROOF_pick_one_matl")

    ROOF_FORBIDS = {
        "Gable":   ["Membran"],
        "Hip":     ["Membran"],
        "Flat":    ["WdShngl", "ClyTile", "CompShg"],
        "Mansard": ["Membran"],
        "Shed":    ["ClyTile"],
        "Gambrel": [],
    }
    if m_bin and base_style:
        for mn in ROOF_FORBIDS.get(base_style, []):
            if mn in m_bin:
                m.addConstr(m_bin[mn] == 0.0, name=f"ROOF_incompat_{base_style}_{mn}")

    for nm in matl_names:
        col = f"Roof Matl_{nm}"
        if col in X_input.columns and nm in all_m_bin:
            _put_var_obj(X_input, col, all_m_bin[nm])

    cost_roof = gp.LinExpr(0.0)
    for mat, y in all_m_bin.items():
        try:
            mat_cost = float(ct.get_roof_matl_cost(mat))
        except Exception:
            mat_cost = 0.0
        if (not base_mat) or mat != base_mat:
            cost_roof += mat_cost * y

    if base_mat and float(ct.roof_demo_cost) != 0.0:
        non_base_vars = [y for nm, y in all_m_bin.items() if nm != base_mat]
        if non_base_vars:
            roof_change = m.addVar(vtype=gp.GRB.BINARY, name="roof_change_mat")
            m.addConstr(roof_change == gp.quicksum(non_base_vars), name="ROOF_change_eq_sum_nonbase")
            cost_roof += float(ct.roof_demo_cost) * roof_change

    lin_cost += cost_roof

    # ========== GARAGE FINISH ==========
    GF = ["Fin", "RFn", "Unf", "No aplica"]

    # asegurar que exista la binaria base si falta
    def _ensure_fixed_bin(name: str, set_one: bool) -> gp.Var:
        v = m.addVar(vtype=gp.GRB.BINARY, name=name)
        v.LB = v.UB = 1.0 if set_one else 0.0
        return v

    gar = {g: x.get(f"garage_finish_is_{g}") for g in GF}

    # base one-hot robusto
    def _gf_base_dummy(tag: str) -> float:
        for label in ["No aplica", "NA"]:
            col = f"Garage Finish_{label if tag=='No aplica' else tag}"
            if col in base_row:
                return float(base_row.get(col, 0.0) or 0.0)
        if "Garage Finish" in base_row:
            txt = str(base_row["Garage Finish"]).strip()
            if txt in {"NA", "No aplica"} and tag == "No aplica":
                return 1.0
            if txt == tag:
                return 1.0
        return 0.0

    BaseGa = {g: _gf_base_dummy(g) for g in GF}

    # crea fija para la categoria base si la var falta
    for g in GF:
        if gar[g] is None:
            need_one = (BaseGa[g] == 1.0)
            gar[g] = _ensure_fixed_bin(f"garage_finish_is_{g}", need_one)
            # inyectar OHE si existe
            if g == "No aplica":
                for col in ["Garage Finish_No aplica", "Garage Finish_NA"]:
                    if col in X_input.columns:
                        _put_var_obj(X_input, col, gar[g])
            else:
                col = f"Garage Finish_{g}"
                if col in X_input.columns:
                    _put_var_obj(X_input, col, gar[g])

    # pick-one sobre el universo
    m.addConstr(gp.quicksum(v for v in gar.values() if v is not None) == 1.0, name="GaFin_pick_one")

    # elegibilidad y restricciones (con UpgGarage si existe)
    UpgGa = x.get("UpgGarage")
    try:
        gt   = str(base_row.get("Garage Type", "No aplica")).strip()
        area = float(pd.to_numeric(base_row.get("Garage Area"), errors="coerce") or 0.0)
        cars = float(pd.to_numeric(base_row.get("Garage Cars"), errors="coerce") or 0.0)
        has_garage = (gt not in {"NA","No aplica"}) or (area > 0) or (cars > 0)
    except Exception:
        has_garage = False

    if has_garage and gar.get("No aplica") is not None:
        gar["No aplica"].UB = 0.0

    if BaseGa["No aplica"] == 1.0:
        for g, v in gar.items():
            v.UB = 1.0 if g == "No aplica" else 0.0
        if UpgGa is not None: m.addConstr(UpgGa == 0.0, name="UpgGa_off_for_NA")
    elif BaseGa["Fin"] == 1.0:
        for g, v in gar.items():
            v.UB = 1.0 if g == "Fin" else 0.0
        if UpgGa is not None: m.addConstr(UpgGa == 0.0, name="UpgGa_off_for_Fin")
    else:
        if UpgGa is not None:
            m.addConstr(UpgGa <= BaseGa["RFn"] + BaseGa["Unf"], name="UpgGa_allowed_if_eligible")
            # cambio solo si UpgGa=1
            mask = {g: 1.0 - BaseGa[g] for g in GF}
            m.addConstr(gp.quicksum(mask[g] * gar[g] for g in GF) <= UpgGa, name="GaFin_change_only_if_upg")
            if gar.get("Fin") is not None:
                m.addConstr(gar["Fin"] <= UpgGa, name="GaFin_Fin_requires_Upg")
            m.addConstr(gp.quicksum(gar[g] for g in ["RFn","Unf"]) <= 1.0 - UpgGa, name="GaFin_RFnUnf_disallowed_when_upg")
        else:
            ORDER = ["Unf","RFn","Fin"]
            base_cat = next((g for g in ORDER if BaseGa[g] == 1.0), None)
            if base_cat:
                base_idx = ORDER.index(base_cat)
                for i, g in enumerate(ORDER):
                    if i < base_idx and g in gar:
                        gar[g].UB = 0.0

    lin_cost += gp.quicksum(float(ct.garage_finish_cost(g)) * (1.0 - BaseGa[g]) * gar[g] for g in GF)

    # ================== (CENTRAL AIR) ==================
    base_air_raw = str(base_row.get("Central Air", "N")).strip()
    base_is_Y = base_air_raw in {"Y", "Yes", "1", "True"}

    col_Y = "Central Air_Y"
    col_N = "Central Air_N"
    has_Y = col_Y in X_input.columns
    has_N = col_N in X_input.columns

    if has_Y or has_N:
        air_yes = m.addVar(vtype=gp.GRB.BINARY, name="central_air_yes")

        if base_is_Y:
            air_yes.LB = air_yes.UB = 1.0
        else:
            lin_cost += ct.central_air_install * air_yes

        air_no = None
        if has_N:
            air_no = m.addVar(vtype=gp.GRB.BINARY, name="central_air_no")
            m.addConstr(air_yes + air_no == 1, name="CentralAir_onehot")

        if has_Y:
            _put_var_obj(X_input, col_Y, air_yes)
        if has_N:
            _put_var_obj(X_input, col_N, air_no)

    # ================== POOL QC ==================
    PQC = ["Po", "Fa", "TA", "Gd", "Ex", "No aplica"]
    PQ_ORD = {"Po":0, "Fa":1, "TA":2, "Gd":3, "Ex":4, "No aplica":-1}
    P_LEQ_AV = {"Po","Fa","TA"}

    def _pq_cost(name: str) -> float:
        try:
            return float(ct.poolqc_cost(name))
        except Exception:
            try:
                return float(getattr(ct, "poolqc_costs", {}).get(name, 0.0))
            except Exception:
                return 0.0

    pq_base_raw = base_row.get("Pool QC", "No aplica")
    base_pq_is_na = _is_noaplica(pq_base_raw)
    pq_base_txt = "No aplica" if base_pq_is_na else str(pq_base_raw).strip()
    pq_base_ord = -1 if base_pq_is_na else PQ_ORD.get(pq_base_txt, 2)
    base_cost = 0.0 if base_pq_is_na else _pq_cost(pq_base_txt)

    pq_ord = x.get("Pool QC", None)

    pq = {nm: m.addVar(vtype=gp.GRB.BINARY, name=f"poolqc_is_{nm}") for nm in PQC}
    m.addConstr(gp.quicksum(pq.values()) == 1.0, name="PoolQC_pick_one")

    if pq_ord is not None:
        m.addConstr(
            pq_ord == (-1)*pq["No aplica"] + 0*pq["Po"] + 1*pq["Fa"] + 2*pq["TA"] + 3*pq["Gd"] + 4*pq["Ex"],
            name="PoolQC_link_ord"
        )

    UpgPool = m.addVar(vtype=gp.GRB.BINARY, name="UpgPool")
    eligible = 0 if base_pq_is_na else (1 if pq_base_txt in P_LEQ_AV else 0)
    m.addConstr(UpgPool <= eligible, name="PoolQC_eligibility")

    if base_pq_is_na:
        pq["No aplica"].LB = 1.0
        for nm in ["Po","Fa","TA","Gd","Ex"]:
            pq[nm].UB = 0.0
        UpgPool.LB = 0.0; UpgPool.UB = 0.0
    else:
        pq["No aplica"].UB = 0.0
        # no empeorar
        for nm in ["Po","Fa","TA","Gd","Ex"]:
            if PQ_ORD[nm] < pq_base_ord:
                pq[nm].UB = 0.0
        # stay in base si no hay upgrade
        m.addConstr(pq[pq_base_txt] >= 1.0 - UpgPool, name="PoolQC_stay_base_if_no_upg")
        # sólo niveles con costo >= base si hay upgrade
        for nm in ["Po","Fa","TA","Gd","Ex"]:
            if nm != pq_base_txt:
                if _pq_cost(nm) >= base_cost:
                    m.addConstr(pq[nm] <= UpgPool, name=f"PoolQC_allow_if_upg_{nm}")
                else:
                    pq[nm].UB = 0.0

    for nm in ["Po","Fa","TA","Gd","Ex"]:
        if not base_pq_is_na and nm != pq_base_txt:
            lin_cost += (_pq_cost(nm) - base_cost) * pq[nm]

    for nm in PQC:
        col = f"Pool QC_{nm}"
        if col in X_input.columns:
            _put_var_obj(X_input, col, pq[nm])


    # ================== ÁREA LIBRE Y AMPLIACIONES ==================
    A_Full, A_Half, A_Kitch, A_Bed = 40.0, 20.0, 75.0, 70.0

    AddFull  = x.get("AddFull", None)
    AddHalf  = x.get("AddHalf", None)
    AddKitch = x.get("AddKitch", None)
    AddBed   = x.get("AddBed", None)

    COMPONENTES = [
        "Garage Area", "Wood Deck SF", "Open Porch SF",
        "Enclosed Porch", "3Ssn Porch", "Screen Porch", "Pool Area"
    ]
    # ======== CREATE Z-VARIABLES FOR AREA EXPANSIONS ========
    # z[c][s] = binary variable indicating s% expansion of component c
    # Must be created as gurobi.Var objects BEFORE being used in constraints
    
    for c in COMPONENTES:
        for s in [10, 20, 30]:
            var_name = f"z{s}_{c.replace(' ', '')}"
            if var_name not in x:
                x[var_name] = m.addVar(vtype=gp.GRB.BINARY, name=var_name)
    

    z = {c: {s: x.get(f"z{s}_{c.replace(' ', '')}") for s in [10, 20, 30]} for c in COMPONENTES}

    def _val(col):
        try:
            return float(pd.to_numeric(base_row.get(col), errors="coerce") or 0.0)
        except Exception:
            return 0.0

    lot_area  = _val("Lot Area")
    first_flr = _val("1st Flr SF")
    garage    = _val("Garage Area")
    wooddeck  = _val("Wood Deck SF")
    openporch = _val("Open Porch SF")
    enclosed  = _val("Enclosed Porch")
    ssn3      = _val("3Ssn Porch")
    screen    = _val("Screen Porch")
    pool      = _val("Pool Area")

    area_libre_base = lot_area - (first_flr + garage + wooddeck + openporch + enclosed + ssn3 + screen + pool)
    if area_libre_base < 0:
        area_libre_base = 0.0

    for c in COMPONENTES:
        m.addConstr(sum(z[c][s] for s in [10, 20, 30] if z[c][s] is not None) <= 1,
                    name=f"AMPL_one_scale_{c.replace(' ', '')}")

    delta = {}
    for c in COMPONENTES:
        base_val = _val(c)
        delta[c] = {s: round(base_val * s / 100, 3) for s in [10, 20, 30]}

    m.addConstr(
        (
            area_libre_base
            - (A_Full * (AddFull or 0) + A_Half * (AddHalf or 0)
               + A_Kitch * (AddKitch or 0) + A_Bed * (AddBed or 0))
            - gp.quicksum(delta[c][s] * z[c][s] for c in COMPONENTES for s in [10, 20, 30] if z[c][s] is not None)
        ) >= 0,
        name="AREA_libre_no_negativa"
    )

    lin_cost += (
        ct.add_fullbath_cost      * (AddFull  or 0)
        + ct.add_halfbath_cost    * (AddHalf  or 0)
        + ct.add_kitchen_cost_per_sf * A_Kitch * (AddKitch or 0)
        + ct.add_bedroom_cost_per_sf * A_Bed   * (AddBed   or 0)
    )

    for c in COMPONENTES:
        if c in x:
            m.addConstr(
                x[c] == _val(c) + gp.quicksum(delta[c][s] * z[c][s] for s in [10, 20, 30] if z[c][s] is not None),
                name=f"AMPL_link_{c.replace(' ', '')}"
            )

    if "1st Flr SF" in x:
        m.addConstr(
            x["1st Flr SF"]
            == first_flr
            + A_Full  * (AddFull  or 0)
            + A_Half  * (AddHalf  or 0)
            + A_Kitch * (AddKitch or 0)
            + A_Bed   * (AddBed   or 0),
            name="AMPL_link_1stflr"
        )
        v_1flr = m.addVar(vtype=gp.GRB.BINARY, name="x_Add1stFlr")
        delta_1flr = 40.0
        lin_cost += ct.construction_cost * delta_1flr * v_1flr
        m.addConstr(x["1st Flr SF"] >= first_flr + v_1flr * delta_1flr, name="AMPL_1stflr_upgrade")

    def _num_b(col: str) -> float:
        try:
            return float(pd.to_numeric(base_row.get(col), errors="coerce") or 0.0)
        except Exception:
            return 0.0

    base_counts = {
        "Full Bath":      _num_b("Full Bath"),
        "Half Bath":      _num_b("Half Bath"),
        "Bedroom AbvGr":  _num_b("Bedroom AbvGr"),
        "Kitchen AbvGr":  _num_b("Kitchen AbvGr"),
    }

    if "Full Bath" in x:
        m.addConstr(x["Full Bath"] == base_counts["Full Bath"] + (AddFull  or 0), name="COUNT_fullbath")
    if "Half Bath" in x:
        m.addConstr(x["Half Bath"] == base_counts["Half Bath"] + (AddHalf  or 0), name="COUNT_halfbath")
    if "Bedroom AbvGr" in x:
        m.addConstr(x["Bedroom AbvGr"] == base_counts["Bedroom AbvGr"] + (AddBed   or 0), name="COUNT_bedroom")
    if "Kitchen AbvGr" in x:
        m.addConstr(x["Kitchen AbvGr"] == base_counts["Kitchen AbvGr"] + (AddKitch or 0), name="COUNT_kitchen")

    # TotRms AbvGrd = base + nuevos dormitorios + nuevas cocinas (baños no se cuentan)
    if "TotRms AbvGrd" in x:
        tot_base = _num_b("TotRms AbvGrd")
        tot_var = x["TotRms AbvGrd"]
        m.addConstr(
            tot_var == tot_base + (AddBed or 0) + (AddKitch or 0),
            name="COUNT_totrooms"
        )
        _put_var_obj(X_input, "TotRms AbvGrd", tot_var)

    for c in COMPONENTES:
        if z[c][10] is not None:
            lin_cost += ct.ampl10_cost * delta[c][10] * z[c][10]
        if z[c][20] is not None:
            lin_cost += ct.ampl20_cost * delta[c][20] * z[c][20]
        if z[c][30] is not None:
            lin_cost += ct.ampl30_cost * delta[c][30] * z[c][30]

    # ================== GARAGE QUAL / COND ==================
    G_CATS = ["Ex", "Gd", "TA", "Fa", "Po", "No aplica"]

    gq = {g: x.get(f"garage_qual_is_{g}") for g in G_CATS}
    gc = {g: x.get(f"garage_cond_is_{g}") for g in G_CATS}

    def _base_txt(col):
        raw = _na2noaplica(base_row.get(col, "No aplica"))
        M = {0: "Po", 1: "Fa", 2: "TA", 3: "Gd", 4: "Ex"}
        try:
            v = int(pd.to_numeric(raw, errors="coerce"))
            if v in M:
                return M[v]
        except Exception:
            pass
        return str(raw)

    base_qual_txt = _base_txt("Garage Qual")
    base_cond_txt = _base_txt("Garage Cond")

    # máscara: 0 si es la base (no cobra), 1 si es distinta a la base (cobra)
    def _mask_for(base_txt):
        return {g: (0.0 if g == base_txt else 1.0) for g in G_CATS}

    maskQ = _mask_for(base_qual_txt)
    maskC = _mask_for(base_cond_txt)

    # Si alguno es "No aplica", se fija en "No aplica" y no se puede mejorar
    if base_qual_txt == "No aplica" or base_cond_txt == "No aplica":
        for g, v in gq.items():
            if v is not None:
                m.addConstr(v == (1.0 if g == "No aplica" else 0.0), name=f"GQ_fix_NA_{g}")
        for g, v in gc.items():
            if v is not None:
                m.addConstr(v == (1.0 if g == "No aplica" else 0.0), name=f"GC_fix_NA_{g}")
    else:
        # pick-one si existen las dummies
        if any(v is not None for v in gq.values()):
            m.addConstr(gp.quicksum(v for v in gq.values() if v is not None) == 1, name="GQ_pick_one")
        if any(v is not None for v in gc.values()):
            m.addConstr(gp.quicksum(v for v in gc.values() if v is not None) == 1, name="GC_pick_one")

        # “No bajar”: si tu tabla es monótona (más caro = mejor), bloquea niveles más baratos que la base
        def _cost(name): 
            return float(ct.garage_qc_costs.get(name, 0.0))

        base_cost_q = _cost(base_qual_txt)
        base_cost_c = _cost(base_cond_txt)

        for g, v in gq.items():
            if v is None: 
                continue
            if g == "No aplica" or _cost(g) < base_cost_q:
                v.UB = 0.0

        for g, v in gc.items():
            if v is None: 
                continue
            if g == "No aplica" or _cost(g) < base_cost_c:
                v.UB = 0.0

        # COSTO: si cambias, pagas el costo fijo de la categoría elegida (misma lógica que en otros qual/cond)
        lin_cost += gp.quicksum(_cost(g) * maskQ[g] * gq[g]
                                for g in G_CATS if g != "No aplica" and gq[g] is not None)
        lin_cost += gp.quicksum(_cost(g) * maskC[g] * gc[g]
                                for g in G_CATS if g != "No aplica" and gc[g] is not None)

 
    # ================== PAVED DRIVE ==================
    PAVED_CATS = ["Y", "P", "N"]
    paved = {d: x.get(f"paved_drive_is_{d}") for d in PAVED_CATS if f"paved_drive_is_{d}" in x}

    if paved:
        m.addConstr(gp.quicksum(paved.values()) == 1, name="PAVED_pick_one")

    base_pd = str(base_row.get("Paved Drive", "N")).strip()
    if base_pd not in PAVED_CATS:
        base_pd = "N"

    if base_pd == "Y":
        allowed = ["Y"]
    elif base_pd == "P":
        allowed = ["P", "Y"]
    else:
        allowed = ["N", "P", "Y"]

    for d in PAVED_CATS:
        if d not in allowed and d in paved:
            paved[d].UB = 0

    lin_cost += gp.quicksum(
        ct.paved_drive_costs[d] * paved[d]
        for d in PAVED_CATS
        if d != base_pd and d in paved
    )

    # ================== FENCE ==================
    FENCE_CATS = ["GdPrv", "MnPrv", "GdWo", "MnWw", "No aplica"]
    fn = {f: x.get(f"fence_is_{f}") for f in FENCE_CATS if f"fence_is_{f}" in x}

    if fn:
        m.addConstr(gp.quicksum(fn.values()) == 1, name="FENCE_pick_one")

    base_f = str(base_row.get("Fence", "No aplica")).strip()
    if base_f not in FENCE_CATS:
        base_f = "No aplica"

    if base_f == "No aplica":
        allowed_f = ["No aplica", "MnPrv", "GdPrv"]
    elif base_f in ["GdWo", "MnWw"]:
        allowed_f = [base_f, "MnPrv", "GdPrv"]
    else:
        allowed_f = [base_f]

    for fcat in FENCE_CATS:
        if fcat not in allowed_f and fcat in fn:
            fn[fcat].UB = 0

    lin_cost += gp.quicksum(
        ct.fence_category_cost(f) * fn[f]
        for f in FENCE_CATS if f != base_f and f in fn
    )

    lot_front = float(pd.to_numeric(base_row.get("Lot Frontage"), errors="coerce") or 0.0)
    if base_f == "No aplica":
        for f in ["MnPrv", "GdPrv"]:
            if f in fn:
                lin_cost += ct.fence_build_cost_per_ft * lot_front * fn[f]
    
    # Empujar Paved Drive al X_input si el pipeline usa one-hot
    for d, v in paved.items():
        if v is None:
            continue
        col = f"Paved Drive_{d}"
        if col in X_input.columns:
            _put_var_obj(X_input, col, v)

    # Empujar Fence al X_input si el pipeline usa one-hot
    for fcat, v in fn.items():
        if v is None:
            continue
        col = f"Fence_{fcat}"
        if col in X_input.columns:
            _put_var_obj(X_input, col, v)


    # ================== ELECTRICAL (universo completo) ==================
    ELECT_TYPES = ["SBrkr", "FuseA", "FuseF", "FuseP", "Mix"]
    elec_base_name = str(base_row.get("Electrical", "SBrkr")).strip()

    all_e_bin = {}
    for nm in ELECT_TYPES:
        var_name = f"elect_is_{nm}"
        if var_name in x:
            all_e_bin[nm] = x[var_name]
        else:
            v = m.addVar(vtype=gp.GRB.BINARY, name=var_name)
            if nm == elec_base_name:
                v.LB = v.UB = 1.0
            else:
                v.LB = v.UB = 0.0
            all_e_bin[nm] = v

    m.addConstr(gp.quicksum(all_e_bin.values()) == 1, name="ELEC_pick_one_full")

    for nm, vb in all_e_bin.items():
        col = f"Electrical_{nm}"
        if col in X_input.columns:
            _put_var_obj(X_input, col, vb)

    base_cost_e = ct.electrical_cost(elec_base_name)
    for nm, vb in all_e_bin.items():
        if ct.electrical_cost(nm) < base_cost_e:
            vb.UB = 0
    for nm, vb in all_e_bin.items():
        if nm != elec_base_name:
            lin_cost += ct.electrical_demo_small * vb
            lin_cost += (ct.electrical_cost(nm) - base_cost_e) * vb

    # ================== HEATING + HEATING QC ==================
    HEAT_TYPES = ["Floor", "GasA", "GasW", "Grav", "OthW", "Wall"]
    # Intenta obtener del diccionario x primero
    heat_bin = {nm: x.get(f"heat_is_{nm}") for nm in HEAT_TYPES if f"heat_is_{nm}" in x}

    heat_base = str(base_row.get("Heating", "GasA")).strip()
    q_map = {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4}
    try:
        qc_base = int(pd.to_numeric(base_row.get("Heating QC"), errors="coerce"))
        if qc_base not in (0, 1, 2, 3, 4):
            qc_base = q_map.get(str(base_row.get("Heating QC")).strip(), 2)
    except Exception:
        qc_base = q_map.get(str(base_row.get("Heating QC")).strip(), 2)

    if heat_bin:
        m.addConstr(gp.quicksum(heat_bin.values()) == 1, name="HEAT_pick_one_type")
        for nm, vb in heat_bin.items():
            _put_var_obj(X_input, f"Heating_{nm}", vb)
    else:
        # Si no están en x, crear variables binarias de Heating directamente en el modelo
        # Estas variables PERMANECERÁN después del XGBoost embedding como "heat_is_*"
        heat_bin = {}
        for nm in HEAT_TYPES:
            v = m.addVar(vtype=gp.GRB.BINARY, name=f"heat_is_{nm}")
            heat_bin[nm] = v
        
        # Agregar restricción one-hot para asegurar que solo UNA opción está activa
        m.addConstr(gp.quicksum(heat_bin.values()) == 1, name="HEAT_pick_one_type")
        
        # Agregar al diccionario X_input para que estén disponibles al embedding
        for nm, vb in heat_bin.items():
            _put_var_obj(X_input, f"Heating_{nm}", vb)

    upg_type = x.get("heat_upg_type")
    upg_qc   = x.get("heat_upg_qc")

    eligible_heat = 1 if qc_base <= 2 else 0
    
    # Solo limita el upgrade de CALIDAD si no es eligible
    # El cambio de TIPO siempre está permitido
    if (upg_qc is not None):
        m.addConstr(upg_qc <= eligible_heat, name="HEAT_qc_upgrade_only_if_eligible")

    if "Heating QC" in x:
        m.addConstr(x["Heating QC"] >= qc_base, name="HEAT_qc_upgrade_only")

    any_rebuild = m.addVar(vtype=gp.GRB.BINARY, name="HEAT_any_rebuild")
    if (upg_type is not None) and (upg_qc is not None):
        m.addConstr(any_rebuild >= upg_type)
        m.addConstr(any_rebuild >= upg_qc)
        m.addConstr(any_rebuild <= upg_type + upg_qc)
    else:
        m.addConstr(any_rebuild == 0)

    if "Heating QC" in x:
        m.addConstr(x["Heating QC"] >= 2 * any_rebuild, name="HEAT_qc_min_if_any_rebuild")

    base_type_cost = ct.heating_type_cost(heat_base)
    for nm, vb in heat_bin.items():
        if ct.heating_type_cost(nm) < base_type_cost:
            vb.UB = 0

    if eligible_heat == 0:
        # No se puede cambiar tipo de calefacción si la calidad ya es buena
        for nm, vb in heat_bin.items():
            vb.UB = 1 if nm == heat_base else 0
        if "Heating QC" in x:
            x["Heating QC"].LB = qc_base
            x["Heating QC"].UB = qc_base

    change_type = None
    if heat_bin:
        change_type = m.addVar(vtype=gp.GRB.BINARY, name="heat_change_type")
        if heat_base in heat_bin:
            m.addConstr(change_type == gp.quicksum(v for nm, v in heat_bin.items() if nm != heat_base),
                        name="HEAT_change_def")
        else:
            m.addConstr(change_type == 0, name="HEAT_change_def_guard")

    qc_bins = {}
    if "Heating QC" in x:
        for lvl, nm in enumerate(["Po", "Fa", "TA", "Gd", "Ex"]):
            qb = m.addVar(vtype=gp.GRB.BINARY, name=f"heat_qc_is_{nm}")
            qc_bins[nm] = qb
        m.addConstr(gp.quicksum(qc_bins.values()) == 1, name="HEAT_qc_onehot")
        m.addConstr(x["Heating QC"] == 0*qc_bins["Po"] + 1*qc_bins["Fa"] + 2*qc_bins["TA"]
                                    + 3*qc_bins["Gd"] + 4*qc_bins["Ex"], name="HEAT_qc_link")
        for nm, qb in qc_bins.items():
            _put_var_obj(X_input, f"Heating QC_{nm}", qb)

    # FIXED: Add path exclusion constraint - Type change XOR Quality upgrade (per specification)
    if eligible_heat == 1 and change_type is not None and qc_bins:
        UpgType_heat = m.addVar(vtype=gp.GRB.BINARY, name="heat_upg_type_flag")
        UpgQC_heat = m.addVar(vtype=gp.GRB.BINARY, name="heat_upg_qc_flag")
        m.addConstr(UpgType_heat + UpgQC_heat <= 1, name="HEAT_exclusive_paths")

    cost_heat = gp.LinExpr(0.0)
    for nm, vb in heat_bin.items():
        if nm != heat_base:
            cost_heat += ct.heating_type_cost(nm) * vb
    if qc_bins:
        for nm, qb in qc_bins.items():
            if q_map[nm] != qc_base:
                cost_heat += ct.heating_qc_cost(nm) * qb
    lin_cost += cost_heat

    # ================== FIREPLACE QU ==================
    # Permite: Po, Fa, TA, Gd, Ex y "No aplica" (=-1)
    FQ_CATS = ["Po", "Fa", "TA", "Gd", "Ex", "No aplica"]
    FQ_ORD  = {"No aplica": -1, "Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4}

    v_fq = x.get("Fireplace Qu")  # ya existe como variable numérica en el embed (LB=-1, UB=4)
    if isinstance(v_fq, gp.Var):
        # base como texto normalizado
        def _fq_txt(v):
            M = { -1:"No aplica", 0:"Po", 1:"Fa", 2:"TA", 3:"Gd", 4:"Ex" }
            try:
                n = int(pd.to_numeric(v, errors="coerce"))
                return M.get(n, "No aplica")
            except Exception:
                s = str(v).strip()
                return {"", "NA", "N/A", "NoAplica"}.__contains__(s) and "No aplica" or s

        base_fq_txt = _fq_txt(base_row.get("Fireplace Qu", "No aplica"))
        base_ord    = FQ_ORD[base_fq_txt]

        # dummies de elección del nivel destino
        fq = {nm: m.addVar(vtype=gp.GRB.BINARY, name=f"fireplace_is_{nm}") for nm in FQ_CATS}
        m.addConstr(gp.quicksum(fq.values()) == 1, name="FQ_pick_one")

        # enlaza el ordinal con la elección: x_FireplaceQu = sum(ord * dummy)
        m.addConstr(v_fq == gp.quicksum(FQ_ORD[nm] * fq[nm] for nm in FQ_CATS), name="FQ_level_match")

        # IMPLEMENT PATH RESTRICTIONS per specification
        # Po → {Po, Fa}
        # TA → {TA, Gd, Ex}
        # Fa/Gd/Ex → no change (fixed)
        # NA → NA (already handled above)
        
        allowed_paths = {
            -1: ["No aplica"],      # NA → NA
            0:  ["Po", "Fa"],       # Po → {Po, Fa}
            1:  ["Fa"],             # Fa → Fa (fixed)
            2:  ["TA", "Gd", "Ex"], # TA → {TA, Gd, Ex}
            3:  ["Gd"],             # Gd → Gd (fixed)
            4:  ["Ex"],             # Ex → Ex (fixed)
        }
        
        allowed = allowed_paths.get(base_ord, [base_fq_txt])
        for nm in FQ_CATS:
            if nm not in allowed:
                fq[nm].UB = 0.0

        # Si NO quieres permitir "agregar chimenea" cuando la base es "No aplica", descomenta:
        if base_fq_txt == "No aplica":
            for nm in FQ_CATS:
                fq[nm].UB = 0.0
            fq["No aplica"].LB = 1.0
            fq["No aplica"].UB = 1.0

        # costo por nivel destino (sin cobrar quedarse igual)
        def _fq_cost(name):
            # Acepta cualquiera de estos esquemas en ct:
            #   - ct.fireplace_qc_costs: dict { "TA": 1200, ... }
            #   - ct.fireplace_costs:     dict idem
            #   - ct.fireplace_cost(name) -> float
            if hasattr(ct, "fireplace_qc_costs"):
                return float(ct.fireplace_qc_costs.get(name, 0.0))
            if hasattr(ct, "fireplace_costs"):
                return float(ct.fireplace_costs.get(name, 0.0))
            if hasattr(ct, "fireplace_cost"):
                try:
                    return float(ct.fireplace_cost(name))
                except Exception:
                    return 0.0
            return 0.0

        fq_base_cost = _fq_cost(base_fq_txt)
        for nm in FQ_CATS:
            if nm == base_fq_txt:
                continue
            lin_cost += (_fq_cost(nm) - fq_base_cost) * fq[nm]

 

    # Empujar Garage Qual one-hot si existen columnas
    for g, v in gq.items():
        if v is None: continue
        col = f"Garage Qual_{g}"
        if col in X_input.columns:
            _put_var_obj(X_input, col, v)

    # Empujar Garage Cond one-hot si existen columnas
    for g, v in gc.items():
        if v is None: continue
        col = f"Garage Cond_{g}"
        if col in X_input.columns:
            _put_var_obj(X_input, col, v)

    # ================== BSMT: FINISH ALL ================== 
    b1_var = x.get("BsmtFin SF 1")
    b2_var = x.get("BsmtFin SF 2")
    bu_var = x.get("Bsmt Unf SF")
    if (b1_var is None) or (b2_var is None) or (bu_var is None):
        raise RuntimeError("Faltan variables de sótano: 'BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF'.")

    def _num(v):
        try: return float(pd.to_numeric(v, errors="coerce") or 0.0)
        except: return 0.0

    b1_base = _num(base_row.get("BsmtFin SF 1"))
    b2_base = _num(base_row.get("BsmtFin SF 2"))
    bu_base = _num(base_row.get("Bsmt Unf SF"))
    tb_base = _num(base_row.get("Total Bsmt SF"))
    if tb_base <= 0.0:
        tb_base = b1_base + b2_base + bu_base

    finish_bsmt = m.addVar(vtype=gp.GRB.BINARY, name="bsmt_finish_all")
    x1 = m.addVar(lb=0.0, name="bsmt_to_fin1")
    x2 = m.addVar(lb=0.0, name="bsmt_to_fin2")

    m.addConstr(x1 + x2 == bu_base * finish_bsmt, name="BSMT_all_or_nothing")
    m.addConstr(b1_var == b1_base + x1, name="BSMT_link_fin1")
    m.addConstr(b2_var == b2_base + x2, name="BSMT_link_fin2")
    m.addConstr(bu_var == bu_base * (1.0 - finish_bsmt), name="BSMT_link_unf")
    m.addConstr(b1_var + b2_var + bu_var == tb_base, name="BSMT_conservation_sum")

    lin_cost += ct.finish_basement_per_f2 * bu_base * finish_bsmt

    # ================== BSMT COND (ordinal) ==================
    BC_LEVELS = ["Po", "Fa", "TA", "Gd", "Ex"]
    BC_ORD = {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4}

    bc_base_txt = str(base_row.get("Bsmt Cond", "TA")).strip()
    bc_base = BC_ORD.get(bc_base_txt, 2)

    bc_bin = {nm: m.addVar(vtype=gp.GRB.BINARY, name=f"bsmtcond_is_{nm}") for nm in BC_LEVELS}
    m.addConstr(gp.quicksum(bc_bin.values()) == 1, name="BSMTCOND_onehot")

    for nm, vb in bc_bin.items():
        if BC_ORD[nm] < bc_base:
            vb.UB = 0.0

    if "Bsmt Cond" in x:
        m.addConstr(
            x["Bsmt Cond"] ==
            0*bc_bin["Po"] + 1*bc_bin["Fa"] + 2*bc_bin["TA"] + 3*bc_bin["Gd"] + 4*bc_bin["Ex"],
            name="BSMTCOND_link_int"
        )

    for nm in BC_LEVELS:
        col = f"Bsmt Cond_{nm}"
        if col in X_input.columns:
            _put_var_obj(X_input, col, bc_bin[nm])

    bc_base_cost = ct.bsmt_cond_cost(bc_base_txt)
    for nm, vb in bc_bin.items():
        if BC_ORD[nm] > bc_base:
            lin_cost += (ct.bsmt_cond_cost(nm) - bc_base_cost) * vb

    # ================== BSMT FIN TYPE1 / TYPE2 ==================
    BS_TYPES = ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"]
    b1 = {nm: x.get(f"b1_is_{nm}") for nm in BS_TYPES}
    b2 = {nm: x.get(f"b2_is_{nm}") for nm in BS_TYPES}

    b1_base = str(base_row.get("BsmtFin Type 1", "No aplica")).strip()
    b2_base = str(base_row.get("BsmtFin Type 2", "No aplica")).strip()
    has_b2  = 0 if b2_base in ["NA", "None", "nan", "NaN", "No aplica"] else 1

    if all(v is not None for v in b1.values()):
        m.addConstr(gp.quicksum(b1.values()) == 1, name="B1_pick_one")
    if all(v is not None for v in b2.values()):
        m.addConstr(gp.quicksum(b2.values()) == has_b2, name="B2_pick_hasB2")

    for nm, vb in b1.items():
        if vb is not None:
            _put_var_obj(X_input, f"BsmtFin Type 1_{nm}", vb)
    for nm, vb in b2.items():
        if vb is not None:
            _put_var_obj(X_input, f"BsmtFin Type 2_{nm}", vb)

    BAD = {"Rec", "LwQ", "Unf"}
    upgB1 = m.addVar(vtype=gp.GRB.BINARY, name="B1_upg_flag")
    upgB2 = m.addVar(vtype=gp.GRB.BINARY, name="B2_upg_flag")
    is_bad1 = 1 if b1_base in BAD else 0
    is_bad2 = 1 if (has_b2 == 1 and b2_base in BAD) else 0
    upgB1.LB = upgB1.UB = is_bad1
    upgB2.LB = upgB2.UB = is_bad2

    M1 = {nm: (0 if nm == b1_base else 1) for nm in BS_TYPES}
    M2 = {nm: (0 if nm == b2_base else 1) for nm in BS_TYPES}

    if all(v is not None for v in b1.values()):
        m.addConstr(gp.quicksum(M1[nm]*b1[nm] for nm in BS_TYPES if b1[nm] is not None) <= upgB1,
                    name="B1_change_if_upg")
    if has_b2 and all(v is not None for v in b2.values()):
        m.addConstr(gp.quicksum(M2[nm]*b2[nm] for nm in BS_TYPES if b2[nm] is not None) <= upgB2,
                    name="B2_change_if_upg")

    def _apply_allowed(bvars, b_base, upg_flag):
        if not bvars:
            return
        base_cost = ct.bsmt_type_cost(b_base)
        if upg_flag == 0:
            for nm, vb in bvars.items():
                if vb is None: continue
                if nm == b_base:
                    vb.LB = vb.UB = 1.0
                else:
                    vb.UB = 0.0
        else:
            for nm, vb in bvars.items():
                if vb is None: continue
                if ct.bsmt_type_cost(nm) < base_cost:
                    vb.UB = 0.0

    if b1_base == "No aplica":
        for nm, vb in b1.items():
            if vb is None: continue
            if nm == "No aplica": vb.LB = vb.UB = 1.0
            else: vb.UB = 0.0
    else:
        _apply_allowed(b1, b1_base, is_bad1)

    if has_b2 == 0:
        for nm, vb in b2.items():
            if vb is None: continue
            vb.UB = 0.0
    elif b2_base == "No aplica":
        for nm, vb in b2.items():
            if vb is None: continue
            if nm == "No aplica": vb.LB = vb.UB = 1.0
            else: vb.UB = 0.0
    else:
        _apply_allowed(b2, b2_base, is_bad2)

    cost_b1 = gp.LinExpr(0.0)
    cost_b2 = gp.LinExpr(0.0)
    for nm, vb in b1.items():
        if vb is not None and M1[nm] == 1:
            cost_b1 += ct.bsmt_type_cost(nm) * vb
    for nm, vb in b2.items():
        if vb is not None and M2[nm] == 1:
            cost_b2 += ct.bsmt_type_cost(nm) * vb

    lin_cost += cost_b1 + cost_b2

    # ================== UTILITIES ==================
    UTIL_LEVELS = ["ELO", "NoSeWa", "NoSewr", "AllPub"]
    util_bin = {nm: m.addVar(vtype=gp.GRB.BINARY, name=f"util_{nm}") for nm in UTIL_LEVELS}
    m.addConstr(gp.quicksum(util_bin.values()) == 1, name="UTIL_onehot")

    # base como ordinal 0..3, robusto
    def _util_ord(val) -> int:
        try:
            v = int(pd.to_numeric(val, errors="coerce"))
            return v if v in (0,1,2,3) else 0
        except Exception:
            return {"ELO":0, "NoSeWa":1, "NoSewr":2, "AllPub":3}.get(str(val).strip(), 0)

    base_util_raw = base_row.get("Utilities", "ELO")
    base_util_ord = _util_ord(base_util_raw)

    # no permitir bajar por debajo de la base
    for nm, ordv in zip(UTIL_LEVELS, [0,1,2,3]):
        if ordv < base_util_ord:
            util_bin[nm].UB = 0.0

    # enlazar ordinal x["Utilities"] si existe
    if "Utilities" in x:
        m.addConstr(
            x["Utilities"] ==
            0*util_bin["ELO"] + 1*util_bin["NoSeWa"] + 2*util_bin["NoSewr"] + 3*util_bin["AllPub"],
            name="UTIL_link_int"
        )
    else:
        # si no está en MODIFIABLE, inyecta el valor base al X_input por si acaso
        if "Utilities" in X_input.columns:
            X_input.loc[0, "Utilities"] = float(base_util_ord)

    # costo solo si se sube por sobre la base
    for nm, ordv in zip(UTIL_LEVELS, [0,1,2,3]):
        if ordv > base_util_ord:
            lin_cost += float(ct.util_cost(nm)) * util_bin[nm]

    # ================== RESTRICCIONES R1..R8 ==================
    if "1st Flr SF" in x and "2nd Flr SF" in x:
        m.addConstr(x["1st Flr SF"] >= x["2nd Flr SF"], name="R1_floor1_ge_floor2")

    if "Gr Liv Area" in x and "Lot Area" in base_row:
        m.addConstr(x["Gr Liv Area"] <= float(base_row["Lot Area"]), name="R2_grliv_le_lot")

    if "1st Flr SF" in x and "Total Bsmt SF" in x:
        m.addConstr(x["1st Flr SF"] >= x["Total Bsmt SF"], name="R3_floor1_ge_bsmt")

    def _val_or_var(col):
        if col in x:
            return x[col]
        try:
            v = base_row.get(col, 0)
        except Exception:
            return 0.0
        try:
            return float(pd.to_numeric(v, errors="coerce") or 0.0)
        except Exception:
            return 0.0

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
    
    print(X_input)
    # === FORZAR que X_input tenga TODAS las columnas que el modelo espera ===
    # Debe ir justo ANTES de _add_sklearn(...)
    try:
        expected = list(bundle.feature_names_in())  # todas las columnas de entrenamiento

        def _coerce_num_keep_var(x):
            # Si es Var de gurobi, conservar el objeto (no convertir),
            # porque si lo casteas a float lo "congelas".
            try:
                if hasattr(x, "X") or hasattr(x, "VarName"):
                    return x
                return float(x)
            except Exception:
                return x

        if isinstance(X_input, dict):
            present = set(X_input.keys())
            missing = [c for c in expected if c not in present]
            for c in missing:
                if c in base_X.columns:
                    X_input[c] = _coerce_num_keep_var(base_X.iloc[0][c])
                else:
                    X_input[c] = 0.0
            # ordenar
            X_input = {c: X_input[c] for c in expected}
        elif hasattr(X_input, "columns"):
            present = set(X_input.columns)
            missing = [c for c in expected if c not in present]
            for c in missing:
                if c in base_X.columns:
                    X_input[c] = base_X.iloc[0][c]
                else:
                    X_input[c] = 0.0
            # Reordenar columnas
            X_input = X_input[expected]
        else:
            # Caso raro: rehacer dict desde base
            tmp = {}
            for c in expected:
                tmp[c] = _coerce_num_keep_var(base_X.iloc[0][c]) if c in base_X.columns else 0.0
            X_input = tmp

        miss_n = 0 if 'missing' not in locals() else len(missing)
        print(f"[ALIGN] columnas faltantes añadidas a X_input: {miss_n}")
        if miss_n:
            print(f"[ALIGN] ejemplos: {missing[:15]}")

    except Exception as e:
        print(f"[ALIGN] fallo alineando X_input con features del modelo: {e}")

    # ---- DEBUG: captura de entradas del embed (columnas -> ESCALARES Var/float)
    try:
        if isinstance(X_input, dict):
            m._x_cols = list(X_input.keys())
            # dict ya viene como escalar por col (Var o float)
            m._x_vars = {k: X_input[k] for k in m._x_cols}
        elif hasattr(X_input, "loc"):
            # DataFrame de 1 fila -> toma SIEMPRE la celda [0, col]
            m._x_cols = list(X_input.columns)
            m._x_vars = {c: X_input.loc[0, c] for c in m._x_cols}
        else:
            # fallback raro
            m._x_cols = list(bundle.feature_names_in())
            m._x_vars = {c: X_input[i] for i, c in enumerate(m._x_cols)}
    except Exception:
        m._x_cols = None
        m._x_vars = None

    
    ORD_BOUNDS = {
        "Fireplace Qu": (-1.0, 4.0),
        "Pool QC": (-1.0, 4.0),
        "Kitchen Qual": (0.0, 4.0),
        "Exter Qual": (0.0, 4.0),
        "Exter Cond": (0.0, 4.0),
        "Heating QC": (0.0, 4.0),
        "Bsmt Cond": (-1.0, 4.0),
        "Garage Qual": (-1.0, 4.0),
        "Garage Cond": (-1.0, 4.0),
        "Utilities": (0.0, 3.0),
    }
    _MODIF_NAMES = {f.name for f in MODIFIABLE}

    # === BOUNDS suaves para entradas numéricas (después del ALIGN, antes de _add_sklearn) ===
    try:
        lower_default = 0.0
        upper_default = 1e6
        dummy_hints = ("_", "Neighborhood_", "Exterior 1st_", "Exterior 2nd_",
                    "Condition 1_", "Condition 2_", "Electrical_", "Heating_")

        # ¡OJO! Usa X_input para bounds, no m._x_vars,
        # y SIEMPRE respeta las cotas ya puestas (no las “subas” a 0)
        if hasattr(X_input, "loc"):
            for c in X_input.columns:
                v = X_input.loc[0, c]
                # Si hay una Var en una columna no declarada como modificable, fíjala al valor base
                if hasattr(v, "VarName") and c not in _MODIF_NAMES:
                    try:
                        base_val_c = float(base_X.iloc[0][c])
                        m.addConstr(v == base_val_c, name=f"FIX_unmod_{_safe_name(c)}")
                        continue
                    except Exception:
                        pass
                if hasattr(v, "LB") and hasattr(v, "UB"):
                    if c in ORD_BOUNDS:
                        lb, ub = ORD_BOUNDS[c]
                        # Ensancha a lo necesario sin romper cotas existentes
                        v.LB = max(v.LB, lb)
                        v.UB = min(v.UB, ub)
                    else:
                        is_dummy = any(h in c for h in dummy_hints)
                        if is_dummy:
                            v.LB = 0.0
                            v.UB = 1.0
                        
    except Exception as e:
        print(f"[BOUNDS] no pude poner bounds: {e}")

    v_fp = x.get("Fireplace Qu")
    v_pq = x.get("Pool QC")
    if v_fp is not None:
        print(f"[CHK] Fireplace Qu LB={v_fp.LB} UB={v_fp.UB}")
    if v_pq is not None:
        print(f"[CHK] Pool QC      LB={v_pq.LB} UB={v_pq.UB}")

    # ================== PREDICTOR XGB (embedding manual árbol a árbol) ==================
    # Features en el orden del booster (crítico para que las pruebas de cada árbol lean el split correcto)
    feat_full = list(bundle.feature_names_in())
    var_by_name: dict[str, gp.Var] = {}

    def _safe_name(name: str) -> str:
        return (
            str(name)
            .replace(" ", "_")
            .replace("[", "")
            .replace("]", "")
            .replace("/", "_")
        )

    for c in feat_full:
        val = X_input.loc[0, c]
        safe_c = _safe_name(c)
        if isinstance(val, gp.Var):
            v = val
        elif isinstance(val, gp.LinExpr):
            # Si viene como expresión lineal (p.ej. variable derivada), crear var y link
            v = m.addVar(lb=-gp.GRB.INFINITY, name=f"expr_{safe_c}")
            m.addConstr(v == val, name=f"link_{safe_c}")
        else:
            try:
                num = float(val)
            except Exception:
                num = 0.0
            v = m.addVar(lb=num, ub=num, name=f"const_{safe_c}")
        var_by_name[c] = v

    booster_order = list(bundle.booster_feature_order())
    missing = [f for f in booster_order if f not in var_by_name]
    if missing:
        sample = ", ".join(missing[:10])
        print(f"[EMBED] features del booster que no encontré en X_input ({len(missing)}): {sample}")

    x_vars = []
    for fname in booster_order:
        v = var_by_name.get(fname)
        if v is None:
            # crea dummy fija en 0 para cualquier feature que no esté en X_input
            v = m.addVar(lb=0.0, ub=0.0, name=f"const_{fname}")
        x_vars.append(v)

    # y_log crudo de los árboles (sin bias/base_score)
    y_log_raw = m.addVar(lb=-gp.GRB.INFINITY, name="y_log_raw")
    m._y_log_raw_var = y_log_raw  # para auditorías
    # Embedding de árboles XGB: usamos la versión estándar (sin epsilon) para
    # reproducir exactamente la lógica < vs >= de XGBoost.  Si en algún árbol
    # las ramas quedaran solapadas, el one‑hot de hojas lo resolverá al elegir
    # una sola hoja factible.
    # eps negativo desplaza el umbral para que empates (x==thr) no se cuelen al lado izquierdo.
    bundle.attach_to_gurobi(m, x_vars, y_log_raw, eps=-1e-6)

    # offset/base_score del modelo (si no pudo calcularse, se asume 0)
    try:
        b0 = float(bundle.b0_offset if hasattr(bundle, "b0_offset") else 0.0)
        if not np.isfinite(b0):
            b0 = 0.0
    except Exception:
        b0 = 0.0

    # y_log raw value computed by tree embedding
    y_log = m.addVar(lb=-gp.GRB.INFINITY, name="y_log")
    m.addConstr(y_log == y_log_raw + b0, name="YLOG_with_offset")

    # Exponer handlers para auditorías
    m._feat_order = booster_order
    m._x_cols = booster_order
    m._x_vars = {c: v for c, v in zip(booster_order, x_vars)}
    m._y_log_embed_var = y_log
    m._embed_pc = None  # no aplica cuando usamos embedding manual

    # Si el target está en log, genero y_price con PWL(exp)
    if bundle.is_log_target():
        y_price = m.addVar(lb=0.0, name="y_price")
        grid = np.linspace(10.0, 14.0, 161)
        m.addGenConstrPWL(y_log, y_price, grid.tolist(), np.expm1(grid).tolist(), name="exp_expm1")
    else:
        y_price = y_log

    m._y_log_var = y_log
    m._y_price_var = y_price
    m._base_price_val = base_price

    try:
        import numpy as _np
        exact = float(_np.expm1(m._y_log_var.X))
        pwl   = float(m._y_price_var.X)
        print(f"[PWL] expm1(y_log) exact={exact:,.2f} | y_price(PWL)={pwl:,.2f} | Δ={pwl-exact:+.2f}")
    except Exception:
        pass

    # ================== PRESUPUESTO / OBJETIVO ==================
    # Variable de costo explícita y ligada al LinExpr acumulado
    cost_model = m.addVar(lb=0.0, name="cost_model")
    m.addConstr(cost_model == lin_cost, name="COST_eq_linexpr")

    # Guardar refs para auditorías
    m._cost_var = cost_model
    m._lin_cost_expr = lin_cost

    # (1) Presupuesto
    m.addConstr(cost_model <= budget_usd, name="BUDGET")

    # === NO-CHANGE GUARD: si cost_model == 0 -> todo igual a base y y_price == base ===
    z_change = m.addVar(vtype=gp.GRB.BINARY, name="z_change")
    M_cost = max(1.0, float(m._budget if hasattr(m, "_budget") else budget_usd))
    m.addConstr(m._cost_var <= M_cost * z_change, name="NOCHG_cost_link")
    eps_cost = 1.0  
    m.addConstr(m._cost_var >= eps_cost * z_change, name="NOCHG_cost_lb")

    bigM_fallback = 1e6

    if getattr(m, "_x_vars", None) and getattr(m, "_x_cols", None):
        for fname in m._x_cols:
            var = m._x_vars[fname]
            if not hasattr(var, "LB"):
                # No es Var de Gurobi (es constante float): no hace falta ligar
                continue

            # base_val desde base_X (si no existe, convertir desde base_row)
            base_val = None
            try:
                if fname in base_X.columns:
                    base_val = float(base_X.iloc[0][fname])
                else:
                    base_val = float(_coerce_base_value(fname, base_row.get(fname, 0)))
            except Exception:
                base_val = None

            # Si no es finito, loguea y sáltalo (evita el NaN en la constante)
            if (base_val is None) or (not np.isfinite(base_val)):
                print(f"[NOCHG] skip '{fname}': base_val={base_val}")
                continue

            # M por variable usando LB/UB (con fallback)
            try:
                lb = float(var.LB); ub = float(var.UB)
            except Exception:
                lb, ub = 0.0, bigM_fallback

            Mpos = ub - base_val
            Mneg = base_val - lb
            if (not np.isfinite(Mpos)) or (Mpos < 0): Mpos = bigM_fallback
            if (not np.isfinite(Mneg)) or (Mneg < 0): Mneg = bigM_fallback

            m.addConstr(var - base_val       <= Mpos * z_change, name=f"NOCHG_pos_{fname}")
            m.addConstr(base_val - var       <= Mneg * z_change, name=f"NOCHG_neg_{fname}")

    # Anclar el precio cuando no hay cambios
    M_price = 2e6
    m.addConstr(m._y_price_var - float(base_price) <= M_price * z_change, name="NOCHG_price_pos")
    m.addConstr(float(base_price) - m._y_price_var <= M_price * z_change, name="NOCHG_price_neg")


    # (2) No permitir empeorar el precio base (puedes dejarlo si querías esa política)
    m.addConstr(y_price >= float(base_price) - 1e-6, name="MIN_PRICE_BASE")
    # Prohibir ROI negativo (opcional pero recomendado)
    m.addConstr(y_price - cost_model >= float(base_price) - 1e-6, name="NO_NEGATIVE_ROI")


    # (3) Objetivo
    m.setObjective(y_price - cost_model - float(base_price), gp.GRB.MAXIMIZE)

    return m
