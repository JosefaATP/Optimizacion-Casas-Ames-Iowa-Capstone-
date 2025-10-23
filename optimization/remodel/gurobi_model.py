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


def _qual_base_ord(base_row, col: str) -> int:
    MAP = {"No aplica": -1, "Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4}
    val = str(base_row.get(col, "No aplica")).strip()
    try:
        nv = int(pd.to_numeric(val, errors="coerce"))
        if nv in (-1, 0, 1, 2, 3, 4):
            return nv
    except Exception:
        pass
    return MAP.get(val, -1)

def _na2noaplica(s: str) -> str:
    s = str(s).strip()
    return "No aplica" if s in {"NA", "No aplica"} else s

# ---------- COERCIÓN DE VALORES BASE A NUMÉRICOS ----------
# Map ordinal estándar -1..4 (misma convención que usaste en el resto)
_ORD_MAP = {"Po":0, "Fa":1, "TA":2, "Gd":3, "Ex":4, "No aplica":-1, "NA":-1}

def _as_int_or_map(val, default=0, allow_minus1=True):
    try:
        v = int(pd.to_numeric(val, errors="coerce"))
        # aceptamos -1..4 si allow_minus1, o 0..4 si no
        if allow_minus1:
            return v if v in (-1, 0, 1, 2, 3, 4) else default
        else:
            return v if v in (0, 1, 2, 3, 4) else default
    except Exception:
        s = str(val).strip()
        if s in _ORD_MAP:
            v = _ORD_MAP[s]
            if (not allow_minus1) and v == -1:
                return default
            return v
        return default

_UTIL_MAP = {"ELO":0, "NoSeWa":1, "NoSewr":2, "AllPub":3}

# Columnas que el XGB espera como ordinales (entran como números)
_ORDINAL_MINUS1 = {"Fireplace Qu", "Pool QC"}   # permiten -1
_ORDINAL_0TO4   = {"Kitchen Qual", "Exter Qual", "Exter Cond", "Heating QC",
                   "Bsmt Qual", "Bsmt Cond", "Garage Qual", "Garage Cond"}

def _coerce_base_value(col: str, val):
    col = str(col)
    # Ordinales
    if col in _ORDINAL_MINUS1:
        return float(_as_int_or_map(val, default=-1, allow_minus1=True))
    if col in _ORDINAL_0TO4:
        return float(_as_int_or_map(val, default=2, allow_minus1=True))  # TA=2 por defecto

    # Utilities (codificación entera 0..3)
    if col == "Utilities":
        s = str(val).strip()
        if s in _UTIL_MAP: 
            return float(_UTIL_MAP[s])
        try:
            v = int(pd.to_numeric(val, errors="coerce"))
            return float(v if v in (0,1,2,3) else 3)  # default AllPub
        except Exception:
            return 3.0  # AllPub

    # Booleans tipo Y/N que el pipeline pudiera tener como 0/1 (solo por seguridad)
    if col in {"Central Air"}:
        s = str(val).strip().upper()
        return 1.0 if s in {"Y", "YES", "1", "TRUE"} else 0.0

    # Genérico: si es numérico, lo devuelvo como float; si es "No aplica"/"NA" → 0.0
    s = str(val).strip()
    if s in {"No aplica","NA","nan","None",""}:
        return 0.0
    try:
        return float(pd.to_numeric(val, errors="coerce"))
    except Exception:
        # último recurso: deja 0.0 (pero evita strings)
        return 0.0
# ---------- FIN COERCIÓN ----------

def _norm_cat(s: str) -> str:
    s = str(s).strip()
    return "No aplica" if s in {"NA", "No aplica"} else s

def build_base_input_row(bundle, base_row: pd.Series) -> pd.DataFrame:
    """
    Devuelve una fila 1xN con EXACTAMENTE las columnas que espera el regressor
    (bundle.feature_names_in()), ya numéricas (one-hots/ordinales) y alineadas
    a como las usa el MIP. Así, y_base y y_price son comparables 1:1.
    """
    cols = list(bundle.feature_names_in())
    row = {}

    for c in cols:
        # ¿Es dummy OHE del tipo "Col_Cat"?
        if "_" in c and (c.count("_") >= 1):
            root, cat = c.split("_", 1)
            root = root.replace("_", " ").strip()
            cat  = cat.strip()

            base_val = base_row.get(root, None)

            # Normalizaciones mínimas coherentes con el MIP
            if root == "Central Air":
                base_cat = "Y" if str(base_val).strip().upper() in {"Y","YES","1","TRUE"} else "N"
            else:
                base_cat = _norm_cat(base_val)

            row[c] = 1.0 if _norm_cat(base_cat) == _norm_cat(cat) else 0.0
        else:
            # Columna numérica/ordinal ya en la grilla del modelo
            row[c] = float(_coerce_base_value(c, base_row.get(c, 0)))
    return pd.DataFrame([row], columns=cols, dtype=float)


FORBID_BUILD_WHEN_NA = {"Fireplace Qu","Bsmt Cond","Garage Qual","Garage Cond","Pool QC"}

def apply_quality_policy_ordinal(m, x: dict, base_row: pd.Series, col: str):
    """Enlaza política para variable ordinal x[col] (−1..4):
       - Si base = 'No aplica' y col ∈ FORBID_BUILD_WHEN_NA: fija a −1 (no construir).
       - Si base >= 0: no degradar (x[col] >= base).
    """
    q = x.get(col)
    if q is None:
        return
    base_ord = _qual_base_ord(base_row, col)
    base_txt = str(base_row.get(col, "No aplica")).strip()

    if base_txt == "No aplica" and col in FORBID_BUILD_WHEN_NA:
        q.LB = -1
        q.UB = -1
        return

    # si existe (>=0), no permitir empeorar
    if base_ord >= 0:
        m.addConstr(q >= base_ord, name=f"{col.replace(' ','_')}_upgrade_only")



def build_mip_embed(base_row: pd.Series, budget: float, ct: CostTables, bundle: XGBBundle, base_price=None) -> gp.Model:
    # --- Normalizar presupuesto a USD --

    m = gp.Model("remodel_embed")

    # --- Normalizar presupuesto a float USD ---
    try:
        budget_usd = float(budget)
        if not np.isfinite(budget_usd) or budget_usd <= 0:
            print(f"[WARN] Presupuesto inválido o no positivo ({budget}), se usa fallback 40000")
            budget_usd = 40000.0
    except Exception:
        print(f"[WARN] Presupuesto no numérico ({type(budget)}={budget}), se usa fallback 40000")
        budget_usd = 40000.0
    
    m._budget_usd = float(budget_usd)
    m._budget = float(budget_usd)  # compat con tu logger anterior


    # si no viene precalculado, lo calculamos rápido

    if base_price is None:
        Xb = build_base_input_row(bundle, base_row)
        base_price = float(bundle.predict(Xb).iloc[0])


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

    row_vals: Dict[str, Any] = {}
    for fname in feature_order:
        if fname in modif:
            row_vals[fname] = x[fname]
        else:
            base_val = base_row.get(fname, 0)
            row_vals[fname] = _coerce_base_value(fname, base_val)  

    # ... arriba: X_input = pd.DataFrame([row_vals], columns=feature_order)
    X_input = pd.DataFrame([row_vals], columns=feature_order, dtype=object)
    # (mantén el _align_ohe_dtypes(...) tal como ya lo tienes)

    for col in ["Kitchen Qual","Exter Qual","Exter Cond","Heating QC",
                "Fireplace Qu","Bsmt Cond","Garage Qual","Garage Cond","Pool QC"]:
        apply_quality_policy_ordinal(m, x, base_row, col)


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

    # --- Materiales disponibles (usa mismos nombres que en costs.py y en tu CSV)
    EXT_MATS = [
        "AsbShng","AsphShn","BrkComm","BrkFace","CBlock","CemntBd","HdBoard","ImStucc",
        "MetalSd","Other","Plywood","PreCast","Stone","Stucco","VinylSd","Wd Sdng","WdShngl",
    ]

    # --- Binarias de decisión que vienen desde MODIFIABLE
    ex1 = {nm: x.get(f"ex1_is_{nm}") for nm in EXT_MATS if f"ex1_is_{nm}" in x}
    ex2 = {nm: x.get(f"ex2_is_{nm}") for nm in EXT_MATS if f"ex2_is_{nm}" in x}

    # --- ¿Existe Exterior 2nd en la casa base?
    ex2_base_name_raw = str(base_row.get("Exterior 2nd", "None"))
    Ilas2 = 0 if (ex2_base_name_raw in ["None","nan","NaN","NoneType","0"] or pd.isna(base_row.get("Exterior 2nd"))) else 1

    # --- Selección única / activación por Ilas2
    if ex1:
        m.addConstr(gp.quicksum(ex1.values()) == 1, name="EXT_ex1_pick_one")
    if ex2:
        m.addConstr(gp.quicksum(ex2.values()) == Ilas2, name="EXT_ex2_pick_ilas2")

    # --- Inyectar dummies al X_input (pipeline entrenado con OHE de Exterior1st/2nd)
    def _put_var(df: pd.DataFrame, col: str, var: gp.Var):
        if col in df.columns:
            if df[col].dtype != "O":
                df[col] = df[col].astype("object")
            df.loc[0, col] = var

    for nm in EXT_MATS:
        col1 = f"Exterior 1st_{nm}"
        if col1 in X_input.columns and nm in ex1:
            _put_var(X_input, col1, ex1[nm])
        col2 = f"Exterior 2nd_{nm}"
        if col2 in X_input.columns and nm in ex2:
            _put_var(X_input, col2, ex2[nm])

    # --- Base: material actual por frente
    ex1_base_name = str(base_row.get("Exterior 1st", "None")).strip()
    ex2_base_name = str(base_row.get("Exterior 2nd", "None")).strip()

    # --- Elegibilidad: sólo si (Exter Qual <= TA) o (Exter Cond <= TA)
    def _q_to_ord(v):
        M = {"Po":0,"Fa":1,"TA":2,"Gd":3,"Ex":4}
        try:
            return int(pd.to_numeric(v, errors="coerce"))
        except Exception:
            return M.get(str(v).strip(), 2)

    exq_base_ord = _q_to_ord(base_row.get("Exter Qual", "TA"))
    exc_base_ord = _q_to_ord(base_row.get("Exter Cond", "TA"))
    eligible = 1 if (exq_base_ord <= 2 or exc_base_ord <= 2) else 0

    # --- Si NO es elegible: congelar materiales a los de base (si existen esas vars)
    if eligible == 0:
        # Frente 1
        if ex1_base_name in ex1:
            for nm, v in ex1.items():
                v.UB = 1.0 if nm == ex1_base_name else 0.0
        # Frente 2 (sólo si existe y tenemos la var)
        if Ilas2 == 1 and ex2:
            if ex2_base_name in ex2:
                for nm, v in ex2.items():
                    v.UB = 1.0 if nm == ex2_base_name else 0.0

    # --- Costos fijos por CAMBIO de material (sin multiplicar por área; sin demolición extra)
    #     Nota: sólo se cobra si el material final es distinto al base.
    for nm, vb in ex1.items():
        if nm != ex1_base_name:
            lin_cost += ct.ext_mat_cost(nm) * vb

    if Ilas2 == 1:
        for nm, vb in ex2.items():
            if nm != ex2_base_name:
                lin_cost += ct.ext_mat_cost(nm) * vb

    # -------------------------------------------------------------------------
    # Exter Qual / Exter Cond con costos fijos por nivel (lumpsum) y upgrade-only
    # -------------------------------------------------------------------------
    EQ_LEVELS = ["Po","Fa","TA","Gd","Ex"]
    ORD = {"Po":0,"Fa":1,"TA":2,"Gd":3,"Ex":4}

    # --- One-hot de calidad y condición finales
    eq_bin = {nm: m.addVar(vtype=gp.GRB.BINARY, name=f"exterqual_is_{nm}") for nm in EQ_LEVELS}
    ec_bin = {nm: m.addVar(vtype=gp.GRB.BINARY, name=f"extercond_is_{nm}") for nm in EQ_LEVELS}
    m.addConstr(gp.quicksum(eq_bin.values()) == 1, name="EXT_EQ_onehot")
    m.addConstr(gp.quicksum(ec_bin.values()) == 1, name="EXT_EC_onehot")

    # --- No-empeorar (bloquear niveles por debajo de la base)
    for nm in EQ_LEVELS:
        if ORD[nm] < exq_base_ord:
            eq_bin[nm].UB = 0.0
        if ORD[nm] < exc_base_ord:
            ec_bin[nm].UB = 0.0

    # --- Si NO es elegible: fijar ambos al nivel base
    if eligible == 0:
        # Calidad
        for nm, vb in eq_bin.items():
            if ORD[nm] == exq_base_ord:
                vb.LB = vb.UB = 1.0
            else:
                vb.UB = 0.0
        # Condición
        for nm, vb in ec_bin.items():
            if ORD[nm] == exc_base_ord:
                vb.LB = vb.UB = 1.0
            else:
                vb.UB = 0.0

    # --- Enlazar con la variable ordinal si existe en MODIFIABLE
    if "Exter Qual" in x:
        m.addConstr(x["Exter Qual"] ==
                    0*eq_bin["Po"] + 1*eq_bin["Fa"] + 2*eq_bin["TA"] + 3*eq_bin["Gd"] + 4*eq_bin["Ex"],
                    name="EXT_EQ_link_int")
    if "Exter Cond" in x:
        m.addConstr(x["Exter Cond"] ==
                    0*ec_bin["Po"] + 1*ec_bin["Fa"] + 2*ec_bin["TA"] + 3*ec_bin["Gd"] + 4*ec_bin["Ex"],
                    name="EXT_EC_link_int")

    # --- Si tu pipeline trae OHE de Qual/Cond, inyecta dummies
    for nm in EQ_LEVELS:
        col = f"Exter Qual_{nm}"
        if col in X_input.columns:
            _put_var(X_input, col, eq_bin[nm])
    for nm in EQ_LEVELS:
        col = f"Exter Cond_{nm}"
        if col in X_input.columns:
            _put_var(X_input, col, ec_bin[nm])

    # --- Costos fijos por subir de nivel (sólo si el nivel final es > base)
    for nm, vb in eq_bin.items():
        if ORD[nm] > exq_base_ord:
            lin_cost += ct.exter_qual_cost(nm) * vb
    for nm, vb in ec_bin.items():
        if ORD[nm] > exc_base_ord:
            lin_cost += ct.exter_cond_cost(nm) * vb

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
    # 4) ROOF: estilo fijo segun base_row, material con costo fijo + compatibilidad
    # -------------------
    style_names = ["Flat", "Gable", "Gambrel", "Hip", "Mansard", "Shed"]
    matl_names  = ["ClyTile", "CompShg", "Membran", "Metal", "Roll", "Tar&Grv", "WdShake", "WdShngl"]

    # binarios ya creados en x
    s_bin = {nm: x[f"roof_style_is_{nm}"] for nm in style_names if f"roof_style_is_{nm}" in x}
    m_bin = {nm: x[f"roof_matl_is_{nm}"]  for nm in matl_names  if f"roof_matl_is_{nm}"  in x}

    # helper para inyectar gp.Var en X_input
    def _put_var(df: pd.DataFrame, col: str, var: gp.Var):
        if col in df.columns:
            if df[col].dtype != "O":
                df[col] = df[col].astype("object")
            df.loc[0, col] = var

    # inyectar dummies al DataFrame del modelo si existen
    for nm in style_names:
        col = f"Roof Style_{nm}"
        if col in X_input.columns and nm in s_bin:
            _put_var(X_input, col, s_bin[nm])

    for nm in matl_names:
        col = f"Roof Matl_{nm}"
        if col in X_input.columns and nm in m_bin:
            _put_var(X_input, col, m_bin[nm])

    # === estilos prohibidos por compatibilidad segun tu matriz ===
    ROOF_FORBIDS = {
        "Gable":   ["Membran"],
        "Hip":     ["Membran"],
        "Flat":    ["WdShngl", "ClyTile", "CompShg"],
        "Mansard": ["Membran"],
        "Shed":    ["ClyTile"],
        "Gambrel": [],
    }

    # helpers para leer la base sin romper si cambia el nombre
    def _base_val(name, alt=None):
        if name in base_row:
            return str(base_row[name])
        if alt and alt in base_row:
            return str(base_row[alt])
        return None

    base_style = _base_val("RoofStyle", "Roof Style")
    base_mat   = _base_val("RoofMatl",  "Roof Matl")

    # === 1) fijar RoofStyle a la base ===
    if s_bin and base_style is not None:
        m.addConstr(gp.quicksum(s_bin.values()) == 1, name="ROOF_pick_one_style")
        for nm, var in s_bin.items():
            m.addConstr(var == (1.0 if nm == base_style else 0.0), name=f"ROOF_style_fixed_{nm}")

    # === 2) elegir exactamente un material ===
    if m_bin:
        m.addConstr(gp.quicksum(m_bin.values()) == 1, name="ROOF_pick_one_matl")

    # === 3) compatibilidad: apagar materiales no permitidos para el estilo base ===
    if m_bin and base_style is not None:
        for mn in ROOF_FORBIDS.get(base_style, []):
            if mn in m_bin:
                m.addConstr(m_bin[mn] == 0.0, name=f"ROOF_incompat_{base_style}_{mn}")

    # === 4) costo fijo por cambio de material: demolicion + costo del NUEVO material (excluye el base) ===
    cost_roof = gp.LinExpr(0.0)
    if m_bin and base_mat is not None and base_mat in m_bin:
        change_ind = 1.0 - m_bin[base_mat]
        cost_roof += ct.roof_demo_cost * change_ind
        # solo cobramos materiales distintos al base
        for mat, y in m_bin.items():
            if mat != base_mat:
                cost_roof += ct.get_roof_matl_cost(mat) * y


    # sumar al costo total
    lin_cost += cost_roof
    # ================== FIN (ROOF) ==================

    # ========== GARAGE FINISH (pdf: p.29) ==========
    GF = ["Fin", "RFn", "Unf", "No aplica"]      # categorías de decisión
    GF_RFnOrWorse = ["RFn", "Unf"]               # subconjunto "RFN o peor"

    # --- helper NA/No aplica en columnas base
    def _col_exists(name: str) -> bool:
        return name in base_row

    def _gf_dummy(col_suffix: str) -> float:
        # acepta "No aplica" o "NA" como sufijo de dummy en base
        for label in ["No aplica", "NA"]:
            col = f"Garage Finish_{label if col_suffix == 'No aplica' else col_suffix}"
            if _col_exists(col):
                return float(base_row.get(col, 0.0) or 0.0)
        # si no hay dummies, intenta leer columna textual "Garage Finish"
        if "Garage Finish" in base_row:
            txt = str(base_row["Garage Finish"]).strip()
            if txt in {"NA", "No aplica"} and col_suffix == "No aplica": return 1.0
            if txt == col_suffix: return 1.0
        return 0.0

    BaseGa = {g: _gf_dummy(g) for g in GF}                  # one-hot base (robusto NA/No aplica)
    MaskGa = {g: 1.0 - BaseGa[g] for g in GF}               # máscara 1 si cambia a g

    # --- variables binarias de estado (asumo ya creadas en x)
    gar = {g: x.get(f"garage_finish_is_{g}") for g in GF if f"garage_finish_is_{g}" in x}
    UpgGa = x.get("UpgGarage")  # puede ser None si no la modelas

    # --- Inyectar OHE de "Garage Finish" al X_input si el pipeline lo trae
    #      Soporta tanto "..._No aplica" como "..._NA"
    def _put_ohe(df: pd.DataFrame, col: str, var: gp.Var):
        if col in df.columns:
            if df[col].dtype != "O":
                df[col] = df[col].astype("object")
            df.loc[0, col] = var

    gf_cols = [c for c in X_input.columns if c.startswith("Garage Finish_")]
    if gf_cols and gar:
        for nm, vb in gar.items():
            if vb is None:
                continue
            # mapear 'No aplica' a la columna que exista
            if nm == "No aplica":
                if "Garage Finish_No aplica" in X_input.columns:
                    _put_ohe(X_input, "Garage Finish_No aplica", vb)
                elif "Garage Finish_NA" in X_input.columns:
                    _put_ohe(X_input, "Garage Finish_NA", vb)
            else:
                col = f"Garage Finish_{nm}"
                if col in X_input.columns:
                    _put_ohe(X_input, col, vb)

    # --- selección única
    m.addConstr(gp.quicksum(v for v in gar.values() if v is not None) == 1.0, name="GaFin_pick_one")

    try:
        gt   = str(base_row.get("Garage Type", "No aplica")).strip()
        area = float(pd.to_numeric(base_row.get("Garage Area"), errors="coerce") or 0.0)
        cars = float(pd.to_numeric(base_row.get("Garage Cars"), errors="coerce") or 0.0)
        has_garage = (gt not in {"NA", "No aplica"}) or (area > 0) or (cars > 0)
    except Exception:
        has_garage = False

    if has_garage and "No aplica" in gar and gar["No aplica"] is not None:
        gar["No aplica"].UB = 0.0

    # --- conjuntos permitidos / fijaciones (pdf, caso a caso)
    # Si base=NA  -> gar_NA = 1, resto 0
    if BaseGa["No aplica"] == 1.0:
        for g, v in gar.items():
            if v is None: continue
            v.UB = 1.0 if g == "No aplica" else 0.0
        if UpgGa is not None:
            m.addConstr(UpgGa == 0.0, name="UpgGa_off_for_NA")

    # Si base=Fin -> gar_Fin = 1, resto 0
    elif BaseGa["Fin"] == 1.0:
        for g, v in gar.items():
            if v is None: continue
            v.UB = 1.0 if g == "Fin" else 0.0
        if UpgGa is not None:
            m.addConstr(UpgGa == 0.0, name="UpgGa_off_for_Fin")

    # Si base∈{RFn,Unf} -> puede mantener (RFn/Unf) o subir a Fin
    else:
        # --- activación: solo si la base es RFn/Unf se PERMITE upgradear,
        #     pero NO se obliga (UpgGa es opcional)
        if UpgGa is not None:
            m.addConstr(UpgGa <= BaseGa["RFn"] + BaseGa["Unf"],
                        name="UpgGa_allowed_if_eligible")

            # Si UpgGa=0, no se puede cambiar:  sum M*g <= 0  → quedarse en la base
            m.addConstr(gp.quicksum(MaskGa[g] * gar[g] for g in GF if gar[g] is not None)
                        <= UpgGa, name="GaFin_change_only_if_upg")

            # Fin solo si UpgGa=1
            if "Fin" in gar and gar["Fin"] is not None:
                m.addConstr(gar["Fin"] <= UpgGa, name="GaFin_Fin_requires_Upg")

            # Si UpgGa=1, no podés terminar en RFn/Unf
            m.addConstr(gp.quicksum(gar[g] for g in GF_RFnOrWorse if g in gar and gar[g] is not None)
                        <= 1.0 - UpgGa, name="GaFin_RFnUnf_disallowed_when_upg")
        else:
            # (tu fallback sin UpgGa) – dejalo tal cual
            ORDER = ["Unf", "RFn", "Fin"]
            base_cat = next((g for g in ORDER if BaseGa[g] == 1.0), None)
            if base_cat:
                base_idx = ORDER.index(base_cat)
                for i, g in enumerate(ORDER):
                    if i < base_idx and g in gar and gar[g] is not None:
                        gar[g].UB = 0.0


    # --- costo (pdf): CostoGaFin = sum_g C_GF^g * M_i,g * gar_i,g
    lin_cost += gp.quicksum(
        float(ct.garage_finish_cost(g)) * MaskGa[g] * gar[g]
        for g in GF if g in gar and gar[g] is not None
    )

    # ========== FIN GARAGE FINISH ==========



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


    # ========== POOL QC (solo mejora, no crear piscina) ==========
    PQC = ["Ex","Gd","TA","Fa","Po","No aplica"]

    def _pq_base_dummy(tag: str) -> float:
        # lee dummies "Pool QC_*" tolerando No aplica/NA; o la columna textual "Pool QC"
        for lab in ["No aplica", "NA"]:
            col = f"Pool QC_{lab if tag=='No aplica' else tag}"
            if col in base_row:
                return float(base_row.get(col, 0.0) or 0.0)
        if "Pool QC" in base_row:
            txt = str(base_row["Pool QC"]).strip()
            if txt in {"NA","No aplica"} and tag == "No aplica": return 1.0
            if txt == tag: return 1.0
        return 0.0

    BasePQ = {g: _pq_base_dummy(g) for g in PQC}
    MaskPQ = {g: 1.0 - BasePQ[g] for g in PQC}

    # helper: conseguir la variable con varios alias
    def _get_var_for(g: str):
        aliases = [
            f"x_pool_qc_is_{g}", f"pool_qc_is_{g}",
            # alias NA
            f"x_pool_qc_is_NA",  f"pool_qc_is_NA"
        ] if g == "No aplica" else [f"x_pool_qc_is_{g}", f"pool_qc_is_{g}"]
        for nm in aliases:
            v = m.getVarByName(nm)
            if v is not None:
                return v
        # búsqueda laxa por si hay diferencias de espacios/underscores/case
        key_targets = {a.replace(" ", "").replace("_", "").lower() for a in aliases}
        for v in m.getVars():
            key = v.VarName.replace(" ", "").replace("_", "").lower()
            if key in key_targets:
                return v
        return None

    pq = {g: _get_var_for(g) for g in PQC}
    pq = {g: v for g, v in pq.items() if v is not None}  # filtra solo las que existen

    # Caso base: si No aplica = 1, fijar NA y apagar el resto
    if BasePQ.get("No aplica", 0.0) == 1.0:
        if "No aplica" in pq:
            m.addConstr(pq["No aplica"] == 1.0, name="PoolQC_fix_NA")
        for g, v in pq.items():
            if g != "No aplica":
                v.UB = 0.0
        # no impongas pick_one si solo tienes variables apagadas/ausentes
        if "No aplica" in pq:
            m.addConstr(gp.quicksum(pq.values()) == 1.0, name="PoolQC_pick_one")
    else:
        # anti-downgrade
        ORDER = ["Po","Fa","TA","Gd","Ex"]
        base_cat = next((g for g in ORDER if BasePQ.get(g, 0.0) == 1.0), None)
        if base_cat:
            b = ORDER.index(base_cat)
            for i, g in enumerate(ORDER):
                if i < b and g in pq:
                    pq[g].UB = 0.0  # prohíbe bajar

        # pick-one solo sobre las variables existentes y permitidas
        if pq:
            m.addConstr(gp.quicksum(pq.values()) == 1.0, name="PoolQC_pick_one")

    # costo: calidad + área de piscina
    lin_cost += gp.quicksum(
        (float(ct.poolqc_costs.get(g, 0.0)) + float(ct.pool_area_cost)) * MaskPQ.get(g, 1.0) * pq[g]
        for g in PQC if g in pq
    )

    # ========== FIN POOL QC ==========



    # =======================================================================
    # ========== ÁREA LIBRE Y DECISIONES DE AMPLIACIÓN / AGREGADO ==========
    # Basado en PDF de modelo de remodelación (p.26–28)
    # =======================================================================

    # --- 1. Parámetros de superficie por agregado directo (ft²) ---
    A_Full, A_Half, A_Kitch, A_Bed = 40.0, 20.0, 75.0, 70.0  # baño completo, medio, cocina, dormitorio

    # Variables binarias de agregados (deben existir en x)
    AddFull  = x.get("AddFull", None)
    AddHalf  = x.get("AddHalf", None)
    AddKitch = x.get("AddKitch", None)
    AddBed   = x.get("AddBed", None)

    # --- 2. Componentes con ampliaciones porcentuales (10, 20, 30%) ---
    COMPONENTES = [
        "Garage Area", "Wood Deck SF", "Open Porch SF",
        "Enclosed Porch", "3Ssn Porch", "Screen Porch", "Pool Area"
    ]
    z = {c: {s: x[f"z{s}_{c.replace(' ', '')}"] for s in [10, 20, 30]} for c in COMPONENTES}

    # --- 3. Helper para obtener valor numérico base ---
    def _val(col):
        try:
            return float(pd.to_numeric(base_row.get(col), errors="coerce") or 0.0)
        except Exception:
            return 0.0

    # --- 4. Calcular área libre del terreno (ft² disponibles para ampliar) ---
    lot_area = _val("Lot Area")
    first_flr = _val("1st Flr SF")
    garage = _val("Garage Area")
    wooddeck = _val("Wood Deck SF")
    openporch = _val("Open Porch SF")
    enclosed = _val("Enclosed Porch")
    ssn3 = _val("3Ssn Porch")
    screen = _val("Screen Porch")
    pool = _val("Pool Area")

    area_libre_base = lot_area - (first_flr + garage + wooddeck + openporch + enclosed + ssn3 + screen + pool)
    if area_libre_base < 0:
        area_libre_base = 0.0

    # --- 5. Restricción: solo una escala de ampliación por componente ---
    for c in COMPONENTES:
        m.addConstr(sum(z[c][s] for s in [10, 20, 30]) <= 1, name=f"AMPL_one_scale_{c.replace(' ', '')}")

    # --- 6. Δ (delta) de ampliación para cada porcentaje ---
    delta = {}
    for c in COMPONENTES:
        base_val = _val(c)
        if pd.isna(base_val) or np.isinf(base_val):
            base_val = 0.0
        delta[c] = {s: round(base_val * s / 100, 3) for s in [10, 20, 30]}

    # --- 7. Restricción: área libre no negativa (PDF p.27) ---
    m.addConstr(
        (
            area_libre_base
            - (A_Full * (AddFull or 0) + A_Half * (AddHalf or 0)
            + A_Kitch * (AddKitch or 0) + A_Bed * (AddBed or 0))
            - gp.quicksum(delta[c][s] * z[c][s] for c in COMPONENTES for s in [10, 20, 30])
        ) >= 0,
        name="AREA_libre_no_negativa"
    )

    # --- 8. Costo de construcción de agregados directos (PDF p.28) ---
    lin_cost += ct.construction_cost * (
        A_Full  * (AddFull  or 0)
        + A_Half  * (AddHalf  or 0)
        + A_Kitch * (AddKitch or 0)
        + A_Bed   * (AddBed   or 0)
    )

    # --- 9. Áreas finales = base + Δ ampliaciones ---
    for c in COMPONENTES:
        if c in x:  # solo si la feature es modificable
            m.addConstr(
                x[c] == _val(c) + gp.quicksum(delta[c][s] * z[c][s] for s in [10, 20, 30]),
                name=f"AMPL_link_{c.replace(' ', '')}"
            )

    # --- 10. Vincular agregados al 1st Flr SF ---
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

        # Ampliación discreta adicional de 40 ft² (PDF p.27)
        v_1flr = m.addVar(vtype=gp.GRB.BINARY, name="x_Add1stFlr")
        delta_1flr = 40.0
        cost_1flr = ct.construction_cost * delta_1flr
        m.addConstr(x["1st Flr SF"] >= first_flr + v_1flr * delta_1flr, name="AMPL_1stflr_upgrade")
        lin_cost += cost_1flr * v_1flr

    # --- 11. Vincular contadores de habitaciones/baños/cocinas ---
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

    # --- 12. Costos de ampliaciones porcentuales (PDF p.28) ---
    for c in COMPONENTES:
        lin_cost += (
            ct.ampl10_cost * delta[c][10] * z[c][10]
            + ct.ampl20_cost * delta[c][20] * z[c][20]
            + ct.ampl30_cost * delta[c][30] * z[c][30]
        )

    # =======================================================================
    # ================== FIN (ÁREA LIBRE Y AMPLIACIONES) ====================
    # =======================================================================

    # ========== GARAGE QUAL / COND ==========
    G_CATS = ["Ex","Gd","TA","Fa","Po","No aplica"]
    G_LEQ_AV = {"TA","Fa","Po"}

    gq = {g: x.get(f"garage_qual_is_{g}") for g in G_CATS if f"garage_qual_is_{g}" in x}
    gc = {g: x.get(f"garage_cond_is_{g}") for g in G_CATS if f"garage_cond_is_{g}" in x}
    upgG = x.get("UpgGarage")

    def _base_txt(col):
        raw = _na2noaplica(base_row.get(col, "No aplica"))
        # acepta 0..4 o texto
        M = {0:"Po",1:"Fa",2:"TA",3:"Gd",4:"Ex"}
        try:
            v = int(pd.to_numeric(raw, errors="coerce"))
            if v in M: return M[v]
        except Exception:
            pass
        return str(raw)

    base_qual_txt = _base_txt("Garage Qual")
    base_cond_txt = _base_txt("Garage Cond")

    def _mask_for(base_txt):  # 1 si g ≠ base
        return {g: (0.0 if g == base_txt else 1.0) for g in G_CATS}

    maskQ = _mask_for(base_qual_txt)
    maskC = _mask_for(base_cond_txt)

    if base_qual_txt == "No aplica" or base_cond_txt == "No aplica":
        for g, v in gq.items():
            if v is not None: m.addConstr(v == (1.0 if g == "No aplica" else 0.0), name=f"GQ_fix_NA_{g}")
        for g, v in gc.items():
            if v is not None: m.addConstr(v == (1.0 if g == "No aplica" else 0.0), name=f"GC_fix_NA_{g}")
        if upgG is not None: m.addConstr(upgG == 0, name="UpgGarage_disabled_for_NA")
    else:
        if gq: m.addConstr(gp.quicksum(v for v in gq.values() if v is not None) == 1, name="GQ_pick_one")
        if gc: m.addConstr(gp.quicksum(v for v in gc.values() if v is not None) == 1, name="GC_pick_one")

        def _cost(name): return ct.garage_qc_costs.get(name, 0.0)
        base_cost_q = _cost(base_qual_txt)
        base_cost_c = _cost(base_cond_txt)

        for g, v in gq.items():
            if v is None: continue
            if g == "No aplica" or _cost(g) < base_cost_q: v.UB = 0
        for g, v in gc.items():
            if v is None: continue
            if g == "No aplica" or _cost(g) < base_cost_c: v.UB = 0

        lin_cost += gp.quicksum(_cost(g) * maskQ[g] * gq[g] for g in gq if g != "No aplica")
        lin_cost += gp.quicksum(_cost(g) * maskC[g] * gc[g] for g in gc if g != "No aplica")
    # ========== FIN GARAGE QUAL / COND ==========


# ========== PAVED DRIVE ==========
    PAVED_CATS = ["Y", "P", "N"]

    # Variables binarias
    paved = {d: x[f"paved_drive_is_{d}"] for d in PAVED_CATS if f"paved_drive_is_{d}" in x}

    # Restricción: selección única
    if paved:
        m.addConstr(gp.quicksum(paved.values()) == 1, name="PAVED_pick_one")

    # Determinar categoría base
    base_pd = str(base_row.get("Paved Drive", "N")).strip()
    if base_pd not in PAVED_CATS:
        base_pd = "N"

    # Conjuntos permitidos según base
    if base_pd == "Y":
        allowed = ["Y"]
    elif base_pd == "P":
        allowed = ["P", "Y"]
    else:  # base N
        allowed = ["N", "P", "Y"]

    # Bloquear categorías no permitidas
    for d in PAVED_CATS:
        if d not in allowed and d in paved:
            paved[d].UB = 0

    # Agregar costo solo si hay cambio
    # Agregar costo solo si hay cambio
    lin_cost += gp.quicksum(
        ct.paved_drive_costs[d] * paved[d]
        for d in PAVED_CATS
        if d != base_pd and d in paved
    )

# ================== FIN (PAVED DRIVE) ==================


# ========== FENCE ==========
    FENCE_CATS = ["GdPrv", "MnPrv", "GdWo", "MnWw", "No aplica"]

    fn = {f: x[f"fence_is_{f}"] for f in FENCE_CATS if f"fence_is_{f}" in x}

    if fn:
        m.addConstr(gp.quicksum(fn.values()) == 1, name="FENCE_pick_one")

    base_f = str(base_row.get("Fence", "NA")).strip()
    if base_f not in FENCE_CATS:
        base_f = "NA"

    if base_f == "No aplica":
        allowed = ["No aplica", "MnPrv", "GdPrv"]
    elif base_f in ["GdWo", "MnWw"]:
        allowed = [base_f, "MnPrv", "GdPrv"]
    else:
        allowed = [base_f]

    for f in FENCE_CATS:
        if f not in allowed and f in fn:
            fn[f].UB = 0

    # costo por cambio de categoría
    lin_cost += gp.quicksum(
        ct.fence_category_cost(f) * fn[f]
        for f in FENCE_CATS if f != base_f and f in fn
    )

    # costo por construcción nueva
    lot_front = float(pd.to_numeric(base_row.get("Lot Frontage"), errors="coerce") or 0.0)
    if base_f == "NA":
        for f in ["MnPrv", "GdPrv"]:
            if f in fn:
                lin_cost += ct.fence_build_cost_per_ft * lot_front * fn[f]
# ================== FIN (FENCE) ==================

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
    if True:
        fp_base_txt = str(base_row.get("Fireplace Qu", "No aplica")).strip()
        has_fp = 0 if fp_base_txt in {"No aplica", "NA"} else 1

        # ¿tu XGB tiene la columna ordinal?
        has_ordinal = ("Fireplace Qu" in X_input.columns)
        # ¿o viene como OHE?
        fp_ohe_cols = [c for c in X_input.columns if c.startswith("Fireplace Qu_")]

        if has_ordinal:
            # ORDENAL esperado: -1..4
            # mapa defensivo por si en base viene 0..4 o texto
            MAP = {"Po":0,"Fa":1,"TA":2,"Gd":3,"Ex":4,"No aplica":-1,"NA":-1}
            if "Fireplace Qu" in x:
                if has_fp == 0:
                    # base sin chimenea: fijar -1 (no 2)
                    x["Fireplace Qu"].LB = -1
                    x["Fireplace Qu"].UB = -1
                else:
                    # con chimenea: permitir mantener o mejorar (no degradar)
                    try:
                        base_ord = int(pd.to_numeric(base_row.get("Fireplace Qu"), errors="coerce"))
                    except Exception:
                        base_ord = MAP.get(fp_base_txt, 2)
                    # cotas de dominio
                    x["Fireplace Qu"].LB = base_ord
                    x["Fireplace Qu"].UB = 4
            # sin costos si no hay cambio; si quieres costo por subir nivel, lo puedes mapear aquí

        elif fp_ohe_cols:
            # OHE: usa dummy explícito "No aplica"
            # crea una binaria por nivel que exista en el modelo
            levels_present = []
            for nm in ["Po","Fa","TA","Gd","Ex","No aplica"]:
                col = f"Fireplace Qu_{nm}"
                if col in X_input.columns:
                    v = m.addVar(vtype=gp.GRB.BINARY, name=f"fp_is_{nm}")
                    X_input.loc[0, col] = v  # inyecta al DF
                    levels_present.append((nm, v))
            if levels_present:
                m.addConstr(gp.quicksum(v for _, v in levels_present) == 1, name="FP_onehot")

                if has_fp == 0:
                    # base sin chimenea: forzar 'No aplica'
                    for nm, v in levels_present:
                        if nm == "No aplica":
                            v.LB = v.UB = 1.0
                        else:
                            v.UB = 0.0
                else:
                    # con chimenea: permitir mantener o mejorar (no degradar)
                    ORD = {"Po":0,"Fa":1,"TA":2,"Gd":3,"Ex":4}
                    base_ord = ORD.get(fp_base_txt, 2)
                    for nm, v in levels_present:
                        if nm == "No aplica":
                            v.UB = 0.0  # no puedes desaparecerla
                        elif ORD.get(nm, -99) < base_ord:
                            v.UB = 0.0  # no degradar

                    # si quieres costo por cambiar nivel:
                    # for nm, v in levels_present:
                    #     if nm != fp_base_txt and nm in {"Po","Fa","TA","Gd","Ex"}:
                    #         lin_cost += ct.fireplace_cost(nm) * v
        else:
            # el modelo no usa Fireplace Qu: no hacemos nada, ni costos
            pass
    # ================== FIN FIREPLACE QU ==================

    # ================== BSMT (TODO O NADA: Fin1/Fin2 vs Unf) ==================
    b1_var = x.get("BsmtFin SF 1")
    b2_var = x.get("BsmtFin SF 2")
    bu_var = x.get("Bsmt Unf SF")
    if (b1_var is None) or (b2_var is None) or (bu_var is None):
        raise RuntimeError("Faltan variables de sótano en MODIFIABLE: "
                           "'BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF'.")

    def _num(v):
        try: return float(pd.to_numeric(v, errors="coerce") or 0.0)
        except: return 0.0

    b1_base = _num(base_row.get("BsmtFin SF 1"))
    b2_base = _num(base_row.get("BsmtFin SF 2"))
    bu_base = _num(base_row.get("Bsmt Unf SF"))
    tb_base = _num(base_row.get("Total Bsmt SF"))
    if tb_base <= 0.0: tb_base = b1_base + b2_base + bu_base

    # binaria: ¿termino TODO lo no terminado?
    finish_bsmt = m.addVar(vtype=gp.GRB.BINARY, name="bsmt_finish_all")

    # reasignaciones desde Unf a 1 y 2 (no negativas)
    x1 = m.addVar(lb=0.0, name="bsmt_to_fin1")
    x2 = m.addVar(lb=0.0, name="bsmt_to_fin2")

    # si terminas, consumes todo lo Unf base
    m.addConstr(x1 + x2 == bu_base * finish_bsmt, name="BSMT_all_or_nothing")

    # enlaces de áreas
    m.addConstr(b1_var == b1_base + x1,                   name="BSMT_link_fin1")
    m.addConstr(b2_var == b2_base + x2,                   name="BSMT_link_fin2")
    m.addConstr(bu_var == bu_base * (1.0 - finish_bsmt),  name="BSMT_link_unf")

    # conservación (sanity)
    m.addConstr(b1_var + b2_var + bu_var == tb_base, name="BSMT_conservation_sum")

    # costo: solo si decides terminar (precio por ft² * Unf_base)
    lin_cost += ct.finish_basement_per_f2 * bu_base * finish_bsmt
    # ================== FIN BSMT TODO O NADA ==================


 
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


    # --- Guardar referencias para inspección desde run_opt.py ---
    m._y_price_var = y_price
    m._y_log_var = y_log
    m._base_price_val = base_price

    # -------------------
    # 6) Costos lineales (numéricos + cocina + utilities + roof)
    # -------------------
    def _num_base(col: str) -> float:
        try:
            return float(pd.to_numeric(base_row.get(col), errors="coerce"))
        except Exception:
            return 0.0

    base_vals_lin = {
        "Bedroom AbvGr": _num_base("Bedroom AbvGr"),
        "Full Bath": _num_base("Full Bath"),
        "Wood Deck SF": _num_base("Wood Deck SF"),
    }


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

        # ================== COSTOS ADICIONALES (CONSTRUCCIÓN Y DEMOLICIÓN) ==================
    try:
        # Costos base generales (por demolición de sectores y construcción)
        if hasattr(ct, "demo_cost") and hasattr(ct, "construction_cost"):
            lin_cost += ct.demo_cost  # demolición general si aplica
            lin_cost += ct.construction_cost * gp.quicksum([
                (AddFull or 0), (AddHalf or 0), (AddKitch or 0), (AddBed or 0)
            ])
        else:
            print("[WARN] CostTables no tiene demo_cost o construction_cost definidos")
    except Exception as e:
        print(f"[WARN] Error agregando costos generales: {e}")
    # ===================================================================


    # === RESTRICCIÓN: la casa remodelada no puede valer menos que la base ===
    m.addConstr(y_price >= base_price - 1e-6, name="MIN_PRICE_BASE")
    # -------------------
    # 7) Presupuesto y objetivo

    # --- Variable explícita de costo total (para depurar y reportar) ---
    cost_model = m.addVar(lb=0.0, name="cost_model")
    m.addConstr(cost_model == lin_cost, name="COST_LINK")

    # --- Restricción de presupuesto (estricta y con tolerancia baja) ---
    m.Params.FeasibilityTol = 1e-9
    m.Params.IntFeasTol = 1e-9
    m.Params.OptimalityTol = 1e-9
    m.Params.NumericFocus = 3
    m.addConstr(cost_model <= budget_usd, name="BUDGET")

# --- Función objetivo ---

    m.setObjective((y_price - lin_cost) - float(base_price), gp.GRB.MAXIMIZE)


    # --- Guardar lin_cost dentro del modelo para debug externo ---
    m._lin_cost_expr = lin_cost

    # --- Debug interno del lin_cost ---
    try:
        nterms = lin_cost.size()
        print(f"[DBG] lin_cost tiene {nterms} términos")
        if nterms > 0:
            sample = [(lin_cost.getVar(i).VarName, lin_cost.getCoeff(i)) for i in range(min(5, nterms))]
            print(f"[DBG] primeros términos: {sample}")
    except Exception as e:
        print(f"[DBG] no se pudo inspeccionar lin_cost ({type(lin_cost)}): {e}")

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
