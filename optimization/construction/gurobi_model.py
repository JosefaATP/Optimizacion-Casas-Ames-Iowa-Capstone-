#optimization/construction/gurobi_model.py
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


    # ================== RESTRICCIONES MINIMAS ==================
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

    # ================== PREDICTOR XGB (embedding) ==================
    # Embebo el pipeline (pre aplanado + regressor). NO paso salida, dejo que gurobi_ml la cree.
    pc = _add_sklearn(m, bundle.pipe_for_gurobi(), X_input)

    cand_attrs = ["output_vars","prediction_vars","output","predictions","y"]
    found = []
    for att in cand_attrs:
        if hasattr(pc, att):
            try:
                obj = getattr(pc, att)
                # intenta indexar como matriz/lista
                try:
                    v = obj[0][0]
                except Exception:
                    try:
                        v = obj[0]
                    except Exception:
                        v = obj
                if hasattr(v, "VarName"):
                    found.append((att, v))
            except Exception:
                pass

    print("[CHK-PC] candidatos a salida:", [a for a,_ in found])
    if found:
        y_emb = found[0][1]
    else:
        # fallback legacy
        try:
            y_emb = pc.output_vars[0][0]
        except Exception:
            try:
                y_emb = pc.output[0][0]
            except Exception:
                try:
                    y_emb = pc.y[0][0]
                except Exception:
                    raise RuntimeError("No pude capturar la salida del embed (gurobi_ml).")
    
    m._embed_pc = pc
    m._feat_order = list(bundle.feature_names_in())

    # Creo mi y_log “oficial” y lo ligo al del embed
    y_log = m.addVar(lb=-gp.GRB.INFINITY, name="y_log")
    m.addConstr(y_log == y_emb, name="LINK_ylog_embed")


    # Si el target está en log, genero y_price con PWL(exp)
    if bundle.is_log_target():
        y_price = m.addVar(lb=0.0, name="y_price")
        grid = np.linspace(10.0, 14.0, 161)
        m.addGenConstrPWL(y_log, y_price, grid.tolist(), np.expm1(grid).tolist(), name="exp_expm1")
    else:
        y_price = y_log

    # Exponer handlers para debug
    m._y_log_embed_var = y_emb
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


