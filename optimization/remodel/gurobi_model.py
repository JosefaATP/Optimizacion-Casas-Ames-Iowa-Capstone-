#gurobi_model.py
from typing import Dict, Any
import pandas as pd
import gurobipy as gp
import numpy as np

from sklearn.compose import ColumnTransformer

# >>> EL SHIM DEBE IR ANTES DE CUALQUIER IMPORT DE gurobi_ml <<<
from .compat_sklearn import ensure_check_feature_names
ensure_check_feature_names()

from .compat_xgboost import patch_get_booster
patch_get_booster()

# Import compatible para distintas versiones de gurobi-ml
try:
    from gurobi_ml.sklearn.pipeline import add_pipeline_constr as _add_sklearn
except Exception:
    from gurobi_ml.sklearn import add_predictor_constr as _add_sklearn

from .features import MODIFIABLE, IMMUTABLE
from .costs import CostTables
from .xgb_predictor import XGBBundle



def _vtype(code: str):
    return gp.GRB.CONTINUOUS if code == "C" else (gp.GRB.BINARY if code == "B" else gp.GRB.INTEGER)


def build_mip_embed(base_row: pd.Series, budget: float, ct: CostTables, bundle: XGBBundle) -> gp.Model:
    m = gp.Model("remodel_embed")

    # decision vars
    x: dict[str, gp.Var] = {}
    for f in MODIFIABLE:
        x[f.name] = m.addVar(lb=f.lb, ub=f.ub, vtype=_vtype(f.vartype), name=f"x_{f.name}")

    # === ENTRADA AL PIPELINE (DataFrame 1xN, en el orden que espera) ===
    feature_order = bundle.feature_names_in()
    modifiable_names = {f.name for f in MODIFIABLE}
    row_vals: Dict[str, Any] = {}

    for fname in feature_order:
        if fname in modifiable_names:
            row_vals[fname] = x[fname]           # var de decisión
        else:
            row_vals[fname] = base_row[fname]  # conservar dtype original del dataset


    X_input = pd.DataFrame([row_vals], columns=feature_order)
    
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline as SKPipeline

    def _align_ohe_dtypes(X: pd.DataFrame, pre: ColumnTransformer) -> pd.DataFrame:
        trs = pre.transformers_ if hasattr(pre, "transformers_") else pre.transformers
        X2 = X.copy()
        for item in trs:
            name, transformer, cols = item[0], item[1], item[2]
            est = transformer.steps[-1][1] if isinstance(transformer, SKPipeline) else transformer
            if isinstance(est, OneHotEncoder):
                for c in cols:
                    if c in X2.columns:
                        # fuerza str para que coincida con categories_ parcheadas
                        X2[c] = X2[c].astype(str)
        return X2

    X_input = pd.DataFrame([row_vals], columns=feature_order)
    X_input = _align_ohe_dtypes(X_input, bundle.pre)  # 👈 nuevo

    #==================================================================
    #=== RESTRICCIONES DE CALIDAD ================================
    # === R9-KITCHEN: Mejora de calidad sólo a TA o Ex, a lo más un paquete ===
    def _q_to_ord(v):
        mapping = {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4}
        try:
            return int(v)
        except Exception:
            return mapping[str(v)]

    kq_base = _q_to_ord(base_row.get("Kitchen Qual", "TA"))  # default defensivo

    # Vars binarias que declaraste en features.py
    dTA = x["delta_KitchenQual_TA"]
    dEX = x["delta_KitchenQual_EX"]

    # A lo más un paquete
    m.addConstr(dTA + dEX <= 1, name="R9_kitchen_at_most_one_pkg")

    # Si la base ya es >= TA, no ofertar paquete TA (sería 'no-op')
    if kq_base >= 2:
        dTA.UB = 0
    # Si la base ya es EX, no ofertar nada
    if kq_base >= 4:
        dEX.UB = 0
    #==================================================================

    # === salida del predictor: LOG(PRECIO) porque pipe_for_gurobi es (pre -> XGB) ===
    y_log = m.addVar(lb=-gp.GRB.INFINITY, name="y_log")

    _add_sklearn(
        m,
        bundle.pipe_for_gurobi(),   # (pre -> XGBRegressor) SIN TTR
        X_input,                    # DataFrame (no dict)
        [y_log],
    )

    # === convertir log->precio con PWL expm1 ===
    if bundle.is_log_target():
        y_price = m.addVar(lb=0.0, name="y_price")
        z_min, z_max = 10.0, 14.0
        grid = np.linspace(z_min, z_max, 81)
        m.addGenConstrPWL(y_log, y_price, grid.tolist(), np.expm1(grid).tolist(), name="exp_expm1")
    else:
        y_price = y_log

    # --- costos (igual que tenías) ---
    base_vals = {f.name: float(base_row.get(f.name, 0.0)) for f in MODIFIABLE}
    lin_cost = gp.LinExpr(ct.project_fixed)
    # después de construir lin_cost:
    total_cost_var = m.addVar(lb=0.0, name="total_cost")
    m.addConstr(total_cost_var == lin_cost, name="def_total_cost")
    total_cost = total_cost_var  # usa esta en el objetivo y en prints

    def pos(expr):
        v = m.addVar(lb=0.0, name=f"pos_{len(m.getVars())}")
        m.addConstr(v >= expr)
        return v

    if "Bedroom AbvGr" in x:
        lin_cost += pos(x["Bedroom AbvGr"] - base_vals.get("Bedroom AbvGr", 0.0)) * ct.add_bedroom
    if "Full Bath" in x:
        lin_cost += pos(x["Full Bath"] - base_vals.get("Full Bath", 0.0)) * ct.add_bathroom
    if "Wood Deck SF" in x:
        lin_cost += pos(x["Wood Deck SF"] - base_vals.get("Wood Deck SF", 0.0)) * ct.deck_per_m2
    if "Garage Cars" in x:
        lin_cost += pos(x["Garage Cars"] - base_vals.get("Garage Cars", 0.0)) * ct.garage_per_car
    if "Total Bsmt SF" in x:
        lin_cost += pos(x["Total Bsmt SF"] - base_vals.get("Total Bsmt SF", 0.0)) * ct.finish_basement_per_m2
    
    lin_cost += dTA * ct.KitchenQual_upgrade_TA
    lin_cost += dEX * ct.KitchenQual_upgrade_EX

    total_cost = lin_cost
    m.addConstr(total_cost <= budget, name="budget")


    '''if "Garage Area" in x and "Garage Cars" in x:
        m.addConstr(x["Garage Area"] >= 150 * x["Garage Cars"], name="garage_min_area")
        m.addConstr(x["Garage Area"] <= 250 * x["Garage Cars"], name="garage_max_area")'''

    initial_cost = ct.initial_cost(base_row)


    #------------------------------------
    #RESTRICCIONES#------------------------------------
    #------------------------------------

    # === Restricciones "ideas" (§5) ===========================================
    # (1) 1stFlrSF ≥ 2ndFlrSF
    if "1st Flr SF" in x and "2nd Flr SF" in x:
        m.addConstr(x["1st Flr SF"] >= x["2nd Flr SF"], name="R1_floor1_ge_floor2")  # [R1] §5.1

    # (2) GrLivArea ≤ LotArea  (si Gr Liv Area es variable, comparar var vs parámetro; si no, usar base)
    if "Gr Liv Area" in x and "Lot Area" in base_row:
        m.addConstr(x["Gr Liv Area"] <= float(base_row["Lot Area"]), name="R2_grliv_le_lot")  # [R2] §5.2
    elif "Gr Liv Area" not in x and "Lot Area" in base_row:
        # Si 'Gr Liv Area' no es variable, revisa que la base respete (o déjala como soft en otra iteración)
        pass

    # (3) 1stFlrSF ≥ TotalBsmtSF
    if "1st Flr SF" in x and "Total Bsmt SF" in x:
        m.addConstr(x["1st Flr SF"] >= x["Total Bsmt SF"], name="R3_floor1_ge_bsmt")  # [R3] §5.3

    # (4) FullBath + HalfBath ≤ Bedroom (arriba de nivel)
    # Usa variable si existe; si no, valor base para la que falte.
    def _val_or_var(col):
        return x[col] if col in x else float(base_row[col])

    need = all(c in base_row for c in ["Full Bath", "Bedroom AbvGr"])  # Half Bath puede faltar a veces
    if need and ("Half Bath" in base_row or "Half Bath" in x):
        fullb   = _val_or_var("Full Bath")
        halfb   = _val_or_var("Half Bath") if ("Half Bath" in x or "Half Bath" in base_row) else 0.0
        beds    = _val_or_var("Bedroom AbvGr")
        m.addConstr(fullb + halfb <= beds, name="R4_baths_le_bedrooms")  # [R4] §5.4

    # (R5) Mínimos: FullBath ≥ 1, BedroomAbvGr ≥ 1, KitchenAbvGr ≥ 1
    def _val_or_var(col):
        return x[col] if col in x else float(base_row[col])

    if ("Full Bath" in x) or ("Full Bath" in base_row):
        m.addConstr(_val_or_var("Full Bath") >= 1, name="R5_min_fullbath")
    if ("Bedroom AbvGr" in x) or ("Bedroom AbvGr" in base_row):
        m.addConstr(_val_or_var("Bedroom AbvGr") >= 1, name="R5_min_bedrooms")
    if ("Kitchen AbvGr" in x) or ("Kitchen AbvGr" in base_row):
        m.addConstr(_val_or_var("Kitchen AbvGr") >= 1, name="R5_min_kitchen")

    # === R7. Consistencia de superficie habitable sobre rasante (PDF §7) ===
    # Gr Liv Area = 1st Flr SF + 2nd Flr SF + Low Qual Fin SF (si existe)
    lowqual_names = ["Low Qual Fin SF", "LowQualFinSF"]
    lowqual_col = next((c for c in lowqual_names if c in X_input.columns), None)

    cols_needed = ["Gr Liv Area", "1st Flr SF", "2nd Flr SF"]
    if all(c in X_input.columns for c in cols_needed):

        def _val(name: str):
            # usa var de decisión si es modificable; si no, valor base
            return x[name] if name in x else float(base_row[name])

        lhs = _val("Gr Liv Area")
        rhs = _val("1st Flr SF") + _val("2nd Flr SF")
        if lowqual_col is not None:
            rhs = rhs + _val(lowqual_col)

        m.addConstr(lhs == rhs, name="R7_gr_liv_equality")

    
    # === R8. Consistencia de habitaciones (PDF §8) ===
    # Creamos una variable auxiliar "other rooms" ≥ 0 (entera).
    r8_ok = all(c in X_input.columns for c in ["TotRms AbvGrd", "Bedroom AbvGr", "Kitchen AbvGr"])
    if r8_ok:
        def _val(name: str):
            return x[name] if name in x else float(base_row[name])

        other_max = 15  # cota superior razonable para evitar no acotación
        other = m.addVar(lb=0, ub=other_max, vtype=gp.GRB.INTEGER, name="R8_other_rooms")

        # Igualdad TotRmsAbvGrd = Bedroom + Kitchen + Otras
        m.addConstr(
            _val("TotRms AbvGrd") == _val("Bedroom AbvGr") + _val("Kitchen AbvGr") + other,
            name="R8_rooms_balance"
        )


    # ==========================================================================

    m.setObjective(y_price - total_cost, gp.GRB.MAXIMIZE)
    return m
