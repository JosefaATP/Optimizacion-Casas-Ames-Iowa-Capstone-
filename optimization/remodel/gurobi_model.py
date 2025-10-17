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

    total_cost = lin_cost
    m.addConstr(total_cost <= budget, name="budget")

    if "2nd Flr SF" in x and "1st Flr SF" in x:
        m.addConstr(x["2nd Flr SF"] <= x["1st Flr SF"], name="floor2_le_floor1")
    if "Garage Area" in x and "Garage Cars" in x:
        m.addConstr(x["Garage Area"] >= 150 * x["Garage Cars"], name="garage_min_area")
        m.addConstr(x["Garage Area"] <= 250 * x["Garage Cars"], name="garage_max_area")

    initial_cost = ct.initial_cost(base_row)
    m.setObjective(y_price - total_cost, gp.GRB.MAXIMIZE)
    return m
