from typing import Dict, Any
import pandas as pd
import gurobipy as gp
import numpy as np

# >>> EL SHIM DEBE IR ANTES DE CUALQUIER IMPORT DE gurobi_ml <<<
from .compat_sklearn import ensure_check_feature_names
ensure_check_feature_names()

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

    # armar input_vars: tomar TODAS las features que espera el pipeline
    feats: Dict[str, Any] = {}
    for fname in bundle.feature_names_in():
        if fname in {f.name for f in MODIFIABLE}:
            continue
        val = base_row[fname]
        # intento de cast a float; si falla, lo dejamos como string
        try:
            feats[fname] = float(val)
        except Exception:
            feats[fname] = str(val)

    # añadir decision vars y chequear que existan en el modelo
    for f in MODIFIABLE:
        if f.name not in bundle.feature_names_in():
            raise KeyError(
                f"La variable modificable '{f.name}' no existe en las features del modelo. "
                f"Revisa el nombre exacto en el CSV/modelo."
            )
        feats[f.name] = x[f.name]


    # salida del predictor en escala ORIGINAL (porque usamos el Pipeline completo con TTR)
    y_price = m.addVar(lb=-gp.GRB.INFINITY, name="y_price")

    # conectar pipeline(pre -> TTR(XGB)) ya fitted (usar posicionales)
    _add_sklearn(
        m,
        bundle.pipe_for_gurobi(),   # == pipeline completo que cargaste del joblib
        feats,
        [y_price],
    )

    # costos de remodelacion (placeholder por delta vs base)
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

    # presupuesto
    m.addConstr(total_cost <= budget, name="budget")

    # ejemplos de restricciones
    if "2nd Flr SF" in x and "1st Flr SF" in x:
        m.addConstr(x["2nd Flr SF"] <= x["1st Flr SF"], name="floor2_le_floor1")
    if "Garage Area" in x and "Garage Cars" in x:
        m.addConstr(x["Garage Area"] >= 150 * x["Garage Cars"], name="garage_min_area")
    if "Garage Area" in x and "Garage Cars" in x:
        m.addConstr(x["Garage Area"] <= 250 * x["Garage Cars"], name="garage_max_area")

    # objetivo: max rentabilidad = y_price - costo_remodelacion - costo_inicial(base)
    initial_cost = ct.initial_cost(base_row)
    m.setObjective(y_price - total_cost - initial_cost, gp.GRB.MAXIMIZE)

    return m
