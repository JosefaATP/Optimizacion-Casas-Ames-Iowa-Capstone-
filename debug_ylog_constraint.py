#!/usr/bin/env python3
"""
Debug: Inspeccionar la restricción YLOG_XGB_SUM
"""
import pandas as pd
import numpy as np
from optimization.remodel.io import get_base_house
from optimization.remodel.xgb_predictor import XGBBundle
from optimization.remodel.gurobi_model import build_base_input_row
import gurobipy as gp

pid = 526351010
base_house = get_base_house(pid)
bundle = XGBBundle()
X_input = build_base_input_row(bundle, base_house.row)

m = gp.Model()
booster_order = bundle.booster_feature_order()

# Crear variables
var_by_name = {}
for c in X_input.columns:
    num = float(X_input.loc[0, c])
    safe_c = c.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
    v = m.addVar(lb=num, ub=num, name=f"v_{safe_c}")
    var_by_name[c] = v

m.update()

x_vars = [var_by_name[c] for c in booster_order]

# Embeber árboles
y_log_raw = m.addVar(lb=-gp.GRB.INFINITY, name="y_log_raw")
bundle.attach_to_gurobi(m, x_vars, y_log_raw, eps=-1e-6)

# Inspeccionar restricción YLOG_XGB_SUM
print("=" * 70)
print("Inspeccionando restricción YLOG_XGB_SUM")
print("=" * 70)

for constr in m.getConstrs():
    if "YLOG_XGB_SUM" in constr.ConstrName:
        print(f"\nConstraint: {constr.ConstrName}")
        print(f"  Sense: {constr.Sense}")
        print(f"  RHS: {constr.RHS}")
        print(f"  Expression: {constr.getAttr(gp.GRB.Attr.ConstrExpr)}")
        
# Optimizar y ver resultado
m.setObjective(0, gp.GRB.MINIMIZE)
m.optimize()

print(f"\n" + "=" * 70)
print(f"Después de optimize:")
print(f"=" * 70)
print(f"y_log_raw.X = {y_log_raw.X}")
print(f"y_log_raw Expected = 12.388893")

# Evaluar manualmente predict_log_raw en el estado actual
X_test = X_input.copy()
y_log_ext = float(bundle.predict_log_raw(X_test).iloc[0])
print(f"External predict_log_raw() = {y_log_ext}")

# Chequear valores de variables
print(f"\nPrimeras 5 variables en la solución:")
for i in range(5):
    v = x_vars[i]
    print(f"  {booster_order[i]}: {v.X:.6f}")
