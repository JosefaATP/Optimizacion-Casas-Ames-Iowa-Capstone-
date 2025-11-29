#!/usr/bin/env python3
"""
Inspecciona qué variables aparecen en lin_cost exactamente.
Esto ayuda a entender por qué algunas renovaciones marcan como "free".
"""
import sys
sys.path.insert(0, r"c:\Users\valen\OneDrive - uc.cl\UNIVERSIDAD\2024-1\Bases de datos\Optimizacion-Casas-Ames-Iowa-Capstone-")

from optimization.remodel import gurobi_model, config, costs
from optimization.remodel.xgb_predictor import XGBBundle
from optimization.remodel.io import get_base_house
import pandas as pd

pid = 526351010

# Load data
base_house = get_base_house(pid)
base_row = base_house.row if hasattr(base_house, 'row') else base_house

# Load bundle and costs
bundle = XGBBundle()
ct = costs.CostTables()

# Get base price
X_base = gurobi_model.build_base_input_row(bundle, base_row)
precio_base = float(bundle.predict(X_base).iloc[0])

# Build model
m = gurobi_model.build_mip_embed(
    base_row=base_row,
    budget=100_000,
    ct=ct,
    bundle=bundle,
    base_price=precio_base,
    fix_to_base=False,
)

# 1. Dump what's in lin_cost
print("=" * 80)
print("VARIABLES IN lin_cost EXPRESSION")
print("=" * 80)
expr = m._lin_cost_expr
if expr is not None:
    try:
        vs = expr.getVars()
        cs = expr.getCoeffs()
        print(f"Total terms: {len(vs)}")
        print()
        for v, c in sorted(zip(vs, cs), key=lambda x: -abs(float(x[1]))):
            coeff = float(c)
            if abs(coeff) > 0.01:  # Solo mostrar significativos
                print(f"  {v.VarName:<50s}  coeff={coeff:>12,.2f}")
    except Exception as e:
        print(f"Error reading lin_cost: {e}")

print()
print("=" * 80)
print("ALL GUROBI VARIABLES (with non-zero coeff in lin_cost)")
print("=" * 80)

cost_vars_set = set()
try:
    for v, c in zip(expr.getVars(), expr.getCoeffs()):
        if abs(float(c)) > 0:
            cost_vars_set.add(v.VarName)
except Exception:
    pass

print(f"Total variables WITH cost in lin_cost: {len(cost_vars_set)}")
print()

all_vars = m.getVars()
print(f"Total Gurobi variables in model: {len(all_vars)}")
print()

print("Variables WITH cost (top 30):")
cost_var_list = [v for v in sorted(all_vars, key=lambda x: x.VarName) if v.VarName in cost_vars_set]
for v in cost_var_list[:30]:
    print(f"  {v.VarName}")

if len(cost_var_list) > 30:
    print(f"  ... and {len(cost_var_list) - 30} more")

print()
print("Variables WITHOUT cost (first 40, sorted):")
no_cost = [v for v in all_vars if v.VarName not in cost_vars_set]
for v in sorted(no_cost, key=lambda x: x.VarName)[:40]:
    print(f"  {v.VarName}")
