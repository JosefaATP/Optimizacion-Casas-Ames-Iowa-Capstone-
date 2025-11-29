#!/usr/bin/env python3
"""
Inspecciona qué variables aparecen en lin_cost exactamente.
Esto ayuda a entender por qué algunas renovaciones marcan como "free".
"""
import sys
sys.path.insert(0, r"c:\Users\valen\OneDrive - uc.cl\UNIVERSIDAD\2024-1\Bases de datos\Optimizacion-Casas-Ames-Iowa-Capstone-")

from optimization.remodel import gurobi_model
from optimization.remodel import config
import pandas as pd

# Load data
base_path = config.PATHS.base_csv
base_df = pd.read_csv(base_path)

pid = 526351010
if pid not in base_df['PID'].values:
    print(f"PID {pid} not found in base_df")
    print(f"Available PIDs: {base_df['PID'].values[:10]}")
    sys.exit(1)

base_row = base_df[base_df['PID'] == pid].iloc[0]

# Build model
m = gurobi_model.build_model(
    base_row=base_row,
    base_df=base_df,
    budget_usd=100_000,
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

print("Variables WITH cost:")
for v in sorted(all_vars, key=lambda x: x.VarName):
    if v.VarName in cost_vars_set:
        print(f"  {v.VarName}")

print()
print("Variables WITHOUT cost (cost=0):")
no_cost = [v for v in all_vars if v.VarName not in cost_vars_set]
for v in sorted(no_cost, key=lambda x: x.VarName)[:30]:  # primeros 30
    print(f"  {v.VarName}")

if len(no_cost) > 30:
    print(f"  ... and {len(no_cost) - 30} more")

print()
print("=" * 80)
print("FEATURE COLUMNS IN X (should all be 0/1 for dummies)")
print("=" * 80)
if hasattr(m, "_x_cols"):
    print(f"Total feature columns: {len(m._x_cols)}")
    # Muestra algunos ejemplos
    for fname in list(m._x_cols)[:10]:
        var = m._x_vars.get(fname)
        if hasattr(var, "VarName"):
            print(f"  {fname:<50s} -> Gurobi var: {var.VarName}")
        else:
            print(f"  {fname:<50s} -> NOT a Gurobi var (constant)")
