#!/usr/bin/env python3
"""
Debug exhaustivo: rastrear todos los valores desde X_input hasta y_log_raw en el MIP
"""
import pandas as pd
import numpy as np
from optimization.remodel.io import get_base_house
from optimization.remodel.xgb_predictor import XGBBundle
from optimization.remodel.gurobi_model import build_base_input_row
from optimization.remodel import costs
import gurobipy as gp

# ============================================================================
# PASO 1: Construcción de X_input
# ============================================================================
print("=" * 80)
print("PASO 1: Construcción de X_input")
print("=" * 80)

pid = 526351010
budget = 100000
base_house = get_base_house(pid)
bundle = XGBBundle()
ct = costs.CostTables()

X_input = build_base_input_row(bundle, base_house.row)
print(f"X_input shape: {X_input.shape}")
print(f"X_input columns: {len(X_input.columns)}")
print(f"X_input dtypes unique: {X_input.dtypes.unique()}")
print(f"X_input values (sample): {X_input.iloc[0, :10].to_dict()}")

# Verificar que no hay NaN
nan_count = X_input.isna().sum().sum()
print(f"NaN count in X_input: {nan_count}")

# Verificar que todos son flotantes
all_numeric = X_input.applymap(lambda x: isinstance(x, (int, float, np.number))).all().all()
print(f"All values numeric: {all_numeric}")

# ============================================================================
# PASO 2: Predicción externa correcta
# ============================================================================
print("\n" + "=" * 80)
print("PASO 2: Predicción externa (referencia correcta)")
print("=" * 80)

y_ext = bundle.predict(X_input).iloc[0]
y_log_raw_ext = bundle.predict_log_raw(X_input).iloc[0]
b0 = bundle.b0_offset

print(f"External predict(X_input) = {y_ext:.6f}")
print(f"External predict_log_raw(X_input) = {y_log_raw_ext:.6f}")
print(f"bundle.b0_offset = {b0:.6f}")
print(f"Expected y_log = y_log_raw + b0 = {y_log_raw_ext + b0:.6f}")
print(f"Verify: expm1({y_log_raw_ext:.6f}) = {np.expm1(y_log_raw_ext):.6f}")

# ============================================================================
# PASO 3: Crear MIP manualmente y verificar variables
# ============================================================================
print("\n" + "=" * 80)
print("PASO 3: Construcción MIP - Verificar variables de base")
print("=" * 80)

m = gp.Model("debug_embed")
booster_order = bundle.booster_feature_order()

# Crear variables igual a gurobi_model.py
feat_full = list(X_input.columns)
print(f"Features en X_input: {len(feat_full)}")

var_by_name = {}
sample_count = 0
for c in feat_full:
    val = X_input.loc[0, c]
    num = float(val)
    safe_c = c.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
    v = m.addVar(lb=num, ub=num, name=f"v_{safe_c}")
    var_by_name[c] = v

m.update()  # Necesario para Gurobi

sample_count = 0
for c in feat_full:
    v = var_by_name[c]
    num = float(X_input.loc[0, c])
    if sample_count < 5:
        print(f"  {c}: LB={v.LB} UB={v.UB} (from X_input={num:.6f})")
        sample_count += 1

print(f"Total variables created: {len(var_by_name)}")
print(f"Total features in booster_order: {len(booster_order)}")

# Verificar alineación
missing = [f for f in booster_order if f not in var_by_name]
extra = [f for f in var_by_name if f not in booster_order]
print(f"Features en booster_order pero no en var_by_name: {len(missing)}")
print(f"Features en var_by_name pero no en booster_order: {len(extra)}")

# ============================================================================
# PASO 4: Crear lista x_vars en el mismo orden que booster_order
# ============================================================================
print("\n" + "=" * 80)
print("PASO 4: Crear x_vars en orden de booster_feature_order")
print("=" * 80)

x_vars = []
for fname in booster_order:
    v = var_by_name.get(fname)
    if v is None:
        print(f"  WARNING: {fname} not found in var_by_name, creating dummy fixed at 0")
        v = m.addVar(lb=0.0, ub=0.0, name=f"missing_{fname}")
    x_vars.append(v)

print(f"x_vars length: {len(x_vars)}")
print(f"First 5 x_vars bounds:")
for i in range(min(5, len(x_vars))):
    print(f"  [{i}] {booster_order[i]}: LB={x_vars[i].LB} UB={x_vars[i].UB}")

# ============================================================================
# PASO 5: Embeber árboles (sin optimizar aún)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 5: Embeber árboles XGB")
print("=" * 80)

y_log_raw = m.addVar(lb=-gp.GRB.INFINITY, name="y_log_raw")
print(f"Created y_log_raw variable")

bundle.attach_to_gurobi(m, x_vars, y_log_raw, eps=-1e-6)
print(f"attach_to_gurobi() completed")

# ============================================================================
# PASO 6: Optimizar con objetivo dummy y evaluar y_log_raw en la base
# ============================================================================
print("\n" + "=" * 80)
print("PASO 6: Optimizar MIP (sin objetivo, solo feasibility)")
print("=" * 80)

# Objetivo dummy: minimizar costo cero (no cambiar nada)
m.setObjective(0, gp.GRB.MINIMIZE)
m.optimize()

if m.status == gp.GRB.OPTIMAL or m.status == gp.GRB.SUBOPTIMAL:
    y_log_raw_mip = y_log_raw.X
    print(f"\ny_log_raw (MIP) = {y_log_raw_mip:.6f}")
    print(f"Expected = {y_log_raw_ext:.6f}")
    print(f"Delta = {y_log_raw_mip - y_log_raw_ext:.6f}")
    
    # Ver primeras 5 variables
    print(f"\nVariable values in solution:")
    for i in range(min(5, len(x_vars))):
        print(f"  [{i}] {booster_order[i]}: X={x_vars[i].X:.6f} (expected={X_input.iloc[0, i]:.6f})")
else:
    print(f"MIP status: {m.status} (not optimal)")
    if hasattr(y_log_raw, 'X'):
        print(f"y_log_raw.X = {y_log_raw.X}")
    else:
        print(f"y_log_raw has no X value")
