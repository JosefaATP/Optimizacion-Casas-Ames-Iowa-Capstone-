#!/usr/bin/env python3
"""
Debug: ver qué valor tiene y_log_raw en el MIP para la casa base
"""
import pandas as pd
import numpy as np
from optimization.remodel.io import get_base_house
from optimization.remodel.xgb_predictor import XGBBundle
from optimization.remodel.gurobi_model import build_base_input_row, build_mip_embed
from optimization.remodel import costs

# Cargar
pid = 526351010
budget = 100000
base_house = get_base_house(pid)
bundle = XGBBundle()
ct = costs.CostTables()

# Input procesado (299 features)
X_input = build_base_input_row(bundle, base_house.row)

# Predicción EXTERNA
y_ext = bundle.predict(X_input).iloc[0]
y_log_raw_ext = bundle.predict_log_raw(X_input).iloc[0]

print("=" * 70)
print("EXTERNAL PREDICTOR (sklearn + booster)")
print("=" * 70)
print(f"predict(X_input) = {y_ext:.6f}")
print(f"predict_log_raw(X_input) = {y_log_raw_ext:.6f}")
print(f"expm1(y_log_raw) = {np.expm1(y_log_raw_ext):.6f}")
print(f"b0_offset = {bundle.b0_offset:.6f}")
print(f"y_log_raw + b0 = {y_log_raw_ext + bundle.b0_offset:.6f}")

# Construir MIP (sin fix_to_base para que optimize)
m = build_mip_embed(base_house.row, budget, ct, bundle, base_price=y_ext, fix_to_base=False)
m.optimize()

print("\n" + "=" * 70)
print("MIP EMBEDDING (gurobi_ml + booster embedding)")
print("=" * 70)

# Valores del MIP en la base
if hasattr(m, '_y_log_raw_var'):
    y_log_raw_mip = m._y_log_raw_var.X
    print(f"y_log_raw (MIP) = {y_log_raw_mip:.6f}")
else:
    y_log_raw_mip = None
    print(f"y_log_raw (MIP) = NOT FOUND")

if hasattr(m, '_y_log_var'):
    y_log_mip = m._y_log_var.X
    print(f"y_log (MIP) = {y_log_mip:.6f}")
else:
    y_log_mip = None
    print(f"y_log (MIP) = NOT FOUND")

if hasattr(m, '_y_price_var'):
    y_price_mip = m._y_price_var.X
    print(f"y_price (MIP) = {y_price_mip:.6f}")
    print(f"expm1(y_log) = {np.expm1(y_log_mip):.6f}")

print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
if y_log_raw_mip is not None:
    print(f"y_log_raw delta: {y_log_raw_mip - y_log_raw_ext:.6f}")
if y_log_mip is not None:
    print(f"y_log delta: {y_log_mip - (y_log_raw_ext + bundle.b0_offset):.6f}")
