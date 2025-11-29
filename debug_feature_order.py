#!/usr/bin/env python3
"""
Debug: Verificar si el orden de features es correcto
"""
import pandas as pd
import numpy as np
from optimization.remodel.io import get_base_house
from optimization.remodel.xgb_predictor import XGBBundle
from optimization.remodel.gurobi_model import build_base_input_row

pid = 526351010
base_house = get_base_house(pid)
bundle = XGBBundle()

X_input = build_base_input_row(bundle, base_house.row)

# 1. Orden de features según X_input
print("=" * 70)
print("Orden de features en X_input:")
print("=" * 70)
for i, c in enumerate(X_input.columns):
    print(f"[{i:3d}] {c:25s} = {X_input.iloc[0, i]:.6f}")
    if i == 10:
        print(f"... ({len(X_input.columns) - 10} más)")
        break

# 2. Orden de features según booster_feature_order()
print("\n" + "=" * 70)
print("Orden de features según booster_feature_order():")
print("=" * 70)
booster_order = bundle.booster_feature_order()
for i, c in enumerate(booster_order):
    val = X_input.loc[0, c]
    print(f"[{i:3d}] {c:25s} = {val:.6f}")
    if i == 10:
        print(f"... ({len(booster_order) - 10} más)")
        break

# 3. ¿Son idénticos?
print("\n" + "=" * 70)
print("¿Coinciden los órdenes?")
print("=" * 70)
x_input_cols = list(X_input.columns)
match = x_input_cols == booster_order
print(f"X_input columns == booster_order: {match}")

if not match:
    # Encontrar diferencias
    for i in range(min(len(x_input_cols), len(booster_order))):
        if x_input_cols[i] != booster_order[i]:
            print(f"  Posición {i}: '{x_input_cols[i]}' vs '{booster_order[i]}'")
            if i > 5:
                print(f"  ... ({sum(1 for j in range(len(x_input_cols)) if x_input_cols[j] != booster_order[j]) - 5} más diferencias)")
                break
