#!/usr/bin/env python3
"""
Test: ¿Coinciden las 299 features entre la reconstrucción manual y booster_feature_order()?
"""
import pandas as pd
from optimization.remodel.io import get_base_house
from optimization.remodel.xgb_predictor import XGBBundle
from optimization.remodel.gurobi_model import build_base_input_row

# 1. Cargar casa + bundle
pid = 526351010
base_house = get_base_house(pid)
bundle = XGBBundle()

# 2. Procesar con build_base_input_row (la misma función que MIP usa)
X_base = build_base_input_row(bundle, base_house.row)
print(f"Raw input: {base_house.row.shape} features")
print(f"After build_base_input_row(): {X_base.shape}")

# Predicción con bundle
y_pred = bundle.predict(X_base)
print(f"bundle.predict(X_base) = {float(y_pred.iloc[0]):.6f}")

# 3. Procesar con bundle.pre.transform (el pipeline entrenado)
X_preprocessed = bundle.pre.transform(X_base)
print(f"\n[1] After bundle.pre.transform():")
print(f"    Shape: {X_preprocessed.shape}")
print(f"    Type: {type(X_preprocessed)}")

# Si es sparse, convertir a dense
if hasattr(X_preprocessed, 'toarray'):
    X_preprocessed_dense = X_preprocessed.toarray()
else:
    X_preprocessed_dense = X_preprocessed

# Obtener nombres de features del booster
feature_order = bundle.booster_feature_order()
print(f"\n[2] booster_feature_order():")
print(f"    Count: {len(feature_order)}")
print(f"    First 10: {feature_order[:10]}")

# Comparar
print(f"\n[3] X_base vs booster_feature_order():")
print(f"    X_base columns: {len(X_base.columns)}")
print(f"    booster order:  {len(feature_order)}")
print(f"    Columns match: {list(X_base.columns) == feature_order}")

# ¿Las columnas de X_base son exactamente las mismas?
if list(X_base.columns) != feature_order:
    print(f"\n    ❌ MISMATCH in columns!")
    # Ver cuáles faltan
    in_base_not_order = set(X_base.columns) - set(feature_order)
    in_order_not_base = set(feature_order) - set(X_base.columns)
    if in_base_not_order:
        print(f"    In X_base but not in booster order: {in_base_not_order}")
    if in_order_not_base:
        print(f"    In booster order but not in X_base: {in_order_not_base}")
else:
    print(f"\n    ✓ Column names and order MATCH!")
