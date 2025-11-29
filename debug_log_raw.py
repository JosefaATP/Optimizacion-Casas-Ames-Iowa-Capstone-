#!/usr/bin/env python3
"""
Debug: ver exactamente qué devuelve predict_log_raw()
"""
import pandas as pd
import numpy as np
from optimization.remodel.io import get_base_house
from optimization.remodel.xgb_predictor import XGBBundle
from optimization.remodel.gurobi_model import build_base_input_row

# Cargar
pid = 526351010
base_house = get_base_house(pid)
bundle = XGBBundle()

# Input procesado (299 features)
X_input = build_base_input_row(bundle, base_house.row)
print(f"X_input shape: {X_input.shape}")
print(f"X_input columns: {len(X_input.columns)}")

# Predicción EXTERNA
y_ext = bundle.predict(X_input).iloc[0]
print(f"\nbundle.predict(X_input) = {y_ext:.6f}")

# Predicción LOG RAW EXTERNA
y_log_raw = bundle.predict_log_raw(X_input).iloc[0]
print(f"bundle.predict_log_raw(X_input) = {y_log_raw:.6f}")

# Verificación: ¿coinciden mediante expm1?
y_back = np.expm1(y_log_raw)
print(f"\nexpm1({y_log_raw:.6f}) = {y_back:.6f}")
print(f"predict() = {y_ext:.6f}")
print(f"Match: {np.isclose(y_back, y_ext, rtol=1e-3)}")

# Ahora: comparar con lo que devuelve el booster directamente
bst = bundle.reg.get_booster()
Xp = np.asarray(X_input.values)
print(f"\nXp shape: {Xp.shape}")

# Con predict_type='margin'
try:
    y_margin = bst.inplace_predict(Xp, predict_type='margin')
    print(f"booster.inplace_predict(..., predict_type='margin') = {y_margin[0]:.6f}")
except Exception as e:
    print(f"Error en inplace_predict: {e}")

# Con output_margin=True
try:
    y_margin2 = bundle.reg.predict(Xp, output_margin=True)
    print(f"reg.predict(..., output_margin=True) = {y_margin2[0]:.6f}")
except Exception as e:
    print(f"Error en predict with output_margin: {e}")

# Sin output_margin
try:
    y_full = bundle.reg.predict(Xp)
    print(f"reg.predict(Xp) = {y_full[0]:.6f}")
except Exception as e:
    print(f"Error: {e}")

print(f"\nbundle.log_target = {bundle.log_target}")
print(f"bundle.b0_offset = {bundle.b0_offset:.6f}")
