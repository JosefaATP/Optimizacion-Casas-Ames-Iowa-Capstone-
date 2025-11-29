#!/usr/bin/env python3
"""
Test: comparar predict_log_raw() externo vs MIP y_log
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

# 1. Predicción EXTERNA
y_ext = bundle.predict(X_input).iloc[0]
print(f"External predict(X_input): {y_ext:.6f}")

# 2. Predicción LOG RAW EXTERNA (sin aplicar transformada inversa)
y_log_raw_ext = bundle.predict_log_raw(X_input).iloc[0]
print(f"External predict_log_raw(X_input): {y_log_raw_ext:.6f}")

# Verificar: exp(y_log_raw_ext) - 1 debería = y_ext
y_back = np.expm1(y_log_raw_ext)
print(f"  expm1({y_log_raw_ext:.6f}) = {y_back:.6f} (should match above: {np.isclose(y_back, y_ext)})")

# 3. Construir MIP y ver y_log del MIP
m = build_mip_embed(base_house.row, budget, ct, bundle, base_price=y_ext, fix_to_base=False)
m.optimize()

# Extraer y_log del MIP
y_log_mip = m._y_log_var if hasattr(m, '_y_log_var') else None
if y_log_mip is not None:
    if hasattr(y_log_mip, 'X'):
        y_log_mip_val = float(y_log_mip.X)
    else:
        y_log_mip_val = float(y_log_mip)
    print(f"\nMIP y_log: {y_log_mip_val:.6f}")
    print(f"External y_log_raw: {y_log_raw_ext:.6f}")
    print(f"Delta: {y_log_mip_val - y_log_raw_ext:.6f}")
else:
    print("\nERROR: No se encontró y_log en el MIP")

# Mostrar b0_offset
print(f"\nbundle.b0_offset = {bundle.b0_offset:.6f}")
