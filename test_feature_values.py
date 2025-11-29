#!/usr/bin/env python3
"""
Test: ¿Coinciden los VALORES entre build_base_input_row y bundle.pre.transform()?
"""
import pandas as pd
import numpy as np
from optimization.remodel.io import get_base_house
from optimization.remodel.xgb_predictor import XGBBundle
from optimization.remodel.gurobi_model import build_base_input_row

# Cargar casa + bundle
pid = 526351010
base_house = get_base_house(pid)
bundle = XGBBundle()

# Método 1: build_base_input_row (lo que usa el MIP)
X_mip = build_base_input_row(bundle, base_house.row)

# Método 2: bundle.pre.transform (lo que usa el predictor externo)
X_ext = bundle.pre.transform(X_mip)

print("=" * 70)
print("COMPARING FEATURE VALUES")
print("=" * 70)

# Convertir a numpy si es necesario
if hasattr(X_ext, 'toarray'):
    X_ext_dense = X_ext.toarray()
else:
    X_ext_dense = np.asarray(X_ext)

X_mip_dense = np.asarray(X_mip.values)

print(f"\nX_mip shape: {X_mip_dense.shape}")
print(f"X_ext shape: {X_ext_dense.shape}")

# Comparar
diff = np.abs(X_mip_dense - X_ext_dense)
max_diff = np.max(diff)
mean_diff = np.mean(diff)

print(f"\nMax absolute difference: {max_diff:.10f}")
print(f"Mean absolute difference: {mean_diff:.10f}")

# Ver dónde están las diferencias
if max_diff > 1e-8:
    mismatch_idx = np.where(diff > 1e-8)
    print(f"\nFeatures with differences > 1e-8:")
    for i, j in zip(mismatch_idx[0], mismatch_idx[1]):
        col = X_mip.columns[j]
        print(f"  [{i},{j}] {col}: {X_mip_dense[i,j]:.6f} vs {X_ext_dense[i,j]:.6f} (diff={diff[i,j]:.6f})")
else:
    print("\nPERFECT MATCH! X_mip and X_ext are identical.")

# Ahora: ¿da la MISMA predicción?
y_direct = bundle.predict(X_mip)
print(f"\n" + "=" * 70)
print(f"bundle.predict(X_mip from build_base_input_row): {float(y_direct.iloc[0]):.6f}")
print("=" * 70)
