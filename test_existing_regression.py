#!/usr/bin/env python3
"""
Probar el modelo de regresión existente: base_reg.joblib
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

print("\n" + "="*70)
print("  TESTEO: Modelo de Regresión Existente (base_reg.joblib)")
print("="*70 + "\n")

# ============================================================================
# CARGAR MODELO Y DATOS
# ============================================================================

model = joblib.load("models/reg/base_reg.joblib")
print("✓ Modelo cargado: Pipeline con preprocessing")

# Ver estructura del pipeline
print(f"\n  Pasos del pipeline:")
for i, (name, step) in enumerate(model.named_steps.items()):
    print(f"    {i+1}. {name}: {type(step).__name__}")

# Cargar datos
df = pd.read_csv("data/raw/df_final_regresion.csv")
print(f"\n✓ Datos cargados: {df.shape}")

# ============================================================================
# ENCONTRAR LA CASA DE PRUEBA
# ============================================================================

pid = 526301100
if pid not in df["PID"].values:
    print(f"❌ PID {pid} no encontrado")
    exit(1)

idx = df[df["PID"] == pid].index[0]
row = df.iloc[idx]

real_price = row["SalePrice_Present"]
print(f"\n🏠 Casa PID {pid}:")
print(f"  - Precio real: ${real_price:,.0f}")

# ============================================================================
# PREPARAR DATOS
# ============================================================================

# El modelo espera las mismas columnas del training
# Examinar qué columnas espera
if hasattr(model, 'feature_names_in_'):
    expected_features = model.feature_names_in_
    print(f"\n  Features esperadas por el modelo: {len(expected_features)}")
    print(f"    - Primeros 10: {expected_features[:10]}")
else:
    print("  ⚠️  No se pueden determinar las features esperadas")
    expected_features = None

# Preparar la fila como DataFrame
df_test = df.iloc[[idx]].copy()

# Eliminar PID y SalePrice
df_test.drop(["PID", "SalePrice"], axis=1, errors="ignore", inplace=True)

# Si el modelo espera features específicas, seleccionar solo esas
if expected_features is not None:
    for col in expected_features:
        if col not in df_test.columns:
            df_test[col] = 0
    df_test = df_test[expected_features]

print(f"\n  DataFrame para predicción: {df_test.shape}")

# ============================================================================
# PREDECIR
# ============================================================================

try:
    pred = model.predict(df_test)
    pred_price = float(pred[0])
    
    error_pct = (pred_price - real_price) / real_price * 100
    
    print(f"\n💰 PREDICCIÓN:")
    print(f"  - Modelo predice: ${pred_price:,.0f}")
    print(f"  - Precio real:    ${real_price:,.0f}")
    print(f"  - Error:          {error_pct:+.2f}%")
    
    if abs(error_pct) < 10:
        print(f"\n  ✅ Error razonable (<10%)")
    elif abs(error_pct) < 20:
        print(f"\n  ⚠️  Error moderado (10-20%)")
    else:
        print(f"\n  ❌ Error inaceptable (>20%)")
        
except Exception as e:
    print(f"❌ Error al predecir: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70 + "\n")
