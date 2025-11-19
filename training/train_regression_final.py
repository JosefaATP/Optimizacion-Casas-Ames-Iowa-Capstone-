#!/usr/bin/env python3
"""
Entrenar regresión lineal con EXACTAMENTE el mismo enfoque que usa run_opt.py:
- One-hot encoding para categóricas
- Features numéricas como están
- Target: log(SalePrice_Present)

La clave: Asegurar que los features coincidan exactamente con los que genera
build_base_input_row() en run_opt.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.linear_model import LinearRegression

print("\n" + "="*70)
print("  ENTRENAMIENTO REGRESIÓN LINEAL (FORMATO run_opt.py)")
print("="*70 + "\n")

# ============================================================================
# CARGAR DATOS
# ============================================================================

df = pd.read_csv("data/raw/df_final_regresion.csv")
print(f"✓ Datos cargados: {df.shape}")

# ============================================================================
# PREPARAR FEATURES EXACTAMENTE COMO LO HACE build_base_input_row()
# ============================================================================

# 1. Identificar columnas categóricas (tienen "_" después de procesarse)
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
print(f"\n📝 Categorías identificadas: {len(cat_cols)}")
print(f"   {cat_cols}")

# 2. Crear one-hot encoding (usando drop_first=True para evitar colinealidad)
df_dummies = pd.get_dummies(df.drop(["PID", "SalePrice"], axis=1, errors="ignore"),
                             columns=cat_cols,
                             drop_first=True)

print(f"\n✓ One-hot encoding aplicado: {df_dummies.shape[1]} features")

# 3. Separar X e y
y = np.log(df_dummies["SalePrice_Present"])
X = df_dummies.drop("SalePrice_Present", axis=1)

print(f"✓ X shape: {X.shape}")
print(f"✓ y shape: {y.shape}")

# ============================================================================
# ENTRENAR MODELO
# ============================================================================

print(f"\n🔄 Entrenando LinearRegression...")
model = LinearRegression()
model.fit(X, y)

print(f"✓ Entrenado")

# ============================================================================
# EVALUACIÓN
# ============================================================================

y_pred_log = model.predict(X)
r2_log = model.score(X, y)

# En escala real
y_real = np.exp(y)
y_pred_real = np.exp(y_pred_log)

rmse_real = np.sqrt(np.mean((y_real - y_pred_real) ** 2))
mae_real = np.mean(np.abs(y_real - y_pred_real))
mape = np.mean(np.abs((y_real - y_pred_real) / y_real)) * 100

print(f"\n📊 MÉTRICAS (en TRAINING):")
print(f"  R² (log): {r2_log:.4f}")
print(f"  RMSE (real): ${rmse_real:,.0f}")
print(f"  MAE (real): ${mae_real:,.0f}")
print(f"  MAPE: {mape:.2f}%")

# ============================================================================
# PRUEBA CON CASA ESPECÍFICA
# ============================================================================

print(f"\n🧪 VALIDACIÓN - PID 526301100:")

# Obtener la casa original
df_orig = pd.read_csv("data/raw/df_final_regresion.csv")
row_orig = df_orig[df_orig["PID"] == 526301100].iloc[0]
real_price = row_orig["SalePrice_Present"]

# Transformar al formato de X
row_clean = row_orig.drop(["PID", "SalePrice"], errors="ignore").copy()
row_dummies = pd.get_dummies(pd.DataFrame([row_clean]), columns=cat_cols, drop_first=True)

# Alinear features
for col in X.columns:
    if col not in row_dummies.columns:
        row_dummies[col] = 0
        
row_dummies = row_dummies[X.columns]

# Predecir
pred_log = model.predict(row_dummies)[0]
pred_real = np.exp(pred_log)

error = pred_real - real_price
error_pct = (error / real_price) * 100

print(f"  - Real: ${real_price:,.0f}")
print(f"  - Predicción: ${pred_real:,.0f}")
print(f"  - Error: ${error:+,.0f} ({error_pct:+.2f}%)")

if abs(error_pct) < 15:
    status = "✅ VÁLIDO (< 15%)"
elif abs(error_pct) < 25:
    status = "⚠️  Moderado (15-25%)"
else:
    status = "❌ Alto (> 25%)"
    
print(f"  {status}")

# ============================================================================
# SERIALIZAR
# ============================================================================

print(f"\n💾 Guardando modelo...")

model_pkg = {
    "model": model,
    "feature_names": list(X.columns),
    "cat_cols": cat_cols,
    "r2": float(r2_log),
    "rmse_real": float(rmse_real),
    "mae_real": float(mae_real),
    "mape": float(mape),
}

output_path = Path("models/regression_model_final.pkl")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "wb") as f:
    pickle.dump(model_pkg, f)

print(f"✓ Guardado en: {output_path}")
print(f"  - Modelo: LinearRegression")
print(f"  - Features: {len(X.columns)}")
print(f"  - Compatible con: run_opt.py")

print("\n" + "="*70)
print("  ✅ ENTRENAMIENTO COMPLETADO")
print("="*70 + "\n")
