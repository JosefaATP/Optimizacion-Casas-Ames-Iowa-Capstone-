#!/usr/bin/env python3
"""
Entrenar regresión lineal CORRECTA replicando SOLO_REGRESION.ipynb

Arquitectura:
- Usa get_dummies() para ONE-HOT ENCODING (categóricas)
- Target: SalePrice_Present (sin transformación log)
- Serializa: modelo + información de features para predicciones futuras
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path
import joblib

# ============================================================================
# CARGAR DATOS
# ============================================================================
print("\n" + "="*70)
print("  ENTRENANDO REGRESIÓN LINEAL (CORRECTA CON ONE-HOT ENCODING)")
print("="*70)

RAW_FILE = Path("data/raw/df_final_regresion.csv")
MODEL_OUTPUT = Path("models/regression_model_correct.joblib")

if not RAW_FILE.exists():
    print(f"❌ ERROR: No encontré {RAW_FILE}")
    exit(1)

df = pd.read_csv(RAW_FILE)
print(f"\n📊 Datos cargados: {df.shape[0]} casas × {df.shape[1]} columnas")

# ============================================================================
# LIMPIAR DATOS (como SOLO_REGRESION.ipynb)
# ============================================================================

# Remover PID y SalePrice (versión anterior)
df_clean = df.copy()
df_clean.drop("PID", axis=1, inplace=True, errors="ignore")
df_clean.drop("SalePrice", axis=1, inplace=True, errors="ignore")

print(f"Después de limpiar: {df_clean.shape[0]} × {df_clean.shape[1]}")

# ============================================================================
# ONE-HOT ENCODING (get_dummies)
# ============================================================================

# Identificar columnas categóricas
cat_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()
print(f"\n📝 Columnas categóricas encontradas ({len(cat_cols)}):")
for col in cat_cols[:10]:  # mostrar primeras 10
    print(f"   - {col}")
if len(cat_cols) > 10:
    print(f"   ... y {len(cat_cols) - 10} más")

# Aplicar get_dummies con drop_first=True (como en el notebook)
df_dummies = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)

print(f"\nDespués de get_dummies: {df_dummies.shape[0]} × {df_dummies.shape[1]} features")

# ============================================================================
# PREPARAR X E y
# ============================================================================

# Target: LOG(SalePrice_Present) - como en SOLO_REGRESION.ipynb
if "SalePrice_Present" not in df_dummies.columns:
    print("❌ ERROR: No encontré columna 'SalePrice_Present'")
    print(f"   Columnas disponibles: {df_dummies.columns.tolist()}")
    exit(1)

y_raw = df_dummies["SalePrice_Present"].copy()
y = np.log(y_raw)  # TRANSFORMACIÓN LOG
X = df_dummies.drop(columns=["SalePrice_Present"], errors="ignore")

print(f"\n✓ Target (y): LOG(SalePrice_Present)")
print(f"  - Shape: {y.shape}")
print(f"  - Min (log): {y.min():.3f}")
print(f"  - Max (log): {y.max():.3f}")
print(f"  - Media (log): {y.mean():.3f}")
print(f"  - Min (precio real): ${np.exp(y.min()):,.0f}")
print(f"  - Max (precio real): ${np.exp(y.max()):,.0f}")
print(f"  - Media (precio real): ${np.exp(y.mean()):,.0f}")
print(f"\n✓ Features (X): {X.shape[1]} características")
print(f"  - Numéricas: {X.select_dtypes(include=[np.number]).shape[1]}")

# ============================================================================
# VALIDACIÓN DE TIPOS
# ============================================================================

# Asegurar que todo sea numérico
for col in X.columns:
    if X[col].dtype not in [np.float64, np.int64, np.float32, np.int32]:
        print(f"⚠️  Convirtiendo {col} de {X[col].dtype} a float64")
        X[col] = pd.to_numeric(X[col], errors="coerce")

# Llenar NaNs con 0 (por si acaso)
X = X.fillna(0)
y = y.fillna(y.mean())

print(f"\nValidación: X contiene {X.isna().sum().sum()} NaNs después de llenar")
print(f"Validación: y contiene {y.isna().sum()} NaNs después de llenar")

# ============================================================================
# ENTRENAR MODELO
# ============================================================================

print(f"\n🔄 Entrenando modelo LinearRegression...")
model = LinearRegression()
model.fit(X.values, y.values)

print(f"✓ Modelo entrenado")

# ============================================================================
# EVALUAR MODELO
# ============================================================================

y_pred = model.predict(X.values)

# Métricas
r2 = model.score(X.values, y.values)
rmse = np.sqrt(np.mean((y - y_pred) ** 2))
mae = np.mean(np.abs(y - y_pred))

print(f"\n📊 MÉTRICAS DEL MODELO:")
print(f"  R²: {r2:.4f}")
print(f"  RMSE: ${rmse:,.0f}")
print(f"  MAE: ${mae:,.0f}")

# ============================================================================
# VALIDACIÓN: Predicción en data de entrenamiento
# ============================================================================

print(f"\n🧪 VALIDACIÓN EN DATOS DE ENTRENAMIENTO:")

# Encontrar la fila de training que corresponde al PID 526301100
pid_col = "PID"
if pid_col in df.columns:
    test_idx = df[df[pid_col] == 526301100].index
    if len(test_idx) > 0:
        idx = test_idx[0]
        real_price = df.loc[idx, "SalePrice_Present"]
        
        # Construir X para esa casa (con one-hot encoding)
        df_test = df.iloc[[idx]].copy()
        df_test.drop("PID", axis=1, inplace=True, errors="ignore")
        df_test.drop("SalePrice", axis=1, inplace=True, errors="ignore")
        df_test_dummies = pd.get_dummies(df_test, columns=cat_cols, drop_first=True)
        
        # Alinear columnas con X del training
        for col in X.columns:
            if col not in df_test_dummies.columns:
                df_test_dummies[col] = 0
        df_test_dummies = df_test_dummies[X.columns]
        
        # Predecir en escala LOG y convertir a precio real
        pred_log = model.predict(df_test_dummies.values)[0]
        pred_price = np.exp(pred_log)
        error_pct = (pred_price - real_price) / real_price * 100
        
        print(f"  PID 526301100:")
        print(f"    - Precio real: ${real_price:,.0f}")
        print(f"    - Predicción (log): {pred_log:.4f}")
        print(f"    - Predicción (actual): ${pred_price:,.0f}")
        print(f"    - Error: {error_pct:+.2f}%")

# ============================================================================
# SERIALIZAR MODELO
# ============================================================================

print(f"\n💾 Serializando modelo...")

# Guardar: modelo + información de features
model_data = {
    "model": model,
    "feature_names": list(X.columns),
    "cat_cols": cat_cols,
    "r2": r2,
    "rmse": rmse,
    "mae": mae,
    "intercept": model.intercept_,
    "coef_shape": model.coef_.shape,
}

MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model_data, MODEL_OUTPUT)

print(f"✓ Modelo guardado en: {MODEL_OUTPUT}")
print(f"  - Size: {MODEL_OUTPUT.stat().st_size / 1024:.1f} KB")

print("\n" + "="*70)
print("  ✅ ENTRENAMIENTO COMPLETADO")
print("="*70 + "\n")
