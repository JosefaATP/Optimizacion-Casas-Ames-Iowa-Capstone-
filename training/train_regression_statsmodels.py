#!/usr/bin/env python3
"""
Entrenar regresión lineal CORRECTA usando statsmodels OLS
Replica exactamente lo que hace SOLO_REGRESION.ipynb

Arquitectura:
- get_dummies() para one-hot encoding
- Target: log(SalePrice_Present)
- Modelo: OLS (statsmodels), NO sklearn
- Serializa modelo para predicciones futuras
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import pickle

# ============================================================================
# CARGAR DATOS
# ============================================================================
print("\n" + "="*70)
print("  ENTRENANDO REGRESIÓN CON STATSMODELS (OLS)")
print("="*70)

RAW_FILE = Path("data/raw/df_final_regresion.csv")
MODEL_OUTPUT = Path("models/regression_model_statsmodels.pkl")

if not RAW_FILE.exists():
    print(f"❌ ERROR: No encontré {RAW_FILE}")
    exit(1)

df = pd.read_csv(RAW_FILE)
print(f"\n📊 Datos cargados: {df.shape[0]} casas × {df.shape[1]} columnas")

# ============================================================================
# PREPARAR DATOS (replicar SOLO_REGRESION.ipynb)
# ============================================================================

df_clean = df.copy()
df_clean.drop("PID", axis=1, inplace=True, errors="ignore")
df_clean.drop("SalePrice", axis=1, inplace=True, errors="ignore")

# Get dummies para categóricas
cat_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()
print(f"📝 Categorías encontradas: {len(cat_cols)}")

# ⚠️  NO usar drop_first=True para que todas las categorías sean representadas
df_dummies = pd.get_dummies(df_clean, columns=cat_cols, drop_first=False)
print(f"Después de get_dummies (sin drop_first): {df_dummies.shape}")

# ============================================================================
# TARGET EN LOG
# ============================================================================

y_raw = df_dummies["SalePrice_Present"]
y = np.log(y_raw)  # LOG transformation

X = df_dummies.drop(columns=["SalePrice_Present"], errors="ignore")

# Limpiar inf/-inf
X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")
y = y.replace([np.inf, -np.inf], np.nan)

# Asegurar float
X = X.astype(float)
y = y.astype(float)

print(f"\n✓ Target (y): log(SalePrice_Present)")
print(f"  - Shape: {y.shape}")
print(f"  - Min: {y.min():.3f}, Max: {y.max():.3f}, Media: {y.mean():.3f}")
print(f"✓ Features (X): {X.shape}")

# ============================================================================
# AGREGAR CONSTANTE
# ============================================================================

X = sm.add_constant(X, has_constant="add")
print(f"\n✓ Constante añadida: {X.shape}")

# ============================================================================
# ENTRENAR MODELO OLS
# ============================================================================

print(f"\n🔄 Entrenando OLS...")
model = sm.OLS(y.values, X.values).fit()

print(f"✓ Modelo entrenado")

# ============================================================================
# MOSTRAR RESUMEN
# ============================================================================

print(f"\n📊 RESUMEN DEL MODELO:")
print(f"  R²: {model.rsquared:.4f}")
print(f"  Adj R²: {model.rsquared_adj:.4f}")
print(f"  AIC: {model.aic:.2f}")
print(f"  BIC: {model.bic:.2f}")

# RMSE (en escala log)
residuals = model.resid
rmse_log = np.sqrt(np.mean(residuals**2))
print(f"  RMSE (log): {rmse_log:.5f}")

# RMSE en escala real
y_pred_log = model.fittedvalues
y_pred = np.exp(y_pred_log)
y_real = np.exp(y)
rmse_real = np.sqrt(np.mean((y_real - y_pred) ** 2))
print(f"  RMSE (real): ${rmse_real:,.0f}")

# MAPE
mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100
print(f"  MAPE: {mape:.2f}%")

# ============================================================================
# VALIDACIÓN EN DATOS DE TRAINING
# ============================================================================

print(f"\n🧪 VALIDACIÓN EN DATOS DE TRAINING:")

# Encontrar PID 526301100
df_original = pd.read_csv("data/raw/df_final_regresion.csv")
if 526301100 in df_original["PID"].values:
    idx = df_original[df_original["PID"] == 526301100].index[0]
    real_price = df_original.loc[idx, "SalePrice_Present"]
    
    # Reconstruir X para esa casa
    row_test = df_original.iloc[[idx]].copy()
    row_test.drop(["PID", "SalePrice"], axis=1, inplace=True, errors="ignore")
    row_test_dummies = pd.get_dummies(row_test, columns=cat_cols, drop_first=True)
    
    # Alinear features
    for col in X.columns:
        if col == "const":
            continue
        if col not in row_test_dummies.columns:
            row_test_dummies[col] = 0
    
    # Ordenar columns igual que en training (excepto const)
    cols_order = [c for c in X.columns if c != "const"]
    row_test_dummies = row_test_dummies[cols_order]
    
    # Agregar constante
    row_test_dummies.insert(0, "const", 1.0)
    
    # Predecir
    try:
        pred_log = model.predict(row_test_dummies.values)[0]
        pred_price = np.exp(pred_log)
        error = pred_price - real_price
        error_pct = (error / real_price) * 100
        
        print(f"  PID 526301100:")
        print(f"    - Real: ${real_price:,.0f}")
        print(f"    - Predicción: ${pred_price:,.0f}")
        print(f"    - Error: ${error:+,.0f} ({error_pct:+.2f}%)")
        
        if abs(error_pct) < 15:
            print(f"    ✅ Error razonable")
        else:
            print(f"    ⚠️  Error alto")
    except Exception as e:
        print(f"    ❌ Error prediciendo: {e}")

# ============================================================================
# SERIALIZAR MODELO
# ============================================================================

print(f"\n💾 Serializando modelo...")

# Guardar: modelo OLS + información necesaria
model_package = {
    "model": model,
    "feature_names": list(X.columns),
    "cat_cols": cat_cols,
    "y_mean": float(y.mean()),
    "y_std": float(y.std()),
    "r2": float(model.rsquared),
    "rmse_log": float(rmse_log),
    "rmse_real": float(rmse_real),
    "mape": float(mape),
}

with open(MODEL_OUTPUT, "wb") as f:
    pickle.dump(model_package, f)

print(f"✓ Modelo guardado en: {MODEL_OUTPUT}")
print(f"  - Tipo: OLS (statsmodels)")
print(f"  - Features: {len(X.columns)}")
print(f"  - R²: {model.rsquared:.4f}")

print("\n" + "="*70)
print("  ✅ ENTRENAMIENTO COMPLETADO")
print("="*70 + "\n")
