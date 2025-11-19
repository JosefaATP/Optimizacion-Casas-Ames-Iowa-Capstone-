#!/usr/bin/env python3
"""
Entrenar regresión lineal COMPATIBLE con XGBoost

Estrategia: Usar el MISMO bundle de XGBoost para extraer features,
entrenar regresión lineal con esos mismos features.

Así garantizamos que regresión y XGBoost reciben EXACTAMENTE los mismos datos.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

print("\n" + "="*70)
print("  ENTRENANDO REGRESIÓN COMPATIBLE CON XGBoost")
print("="*70 + "\n")

# ============================================================================
# CARGAR DATOS Y BUNDLE
# ============================================================================

print("🔄 Cargando datos...")
df = pd.read_csv("data/raw/df_final_regresion.csv")
print(f"✓ Datos: {df.shape}")

# Cargar bundle de XGBoost para ver qué features usa
print("\n🔄 Cargando bundle XGBoost...")
try:
    from optimization.remodel.xgb_predictor import XGBBundle
    from optimization.remodel.gurobi_model import build_base_input_row
    
    bundle = XGBBundle("models/xgb/construction_model_v13.bundle.pkl")
    xgb_features = list(bundle.feature_names_in())
    print(f"✓ XGBoost usa {len(xgb_features)} features")
    
except Exception as e:
    print(f"⚠️  No se pudo cargar XGBoost bundle: {e}")
    print("   Usando features por defecto...")
    xgb_features = None

# ============================================================================
# TRANSFORMAR TODOS LOS DATOS AL FORMATO DE XGBoost
# ============================================================================

if xgb_features:
    print(f"\n🔄 Transformando {len(df)} casas al formato de XGBoost...")
    
    X_transformed = []
    y_values = []
    
    for idx, row in df.iterrows():
        try:
            # Convertir a Series con los datos adecuados
            row_series = row.copy()
            
            # Usar build_base_input_row para transformar
            X_row = build_base_input_row(bundle, row_series)
            
            # Verificar que tenga las features correctas
            if list(X_row.columns) == xgb_features:
                X_transformed.append(X_row.values[0])
                y_values.append(np.log(row["SalePrice_Present"]))
        except Exception as e:
            if idx < 5:  # mostrar primeros errores
                print(f"  ⚠️  Error en fila {idx}: {e}")
    
    if X_transformed:
        X = pd.DataFrame(X_transformed, columns=xgb_features)
        y = np.array(y_values)
        
        print(f"✓ Transformadas {len(X)} casas (descartadas {len(df) - len(X)})")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
    else:
        print("❌ No se pudo transformar ninguna casa!")
        exit(1)
else:
    print("❌ No se pudo obtener features de XGBoost")
    exit(1)

# ============================================================================
# ENTRENAR REGRESIÓN LINEAL
# ============================================================================

from sklearn.linear_model import LinearRegression

print(f"\n🔄 Entrenando LinearRegression con {len(X)} muestras...")
model = LinearRegression()
model.fit(X, y)

print(f"✓ Modelo entrenado")

# ============================================================================
# EVALUACIÓN
# ============================================================================

y_pred = model.predict(X)
r2 = model.score(X, y)

# Métricas en escala real
y_real = np.exp(y)
y_pred_real = np.exp(y_pred)

rmse_real = np.sqrt(np.mean((y_real - y_pred_real) ** 2))
mape = np.mean(np.abs((y_real - y_pred_real) / y_real)) * 100

print(f"\n📊 MÉTRICAS:")
print(f"  R² (log): {r2:.4f}")
print(f"  RMSE (real): ${rmse_real:,.0f}")
print(f"  MAPE: {mape:.2f}%")

# ============================================================================
# VALIDACIÓN EN CASA ESPECÍFICA
# ============================================================================

print(f"\n🧪 VALIDACIÓN - PID 526301100:")

try:
    row = df[df["PID"] == 526301100].iloc[0]
    X_test = build_base_input_row(bundle, row)
    
    pred_log = model.predict(X_test)[0]
    pred_real = np.exp(pred_log)
    real_price = row["SalePrice_Present"]
    
    error = pred_real - real_price
    error_pct = (error / real_price) * 100
    
    print(f"  - Real: ${real_price:,.0f}")
    print(f"  - Predicción: ${pred_real:,.0f}")
    print(f"  - Error: ${error:+,.0f} ({error_pct:+.2f}%)")
    
    if abs(error_pct) < 15:
        print(f"  ✅ VÁLIDO - Error < 15%")
    elif abs(error_pct) < 25:
        print(f"  ⚠️  Moderado - Error 15-25%")
    else:
        print(f"  ❌ Alto - Error > 25%")
        
except Exception as e:
    print(f"  ❌ Error: {e}")

# ============================================================================
# SERIALIZAR
# ============================================================================

print(f"\n💾 Guardando modelo...")

model_pkg = {
    "model": model,
    "feature_names": xgb_features,
    "r2": float(r2),
    "rmse_real": float(rmse_real),
    "mape": float(mape),
    "type": "LinearRegression compatible XGBoost",
}

output_path = Path("models/regression_model_xgb_compatible.pkl")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "wb") as f:
    pickle.dump(model_pkg, f)

print(f"✓ Guardado en: {output_path}")
print(f"  - Tipo: LinearRegression")
print(f"  - Features: {len(xgb_features)}")
print(f"  - Compatible con: {len(X)} casas de training")

print("\n" + "="*70)
print("  ✅ LISTO")
print("="*70 + "\n")
