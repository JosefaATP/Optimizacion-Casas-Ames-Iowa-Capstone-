#!/usr/bin/env python3
"""
RE-PROCESAR df_final_regresion.csv para que calcen exactamente con los datos
que genera run_opt.py cuando crea la casa remodelada.

La idea: 
1. Tomar cada casa en df_final_regresion.csv
2. Procesarla EXACTAMENTE como run_opt.py lo hace (usando build_base_input_row)
3. Esto genera features en el formato correcto
4. Entrenar regresión con ESTOS datos

Resultado: Regresi¢n perfectamente compatible con run_opt.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, '.')

# Cambiar al directorio del proyecto
from pathlib import Path as PathlibPath
project_dir = PathlibPath(__file__).parent.parent
import os
os.chdir(project_dir)

import pandas as pd
import numpy as np
from pathlib import Path
import pickle

print("\n" + "="*70)
print("  REENTRENAMIENTO: Regresión con datos RE-PROCESADOS")
print("="*70 + "\n")

# ============================================================================
# CARGAR DATOS ORIGINALES
# ============================================================================

print("🔄 Cargando datos...")
df_raw = pd.read_csv("data/raw/df_final_regresion.csv")
print(f"✓ {len(df_raw)} casas cargadas")

# ============================================================================
# IMPORTAR FUNCIONES DE PROCESAMIENTO DE run_opt.py
# ============================================================================

print("\n🔄 Importando funciones de procesamiento...")
try:
    from optimization.remodel.xgb_predictor import XGBBundle
    from optimization.remodel.gurobi_model import build_base_input_row
    
    # Usar la ruta del modelo que usa run_opt.py
    bundle = XGBBundle()
    print(f"✓ Bundle XGBoost cargado")
    print(f"  Features esperadas: {len(bundle.feature_names_in())}")
    
except Exception as e:
    print(f"❌ Error importando: {e}")
    print("\nAlternativa: Procesar manualmente con one-hot encoding")
    bundle = None

# ============================================================================
# PROCESAR DATOS USANDO build_base_input_row
# ============================================================================

if bundle:
    print(f"\n🔄 Procesando {len(df_raw)} casas con build_base_input_row...")
    
    X_list = []
    y_list = []
    errors = []
    
    for idx, row in df_raw.iterrows():
        try:
            # EXACTAMENTE como lo hace run_opt.py
            X_processed = build_base_input_row(bundle, row)
            
            # Extraer precio
            y_val = np.log(row["SalePrice_Present"])
            
            X_list.append(X_processed.values[0])
            y_list.append(y_val)
            
            if (idx + 1) % 500 == 0:
                print(f"  Procesadas {idx + 1}/{len(df_raw)} casas...")
                
        except Exception as e:
            errors.append((idx, str(e)))
            if len(errors) <= 5:
                print(f"  ⚠️  Error en fila {idx}: {str(e)[:50]}")
    
    if X_list:
        X = pd.DataFrame(X_list, columns=bundle.feature_names_in())
        y = np.array(y_list)
        
        print(f"\n✓ {len(X)} casas procesadas correctamente")
        print(f"  Errores: {len(errors)}")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
    else:
        print("❌ No se pudo procesar ninguna casa")
        sys.exit(1)
else:
    print("❌ No se puede procesar sin bundle")
    sys.exit(1)

# Opcional: eliminar columnas altamente colineales que bloquean peso en componentes (ej. Total Bsmt SF)
drop_cols = [c for c in ["Total Bsmt SF"] if c in X.columns]
if drop_cols:
    print(f"\n⚙️  Eliminando columnas colineales para dar peso a componentes: {drop_cols}")
    X = X.drop(columns=drop_cols)

# ============================================================================
# ENTRENAR REGRESIÓN CON DATOS RE-PROCESADOS (ESCALADO + RIDGE)
# ============================================================================
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

alpha = 1e-6  # regularización casi nula para no matar metros/baños
print(f"\n🔄 Entrenando Ridge con StandardScaler (alpha={alpha})...")
ridge_reg = Ridge(alpha=alpha, fit_intercept=True)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", ridge_reg),
])
model.fit(X, y)

best_alpha = alpha
print(f"✓ Modelo entrenado (Ridge + scaler) | alpha = {best_alpha}")

# ============================================================================
# EVALUAR
# ============================================================================

y_pred = model.predict(X)
r2 = model.score(X, y)

# En escala real
y_real = np.exp(y)
y_pred_real = np.exp(y_pred)

rmse_real = np.sqrt(np.mean((y_real - y_pred_real) ** 2))
mae_real = np.mean(np.abs(y_real - y_pred_real))
mape = np.mean(np.abs((y_real - y_pred_real) / y_real)) * 100

print(f"\n📊 MÉTRICAS (TRAINING):")
print(f"  R² (log): {r2:.4f}")
print(f"  RMSE (real): ${rmse_real:,.0f}")
print(f"  MAE (real): ${mae_real:,.0f}")
print(f"  MAPE: {mape:.2f}%")
# Mostrar pesos aproximados (en espacio log) para features clave
try:
    ridge = model.named_steps["ridge"]
    scaler = model.named_steps["scaler"]
    coef = ridge.coef_.ravel()
    scale = scaler.scale_
    feat_series = pd.Series(coef / scale, index=X.columns)  # efecto por unidad en log-space
    keys = ["Gr Liv Area", "1st Flr SF", "BsmtFin SF 1", "Bsmt Unf SF", "Full Bath", "Half Bath", "Bedroom AbvGr"]
    print("\n🔎 Pesos (log por unidad) en features remodelables:")
    for k in keys:
        if k in feat_series.index:
            print(f"  - {k}: {feat_series[k]:+.6f}")
except Exception:
    pass
# ============================================================================
# PRUEBA CON CASA ESPECÍFICA (PID 526301100)
# ============================================================================

print(f"\n🧪 VALIDACIÓN - PID 526301100:")

try:
    test_row = df_raw[df_raw["PID"] == 526301100].iloc[0]
    real_price = test_row["SalePrice_Present"]
    
    # Procesar IGUAL que entrenamiento
    X_test = build_base_input_row(bundle, test_row)
    # Alinear columnas con X (por si eliminaste algunas)
    for col in X.columns:
        if col not in X_test.columns:
            X_test[col] = 0.0
    X_test = X_test[X.columns]
    
    # Predecir
    pred_log = model.predict(X_test)[0]
    pred_real = np.exp(pred_log)
    
    error = pred_real - real_price
    error_pct = (error / real_price) * 100
    
    print(f"  - Real: ${real_price:,.0f}")
    print(f"  - Predicción: ${pred_real:,.0f}")
    print(f"  - Error: ${error:+,.0f} ({error_pct:+.2f}%)")
    
    if abs(error_pct) < 10:
        status = "✅ EXCELENTE (< 10%)"
    elif abs(error_pct) < 15:
        status = "✅ BUENO (10-15%)"
    elif abs(error_pct) < 25:
        status = "⚠️  Aceptable (15-25%)"
    else:
        status = "❌ Inaceptable (> 25%)"
    
    print(f"  {status}")
    
except Exception as e:
    print(f"  ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# SERIALIZAR MODELO
# ============================================================================

print(f"\n💾 Guardando modelo...")

model_pkg = {
    "model": model,
    "feature_names": list(X.columns),
    "r2": float(r2),
    "rmse_real": float(rmse_real),
    "mae_real": float(mae_real),
    "mape": float(mape),
    "processed_with": "build_base_input_row from run_opt.py",
    "n_training_samples": len(X),
    "estimator": "Ridge",
    "alpha": best_alpha,
}

output_path = Path("models/regression_model_reprocesed.pkl")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "wb") as f:
    pickle.dump(model_pkg, f)

print(f"✓ Guardado en: {output_path}")
print(f"  - Modelo: Ridge + StandardScaler")
print(f"  - Features: {len(X.columns)}")
print(f"  - Procesados con: build_base_input_row")
print(f"  - Compatible con: run_opt.py al 100%")

print("\n" + "="*70)
print("  ✅ REENTRENAMIENTO COMPLETADO")
print("="*70 + "\n")
