#!/usr/bin/env python3
"""
Reentrena la regresión usando EXACTAMENTE las features que produce build_base_input_row
pero con un pipeline StandardScaler + LinearRegression para que los coeficientes
sean sensibles a m²/baños y no dependan de la escala cruda.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path as PathlibPath
import os
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Asegurar cwd en raíz del repo
project_dir = PathlibPath(__file__).parent.parent
os.chdir(project_dir)
sys.path.insert(0, '.')

print("\n" + "="*70)
print("  REENTRENAMIENTO: Regresión (Scaler + LinearRegression) con features del MIP")
print("="*70 + "\n")

# Cargar datos
df_raw = pd.read_csv("data/raw/df_final_regresion.csv")
print(f"✓ {len(df_raw)} casas cargadas")

# Importar funciones de procesamiento de run_opt
from optimization.remodel.xgb_predictor import XGBBundle
from optimization.remodel.gurobi_model import build_base_input_row

bundle = XGBBundle()
print(f"✓ Bundle XGBoost cargado con {len(bundle.feature_names_in())} features")

# Procesar data al formato del MIP
X_list, y_list, errors = [], [], []
for idx, row in df_raw.iterrows():
    try:
        X_proc = build_base_input_row(bundle, row)
        X_list.append(X_proc.values[0])
        y_list.append(np.log(row["SalePrice_Present"]))
        if (idx + 1) % 500 == 0:
            print(f"  Procesadas {idx+1}/{len(df_raw)}...")
    except Exception as e:
        errors.append((idx, str(e)))
        if len(errors) <= 5:
            print(f"  ⚠️  Error en fila {idx}: {e}")

if not X_list:
    print("❌ No se pudo procesar ninguna casa")
    sys.exit(1)

X = pd.DataFrame(X_list, columns=bundle.feature_names_in())
y = np.array(y_list)

print(f"✓ {len(X)} casas procesadas correctamente")
print(f"  Errores: {len(errors)}")
print(f"  X shape: {X.shape}")

# Entrenar pipeline Scaler + LinearRegression
print("\n🔄 Entrenando StandardScaler + LinearRegression...")
lin_reg = LinearRegression()
model = Pipeline([
    ("scaler", StandardScaler()),
    ("lin", lin_reg),
])
model.fit(X, y)
print("✓ Modelo entrenado")

# Evaluar
y_pred = model.predict(X)
r2 = model.score(X, y)
y_real = np.exp(y)
y_pred_real = np.exp(y_pred)
rmse_real = np.sqrt(np.mean((y_real - y_pred_real) ** 2))
mae_real = np.mean(np.abs(y_real - y_pred_real))
mape = np.mean(np.abs((y_real - y_pred_real) / y_real)) * 100

print("\n📊 MÉTRICAS (TRAIN):")
print(f"  R² (log): {r2:.4f}")
print(f"  RMSE (real): ${rmse_real:,.0f}")
print(f"  MAE (real): ${mae_real:,.0f}")
print(f"  MAPE: {mape:.2f}%")

# Validación en PID 526301100
try:
    row = df_raw[df_raw["PID"] == 526301100].iloc[0]
    real_price = row["SalePrice_Present"]
    X_val = build_base_input_row(bundle, row)
    pred_log = model.predict(X_val)[0]
    pred_real = np.exp(pred_log)
    err_pct = (pred_real - real_price) / real_price * 100
    print("\n🧪 VALIDACIÓN PID 526301100:")
    print(f"  - Real: ${real_price:,.0f}")
    print(f"  - Pred: ${pred_real:,.0f} ({err_pct:+.2f}%)")
except Exception as e:
    print(f"\n⚠️  Validación falló: {e}")

# Guardar modelo
model_pkg = {
    "model": model,
    "feature_names": list(X.columns),
    "r2": float(r2),
    "rmse_real": float(rmse_real),
    "mae_real": float(mae_real),
    "mape": float(mape),
    "processed_with": "build_base_input_row",
    "n_training_samples": len(X),
    "estimator": "LinearRegression",
    "scaler": "StandardScaler",
}
out = Path("models/regression_model_reprocesed.pkl")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "wb") as f:
    pickle.dump(model_pkg, f)

print("\n💾 Guardado en:", out)
print(f"  - Modelo: StandardScaler + LinearRegression")
print(f"  - Features: {len(X.columns)}")
print(f"  - Compatible con run_opt/build_base_input_row")
print("\n" + "="*70)
print("  ✅ REENTRENAMIENTO COMPLETADO")
print("="*70 + "\n")
