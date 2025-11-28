#!/usr/bin/env python3
"""
Entrena un modelo lineal simple sobre SOLO variables remodelables para estimar
el uplift en dinero. La idea es sumar este Δ a la predicción base de la
regresión principal sin recalibrar el caso base.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

print("\n" + "="*70)
print("  ENTRENANDO MODELO UPLIFT (solo variables remodelables)")
print("="*70 + "\n")

# Cargar datos
df = pd.read_csv("data/raw/df_final_regresion.csv")
print(f"✓ Datos cargados: {df.shape}")

# Variables remodelables (consistentes con el MIP). Se filtrarán por disponibilidad en el CSV.
REMODEL_COLS = [
    "Gr Liv Area", "1st Flr SF", "2nd Flr SF",
    "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF",
    "Total Bsmt SF",
    "Full Bath", "Half Bath", "Bedroom AbvGr",
    "Garage Area", "Garage Cars",
    "Wood Deck SF", "Open Porch SF", "Enclosed Porch", "3Ssn Porch", "Screen Porch",
    "Pool Area"
]

# Filtrar columnas disponibles
cols = [c for c in REMODEL_COLS if c in df.columns]
print(f"✓ Usando {len(cols)} variables remodelables")

X = df[cols].copy().fillna(0)
y = df["SalePrice_Present"].copy()

# Pipeline simple: StandardScaler + LinearRegression
model = Pipeline([
    ("scaler", StandardScaler()),
    ("lin", LinearRegression())
])
model.fit(X, y)

# Métricas rápidas
r2 = model.score(X, y)
y_pred = model.predict(X)
rmse = np.sqrt(np.mean((y - y_pred) ** 2))
mape = np.mean(np.abs((y - y_pred) / y)) * 100

print("\n📊 MÉTRICAS (TRAINING uplift):")
print(f"  R²:   {r2:.4f}")
print(f"  RMSE: ${rmse:,.0f}")
print(f"  MAPE: {mape:.2f}%")

# Guardar paquete
pkg = {
    "model": model,
    "feature_names": cols,
    "r2": float(r2),
    "rmse": float(rmse),
    "mape": float(mape),
    "type": "uplift_linear",
}

out = Path("models/regression_uplift.pkl")
out.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(pkg, out)

print(f"\n💾 Guardado uplift en: {out}")
print(f"  - Modelo: StandardScaler + LinearRegression")
print(f"  - Features: {len(cols)}")
print("\n" + "="*70)
print("  ✅ ENTRENAMIENTO UPLIFT COMPLETADO")
print("="*70 + "\n")
