"""
Script de diagnóstico: ver qué features genera run_opt vs qué espera regresión
"""
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Setup
pid = 526301100
print("="*80)
print(f"DIAGNÓSTICO: PID {pid}")
print("="*80)

# 1. Ver datos originales
print("\n📂 1. DATOS ORIGINALES")
df = pd.read_csv("data/raw/df_final_regresion.csv")
row_original = df[df["PID"] == pid].iloc[0]

print(f"   SalePrice_Present (valor real): ${row_original['SalePrice_Present']:,.0f}")
print(f"   SalePrice_Present (log): {np.log(row_original['SalePrice_Present']):.6f}")

# 2. Cargar modelo de regresión
print("\n🤖 2. MODELO DE REGRESIÓN")
reg_model = joblib.load("models/regression_model.joblib")

# Manejo de Pipeline vs Modelo simple
if hasattr(reg_model, 'named_steps'):
    # Es un Pipeline
    regressor = reg_model.named_steps['regressor']
    print(f"   Tipo: Pipeline (StandardScaler + LinearRegression)")
    print(f"   Intercept: {regressor.intercept_:.6f}")
    reg_features = list(regressor.feature_names_in_) if hasattr(regressor, 'feature_names_in_') else []
else:
    # Es un modelo simple
    print(f"   Tipo: LinearRegression simple")
    print(f"   Intercept: {reg_model.intercept_:.6f}")
    regressor = reg_model
    reg_features = list(getattr(reg_model, 'feature_names_in_', []))

print(f"   Features esperados: {len(reg_features)}")

# 3. Predicción con datos originales (baseline)
print("\n✅ 3. PREDICCIÓN BASELINE (datos originales)")
# Pasar como DataFrame con nombres correctos
X_original_df = pd.DataFrame([row_original[reg_features].values], columns=reg_features)
pred_original = reg_model.predict(X_original_df)[0]
precio_original = np.exp(pred_original)

print(f"   Features alineadas: ✓")
print(f"   Predicción (log space): {pred_original:.6f}")
print(f"   Predicción (exponenciado): ${precio_original:,.0f}")
print(f"   Error vs realidad: {abs(precio_original - row_original['SalePrice_Present']):,.0f}")
print(f"   Error %: {abs(precio_original - row_original['SalePrice_Present']) / row_original['SalePrice_Present'] * 100:.2f}%")

# 4. Ahora simular what happens cuando tenemos datos "remodelados"
print("\n⚠️ 4. SIMULACIÓN: Casa con Overall Qual mejorado (5 -> 7)")
X_mejorado_dict = row_original[reg_features].to_dict()
X_mejorado_dict["Overall Qual"] = 7  # Mejoramos a 7
X_mejorado_df = pd.DataFrame([X_mejorado_dict])

pred_mejorado = reg_model.predict(X_mejorado_df)[0]
precio_mejorado = np.exp(pred_mejorado)

print(f"   Overall Qual: {row_original['Overall Qual']:.0f} -> 7")
print(f"   Predicción (log space): {pred_original:.6f} -> {pred_mejorado:.6f}")
print(f"   Predicción (precio): ${precio_original:,.0f} -> ${precio_mejorado:,.0f}")
print(f"   Cambio: ${precio_mejorado - precio_original:,.0f} ({(precio_mejorado - precio_original) / precio_original * 100:+.2f}%)")

# 5. Mostrar algunos coeficientes importantes
print("\n📊 5. COEFICIENTES IMPORTANTES")
coef_dict = dict(zip(reg_features, regressor.coef_))
top_coef = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
for feat, coef in top_coef:
    print(f"   {feat:35s}: {coef:+.8f}")

print("\n" + "="*80)
print("FIN DIAGNÓSTICO")
print("="*80)
