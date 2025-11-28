#!/usr/bin/env python3
"""
Script para analizar y visualizar el sobreajuste del modelo XGBoost.
Compara métricas train vs test y genera gráficos de diagnóstico.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 12)

# Ruta del modelo elegido
MODEL_DIR = Path("models/xgb/completa_present_log_p2_1800_ELEGIDO")
METRICS_FILE = MODEL_DIR / "metrics.json"
META_FILE = MODEL_DIR / "meta.json"

print("="*70)
print("  ANÁLISIS DE OVERFITTING EN XGBOOST")
print("="*70)

# Cargar métricas
if not METRICS_FILE.exists():
    print(f"❌ No se encontró {METRICS_FILE}")
    exit(1)

with open(METRICS_FILE) as f:
    metrics = json.load(f)

with open(META_FILE) as f:
    meta = json.load(f)

print(f"\n✓ Métricas cargadas desde {METRICS_FILE}")
print(f"✓ Meta cargada desde {META_FILE}")

# Extraer datos
train_metrics = metrics['train']
test_metrics = metrics['test']

# ============================================================================
# 1. TABLA COMPARATIVA
# ============================================================================
print("\n" + "="*70)
print("  COMPARACIÓN TRAIN vs TEST")
print("="*70)

comparison = pd.DataFrame({
    'Train': train_metrics,
    'Test': test_metrics,
})

# Calcular ratios y gaps
comparison['Gap Absoluto'] = comparison['Test'] - comparison['Train']
comparison['Ratio (Test/Train)'] = (comparison['Test'] / comparison['Train']).round(2)

# Mostrar
print("\n", comparison[['Train', 'Test', 'Gap Absoluto', 'Ratio (Test/Train)']].to_string())

# ============================================================================
# 2. ANÁLISIS DE SOBREAJUSTE
# ============================================================================
print("\n" + "="*70)
print("  INDICADORES DE SOBREAJUSTE")
print("="*70)

mape_train = train_metrics['MAPE_pct']
mape_test = test_metrics['MAPE_pct']
mape_gap = mape_test - mape_train
mape_ratio = mape_test / mape_train

mae_train = train_metrics['MAE']
mae_test = test_metrics['MAE']
mae_gap = mae_test - mae_train
mae_ratio = mae_test / mae_train

rmse_train = train_metrics['RMSE']
rmse_test = test_metrics['RMSE']
rmse_gap = rmse_test - rmse_train
rmse_ratio = rmse_test / rmse_train

r2_train = train_metrics['R2']
r2_test = test_metrics['R2']
r2_gap = r2_test - r2_train

print(f"\n📊 MAPE (Error Porcentual Medio)")
print(f"   Train: {mape_train:7.2f}%")
print(f"   Test:  {mape_test:7.2f}%")
print(f"   Gap:   {mape_gap:7.2f}% (Test es {mape_ratio:.2f}x peor)")

print(f"\n📊 MAE (Error Absoluto Medio)")
print(f"   Train: ${mae_train:10,.0f}")
print(f"   Test:  ${mae_test:10,.0f}")
print(f"   Gap:   ${mae_gap:10,.0f} (Test es {mae_ratio:.2f}x peor)")

print(f"\n📊 RMSE (Raíz Error Cuadrado Medio)")
print(f"   Train: ${rmse_train:10,.0f}")
print(f"   Test:  ${rmse_test:10,.0f}")
print(f"   Gap:   ${rmse_gap:10,.0f} (Test es {rmse_ratio:.2f}x peor)")

print(f"\n📊 R² Score")
print(f"   Train: {r2_train:.6f}")
print(f"   Test:  {r2_test:.6f}")
print(f"   Gap:   {r2_gap:.6f} ({abs(r2_gap)*100:.2f} puntos porcentuales)")

# Diagnóstico
print("\n" + "="*70)
print("  DIAGNÓSTICO")
print("="*70)

overfitting_severity = "SEVERO" if mape_ratio > 2.5 else "MODERADO" if mape_ratio > 1.5 else "LEVE"
print(f"\n🚨 Severidad de Sobreajuste: {overfitting_severity}")
print(f"   → Ratio MAPE: {mape_ratio:.2f}x (Umbral SEVERO: >2.5x)")

# Análisis residuos
print(f"\n📈 Análisis de Residuos:")
print(f"   Train Skewness:  {train_metrics['residual_skew']:7.3f} (sesgo {'+derecha' if train_metrics['residual_skew'] > 0 else 'izquierda'})")
print(f"   Test Skewness:   {test_metrics['residual_skew']:7.3f} (sesgo {'+derecha' if test_metrics['residual_skew'] > 0 else 'izquierda'})")
print(f"   → Skew aumenta {test_metrics['residual_skew'] / train_metrics['residual_skew']:.1f}x en test")

print(f"\n   Train Kurtosis:  {train_metrics['residual_kurtosis']:7.3f}")
print(f"   Test Kurtosis:   {test_metrics['residual_kurtosis']:7.3f}")
print(f"   → Kurtosis aumenta {test_metrics['residual_kurtosis'] / train_metrics['residual_kurtosis']:.1f}x en test")
print(f"   → Colas más pesadas en test = outliers no capturados")

# Hiperparámetros
print(f"\n🔧 Hiperparámetros del Modelo:")
xgb_params = meta['xgb_params']
print(f"   n_estimators:    {xgb_params['n_estimators']:5} ← ALTO (posible causa)")
print(f"   learning_rate:   {xgb_params['learning_rate']:6.4f} ← Bajo (bien)")
print(f"   max_depth:       {xgb_params['max_depth']:5} ← Bajo (bien)")
print(f"   reg_lambda:      {xgb_params['reg_lambda']:6.2f} ← Moderado")
print(f"   subsample:       {xgb_params['subsample']:6.2f} ← Moderado")

# ============================================================================
# 3. CREAR GRÁFICOS
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Análisis de Overfitting en XGBoost', fontsize=16, fontweight='bold')

# Gráfico 1: MAPE
ax = axes[0, 0]
metrics_names = ['MAPE (%)']
train_vals = [mape_train]
test_vals = [mape_test]
x = np.arange(len(metrics_names))
width = 0.35
ax.bar(x - width/2, train_vals, width, label='Train', color='#2ecc71', alpha=0.8)
ax.bar(x + width/2, test_vals, width, label='Test', color='#e74c3c', alpha=0.8)
ax.set_ylabel('MAPE (%)', fontsize=11)
ax.set_title('Error Porcentual Medio', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()
ax.grid(axis='y', alpha=0.3)
for i, (tv, tev) in enumerate(zip(train_vals, test_vals)):
    ax.text(i - width/2, tv + 0.3, f'{tv:.2f}%', ha='center', va='bottom', fontsize=10)
    ax.text(i + width/2, tev + 0.3, f'{tev:.2f}%', ha='center', va='bottom', fontsize=10, color='#e74c3c', fontweight='bold')

# Gráfico 2: MAE
ax = axes[0, 1]
train_mae_norm = mae_train / 1000
test_mae_norm = mae_test / 1000
ax.bar(['Train', 'Test'], [train_mae_norm, test_mae_norm], color=['#2ecc71', '#e74c3c'], alpha=0.8)
ax.set_ylabel('MAE ($1000s)', fontsize=11)
ax.set_title('Error Absoluto Medio', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for i, (tv, tev) in enumerate(zip([train_mae_norm], [test_mae_norm])):
    ax.text(0, tv + 0.5, f'${tv*1000:,.0f}', ha='center', va='bottom', fontsize=10)
    ax.text(1, tev + 0.5, f'${tev*1000:,.0f}', ha='center', va='bottom', fontsize=10, color='#e74c3c', fontweight='bold')

# Gráfico 3: R² Score
ax = axes[1, 0]
ax.bar(['Train', 'Test'], [r2_train, r2_test], color=['#2ecc71', '#e74c3c'], alpha=0.8)
ax.set_ylabel('R² Score', fontsize=11)
ax.set_title('Coeficiente de Determinación', fontsize=12, fontweight='bold')
ax.set_ylim([0.9, 1.0])
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, linewidth=1)
for i, (tv, tev) in enumerate(zip([r2_train], [r2_test])):
    ax.text(0, tv - 0.005, f'{tv:.4f}', ha='center', va='top', fontsize=10)
    ax.text(1, tev - 0.005, f'{tev:.4f}', ha='center', va='top', fontsize=10, color='#e74c3c', fontweight='bold')

# Gráfico 4: Resumen de Ratios
ax = axes[1, 1]
ratios = [mape_ratio, mae_ratio, rmse_ratio]
labels = ['MAPE', 'MAE', 'RMSE']
colors = ['#e74c3c' if r > 2.5 else '#f39c12' if r > 1.5 else '#2ecc71' for r in ratios]
bars = ax.barh(labels, ratios, color=colors, alpha=0.8)
ax.set_xlabel('Ratio (Test/Train)', fontsize=11)
ax.set_title('Brecha Train-Test (Indicador de Overfitting)', fontsize=12, fontweight='bold')
ax.axvline(x=1.0, color='black', linestyle='-', linewidth=1, alpha=0.3)
ax.axvline(x=1.5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Umbral Moderado')
ax.axvline(x=2.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Umbral Severo')
ax.grid(axis='x', alpha=0.3)
for i, (r, bar) in enumerate(zip(ratios, bars)):
    ax.text(r + 0.05, bar.get_y() + bar.get_height()/2, f'{r:.2f}x', 
            va='center', fontsize=10, fontweight='bold')
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig('analisis/overfitting_analisis.png', dpi=300, bbox_inches='tight')
print("\n\n✅ Gráfico guardado en: analisis/overfitting_analisis.png")

# ============================================================================
# 4. GRÁFICO DE DETERIORO POR MÉTRICA
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

metrics_list = ['MAPE_pct', 'MAE', 'RMSE', 'R2']
train_vals = [train_metrics[m] for m in metrics_list]
test_vals = [test_metrics[m] for m in metrics_list]

# Normalizar para visualización (usando min-max)
# Para MAPE, MAE, RMSE: más alto = peor
# Para R²: más bajo = peor

normalized_data = {
    'MAPE (%)': {'train': mape_train / mape_test, 'test': 1.0},
    'MAE': {'train': mae_train / mae_test, 'test': 1.0},
    'RMSE': {'train': rmse_train / rmse_test, 'test': 1.0},
    'R² Score': {'train': r2_train / r2_test, 'test': 1.0},
}

x = np.arange(len(normalized_data))
width = 0.35
train_vals_norm = [v['train'] for v in normalized_data.values()]
test_vals_norm = [v['test'] for v in normalized_data.values()]

ax.bar(x - width/2, train_vals_norm, width, label='Train (Normalizado)', color='#2ecc71', alpha=0.8)
ax.bar(x + width/2, test_vals_norm, width, label='Test (Normalizado)', color='#e74c3c', alpha=0.8)

ax.set_ylabel('Score Relativo (Train = base)', fontsize=11)
ax.set_title('Deterioro de Rendimiento Train → Test', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(normalized_data.keys())
ax.axhline(y=1.0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('analisis/deterioro_metricas.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico guardado en: analisis/deterioro_metricas.png")

# ============================================================================
# 5. RESUMEN FINAL
# ============================================================================
print("\n" + "="*70)
print("  CONCLUSIÓN")
print("="*70)

if mape_ratio > 2.5:
    print(f"\n🔴 SOBREAJUSTE SEVERO DETECTADO")
    print(f"   El modelo está aprendiendo características específicas del train set")
    print(f"   que NO se generalizan bien al test set.")
    print(f"\n   RECOMENDACIÓN: Reducir complejidad o usar early stopping")
elif mape_ratio > 1.5:
    print(f"\n🟡 SOBREAJUSTE MODERADO DETECTADO")
    print(f"   Hay degradación notable pero el modelo aún generaliza aceptablemente.")
    print(f"\n   RECOMENDACIÓN: Considerar ajustes de hiperparámetros")
else:
    print(f"\n🟢 SOBREAJUSTE LEVE")
    print(f"   El modelo generaliza bien a datos nuevos.")
    print(f"\n   RECOMENDACIÓN: Modelo aceptable tal como está")

print("\n" + "="*70 + "\n")
