#!/usr/bin/env python3
"""
Análisis detallado de ROI - 500 casas benchmark
Genera gráficos y estadísticas para:
1. Distribución de ROI (solo positivos)
2. ROI por presupuesto (solo positivos)
3. ROI por neighborhood
4. Top cambios (sin especificar a qué se cambia)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import re

# Configuración
BENCH_DIR = Path("bench_out/benchmark_remodel")
CSV_PATH = BENCH_DIR / "remodel_benchmark.csv"
FIG_DIR = BENCH_DIR / "figuras"
FIG_DIR.mkdir(exist_ok=True)

# Cargar datos
df = pd.read_csv(CSV_PATH)

# Filtrar solo ROI positivos (para análisis detallado)
df_positive = df[df['roi_pct'] > 0].copy()

print("=" * 80)
print("ANÁLISIS DETALLADO DE ROI - 500 CASAS")
print("=" * 80)

# ============================================================================
# 1. ESTADÍSTICAS DE ROI POSITIVO
# ============================================================================
print("\n1. ROI POSITIVO - ESTADÍSTICAS")
print("-" * 80)
print(f"Total casas con ROI > 0: {len(df_positive)} / {len(df)} ({len(df_positive)/len(df)*100:.1f}%)")
print(f"\nMedia ROI (positivos): {df_positive['roi_pct'].mean():.2f}%")
print(f"Mediana ROI (positivos): {df_positive['roi_pct'].median():.2f}%")
print(f"Std Dev: {df_positive['roi_pct'].std():.2f}%")
print(f"Min: {df_positive['roi_pct'].min():.2f}%")
print(f"Max: {df_positive['roi_pct'].max():.2f}%")
print(f"P25: {df_positive['roi_pct'].quantile(0.25):.2f}%")
print(f"P75: {df_positive['roi_pct'].quantile(0.75):.2f}%")

# ============================================================================
# 2. GRÁFICO 1: DISTRIBUCIÓN DE ROI (SOLO POSITIVOS, histograma)
# ============================================================================
print("\n2. Generando gráfico: Distribución ROI (positivos)")
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(df_positive['roi_pct'], bins=40, color='#2ecc71', edgecolor='black', alpha=0.7)
ax.axvline(df_positive['roi_pct'].mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {df_positive["roi_pct"].mean():.2f}%')
ax.axvline(df_positive['roi_pct'].median(), color='blue', linestyle='--', linewidth=2, label=f'Mediana: {df_positive["roi_pct"].median():.2f}%')
ax.set_xlabel('ROI (%)', fontsize=12)
ax.set_ylabel('Frecuencia', fontsize=12)
ax.set_title('Distribución ROI (solo casos positivos)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "01_roi_distribution_positive.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print("✓ Guardado: 01_roi_distribution_positive.png")

# ============================================================================
# 3. ROI POR PRESUPUESTO (solo positivos)
# ============================================================================
print("\n3. ROI por PRESUPUESTO (solo positivos)")
print("-" * 80)
budget_stats = []
for tier in ['low', 'mid', 'high']:
    tier_data = df_positive[df_positive['tier'] == tier]
    budget_amt = tier_data['budget'].iloc[0] if len(tier_data) > 0 else 0
    stats = {
        'Tier': tier.upper(),
        'Budget': f"${budget_amt:,.0f}",
        'Casas': len(tier_data),
        'Media ROI %': tier_data['roi_pct'].mean(),
        'Mediana ROI %': tier_data['roi_pct'].median(),
        'Std Dev': tier_data['roi_pct'].std(),
        'Min': tier_data['roi_pct'].min(),
        'Max': tier_data['roi_pct'].max(),
    }
    budget_stats.append(stats)
    print(f"\n{tier.upper()} (${budget_amt:,.0f}):")
    print(f"  Casas analizadas: {len(tier_data)}")
    print(f"  Media ROI: {tier_data['roi_pct'].mean():.2f}%")
    print(f"  Mediana ROI: {tier_data['roi_pct'].median():.2f}%")
    print(f"  Rango: {tier_data['roi_pct'].min():.2f}% a {tier_data['roi_pct'].max():.2f}%")

# ============================================================================
# 4. GRÁFICO 2: ROI POR PRESUPUESTO (estilo construccion)
# ============================================================================
print("\n4. Generando gráfico: ROI por presupuesto (box plots)")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Box plot por tier
data_by_tier = [df_positive[df_positive['tier'] == t]['roi_pct'].values for t in ['low', 'mid', 'high']]
bp = axes[0].boxplot(data_by_tier, labels=['LOW\n($18k)', 'MID\n($50k)', 'HIGH\n($120k)'], patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('#3498db')
axes[0].set_ylabel('ROI (%)', fontsize=11)
axes[0].set_title('Distribución ROI por Presupuesto (positivos)', fontsize=12, fontweight='bold')
axes[0].grid(alpha=0.3, axis='y')

# Violin plot
parts = axes[1].violinplot(data_by_tier, positions=[1, 2, 3], showmeans=True, showmedians=True)
axes[1].set_xticks([1, 2, 3])
axes[1].set_xticklabels(['LOW\n($18k)', 'MID\n($50k)', 'HIGH\n($120k)'])
axes[1].set_ylabel('ROI (%)', fontsize=11)
axes[1].set_title('Densidad ROI por Presupuesto (positivos)', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3, axis='y')

fig.tight_layout()
fig.savefig(FIG_DIR / "02_roi_by_budget.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print("✓ Guardado: 02_roi_by_budget.png")

# ============================================================================
# 5. GRÁFICO 3: ROI POR NEIGHBORHOOD (solo positivos)
# ============================================================================
print("\n5. Generando gráfico: ROI por Neighborhood")

# Agrupar por neighborhood
neigh_stats = df_positive.groupby('neighborhood').agg({
    'roi_pct': ['mean', 'median', 'count'],
    'pid': 'count'
}).round(2)
neigh_stats.columns = ['ROI_mean', 'ROI_median', 'count', 'casas']
neigh_stats = neigh_stats.sort_values('ROI_mean', ascending=False)

print("\nTop 10 Neighborhoods por ROI promedio (positivos):")
for idx, (neigh, row) in enumerate(neigh_stats.head(10).iterrows(), 1):
    print(f"{idx:2d}. {neigh:20s} | Media: {row['ROI_mean']:7.2f}% | Casas: {int(row['casas'])}")

# Gráfico barras
fig, ax = plt.subplots(figsize=(12, 7))
neighborhoods = neigh_stats.index[:15]  # Top 15
values = neigh_stats['ROI_mean'].iloc[:15]
colors = ['#27ae60' if v > 0 else '#e74c3c' for v in values]

bars = ax.barh(neighborhoods, values, color=colors, edgecolor='black', alpha=0.8)
ax.set_xlabel('ROI Promedio (%)', fontsize=12)
ax.set_title('ROI Promedio por Neighborhood (solo casos positivos, top 15)', fontsize=13, fontweight='bold')
ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
ax.grid(alpha=0.3, axis='x')

# Añadir valores en las barras
for i, (bar, val) in enumerate(zip(bars, values)):
    ax.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', 
            va='center', fontsize=10)

fig.tight_layout()
fig.savefig(FIG_DIR / "03_roi_by_neighborhood.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print("✓ Guardado: 03_roi_by_neighborhood.png")

# ============================================================================
# 6. TOP CAMBIOS (agregando solo el nombre de la variable)
# ============================================================================
print("\n6. Analizando TOP CAMBIOS (solo variables, sin especificar qué cambian)")

# Extraer todos los cambios de ROI positivos
all_changes = []
for idx, row in df_positive.iterrows():
    if row['changes'] and isinstance(row['changes'], list):
        all_changes.extend(row['changes'])

# Extraer el nombre de la variable (primera palabra antes de :)
variable_changes = []
for change in all_changes:
    # Patrón: "Variable Name: base -> value ..."
    match = re.match(r'^([^:]+):', change)
    if match:
        var_name = match.group(1).strip()
        variable_changes.append(var_name)

# Contar frecuencias
var_counter = Counter(variable_changes)
top_vars = var_counter.most_common(15)

print("\nTop 15 VARIABLES que cambian (en casos con ROI positivo):")
for i, (var, count) in enumerate(top_vars, 1):
    print(f"{i:2d}. {var:30s} | Apariciones: {count:4d}")

# ============================================================================
# 7. GRÁFICO 4: TOP CAMBIOS (variables)
# ============================================================================
print("\n7. Generando gráfico: Top variables que cambian")
fig, ax = plt.subplots(figsize=(12, 7))

vars_names = [v[0] for v in top_vars]
vars_counts = [v[1] for v in top_vars]

bars = ax.barh(vars_names, vars_counts, color='#9b59b6', edgecolor='black', alpha=0.8)
ax.set_xlabel('Frecuencia de cambio', fontsize=12)
ax.set_title('Top 15 Variables que cambian (casos ROI positivo)', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3, axis='x')

# Añadir valores
for bar, val in zip(bars, vars_counts):
    ax.text(val + 10, bar.get_y() + bar.get_height()/2, str(val), 
            va='center', fontsize=10)

fig.tight_layout()
fig.savefig(FIG_DIR / "04_top_variables.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print("✓ Guardado: 04_top_variables.png")

# ============================================================================
# RESUMEN FINAL
# ============================================================================
print("\n" + "=" * 80)
print("RESUMEN FINAL - RECOMENDACIONES")
print("=" * 80)
print(f"""
✓ Casas rentables (ROI > 0): {len(df_positive)} de {len(df)} ({len(df_positive)/len(df)*100:.1f}%)

✓ ROI promedio (rentables): {df_positive['roi_pct'].mean():.2f}%
  - LOW budget:  {df_positive[df_positive['tier']=='low']['roi_pct'].mean():.2f}%
  - MID budget:  {df_positive[df_positive['tier']=='mid']['roi_pct'].mean():.2f}%
  - HIGH budget: {df_positive[df_positive['tier']=='high']['roi_pct'].mean():.2f}%

✓ Mejores neighborhoods:
  - {neigh_stats.index[0]}: {neigh_stats['ROI_mean'].iloc[0]:.2f}%
  - {neigh_stats.index[1]}: {neigh_stats['ROI_mean'].iloc[1]:.2f}%
  - {neigh_stats.index[2]}: {neigh_stats['ROI_mean'].iloc[2]:.2f}%

✓ Variables más modificadas:
  - {top_vars[0][0]}: {top_vars[0][1]} veces
  - {top_vars[1][0]}: {top_vars[1][1]} veces
  - {top_vars[2][0]}: {top_vars[2][1]} veces

📁 Todos los gráficos guardados en: {FIG_DIR}/
""")

print("✓ ANÁLISIS COMPLETADO")
