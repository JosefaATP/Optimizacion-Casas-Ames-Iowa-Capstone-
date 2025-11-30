"""
Análisis de ROI estilo construcción (como los gráficos que pasaste).
- Solo ROI positivos
- Neighborhoods con nombres completos
- Top 10 cambios agrupados por CATEGORÍA (no detalles específicos)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Cargar datos
bench_path = Path("bench_out/benchmark_remodel2/remodel_benchmark.csv")
changes_path = Path("bench_out/benchmark_remodel2/top_changes_by_tier.csv")
desc_path = Path("Info proyecto/data_description (1).txt")

# Mapeo de neighborhoods
neighborhood_map = {
    'Blmngtn': 'Bloomington',
    'BrDale': 'Briardale',
    'BrkSide': 'Brookside',
    'ClearCr': 'Clear Creek',
    'CollgCr': 'College Creek',
    'Crawfor': 'Crawford',
    'Edwards': 'Edwards',
    'Gilbert': 'Gilbert',
    'Greens': 'Greens',
    'GrnHill': 'Green Hills',
    'IDOTRR': 'Iowa DOT and Rail Road',
    'Landmrk': 'Landmark',
    'MeadowV': 'Meadow Village',
    'Mitchel': 'Mitchell',
    'NAmes': 'North Ames',
    'NoRidge': 'North Ridge',
    'NPkVill': 'Northpark Villa',
    'NridgHt': 'Northridge Heights',
    'OldTown': 'Old Town',
    'SWISU': 'South & West of Iowa State University',
    'Sawyer': 'Sawyer',
    'SawyerW': 'Sawyer West',
    'Somerst': 'Somerset',
    'StoneBr': 'Stone Brook',
    'Timber': 'Timberland',
    'Veenker': 'Veenker',
}

df = pd.read_csv(bench_path)
df_changes = pd.read_csv(changes_path)

# Cargar base de datos original para obtener SalePrice_Present
df_original = pd.read_csv("data/processed/base_completa_sin_nulos.csv")

# Mapear presupuesto a etiqueta
presupuesto_map = {'low': 18, 'mid': 50, 'high': 120}
df['budget_k'] = df['tier'].map(presupuesto_map)

# Merge con SalePrice_Present de la base original
df = df.merge(df_original[['PID', 'SalePrice_Present']], left_on='pid', right_on='PID', how='left')

# Filtrar solo ROI positivos
df_pos = df[df['roi_pct'] > 0].copy()

print(f"✓ Casos totales: {len(df)}")
print(f"✓ ROI positivos: {len(df_pos)} ({100*len(df_pos)/len(df):.1f}%)")
print(f"✓ ROI negativo/cero: {len(df) - len(df_pos)}")

# ==================== GRÁFICO 1: Comparación BEFORE vs AFTER (todas las estimaciones) ====================
fig, ax = plt.subplots(figsize=(14, 8))

presupuesto_vals = [18, 50, 120]

# BEFORE (líneas punteadas)
precio_base_db = []      # Negro: Base de datos (SalePrice_Present)
precio_base_lin = []     # Azul: Regresión lineal
precio_base_xgb = []     # Naranja: XGBoost

# AFTER (líneas sólidas)
precio_opt_db = []       # Negro: Base de datos (precio_opt, que es el ajustado)
precio_opt_lin = []      # Azul: Regresión lineal
precio_opt_xgb = []      # Naranja: XGBoost

for pres in presupuesto_vals:
    data = df_pos[df_pos['budget_k'] == pres]
    if len(data) > 0:
        # BEFORE
        precio_base_db.append(data['SalePrice_Present'].mean())
        precio_base_lin.append(data['price_base_lin'].mean())
        precio_base_xgb.append(data['price_base'].mean())  # XGBoost usa price_base para el modelo
        
        # AFTER
        precio_opt_db.append(data['price_opt'].mean())      # Precio optimizado (calculado)
        precio_opt_lin.append(data['price_opt_lin'].mean())
        precio_opt_xgb.append(data['price_opt'].mean())     # XGBoost price_opt

# BEFORE (líneas punteadas)
ax.plot(presupuesto_vals, precio_base_db, 'o--', linewidth=2.5, markersize=10, 
        label='Base de Datos (BEFORE)', color='black', alpha=0.7)
ax.plot(presupuesto_vals, precio_base_lin, 's--', linewidth=2.5, markersize=10,
        label='Regresión Lineal (BEFORE)', color='#1f77b4', alpha=0.7)
ax.plot(presupuesto_vals, precio_base_xgb, '^--', linewidth=2.5, markersize=10,
        label='XGBoost (BEFORE)', color='#ff7f0e', alpha=0.7)

# AFTER: líneas sólidas (solo Regresión Lineal y XGBoost - sin Base de Datos pq no tenemos AFTER real)
ax.plot(presupuesto_vals, precio_opt_lin, 's-', linewidth=2.5, markersize=10,
        label='Regresión Lineal (AFTER)', color='#1f77b4')
ax.plot(presupuesto_vals, precio_opt_xgb, '^-', linewidth=2.5, markersize=10,
        label='XGBoost (AFTER)', color='#ff7f0e')

ax.set_xlabel("Presupuesto (kUSD)", fontsize=12, fontweight='bold')
ax.set_ylabel("Precio Promedio (USD)", fontsize=12, fontweight='bold')
ax.set_title("Comparación de Precios: BEFORE vs AFTER Remodelación\nPor Estimación y Presupuesto (ROI Positivos)", 
             fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)
ax.legend(fontsize=10, loc='upper left', ncol=2)
ax.set_xticks(presupuesto_vals)
ax.set_xticklabels([f'${p}k' for p in presupuesto_vals])

plt.tight_layout()
plt.savefig("bench_out/benchmark_remodel2/figuras/01_precio_before_after.png", dpi=300)
print("✓ Gráfico 1: Comparación BEFORE vs AFTER")
plt.close()

# ==================== GRÁFICO 2: Box plot ROI + línea de uso de presupuesto ====================
fig, ax1 = plt.subplots(figsize=(12, 7))

# Preparar datos para box plot (solo ROI positivos)
roi_data = [df_pos[df_pos['tier'] == tier]['roi_pct'].values for tier in ['low', 'mid', 'high']]
positions = [1, 2, 3]

# Crear box plot
bp = ax1.boxplot(roi_data, positions=positions, widths=0.5, patch_artist=True,
                  boxprops=dict(facecolor='#87CEEB', alpha=0.7),
                  medianprops=dict(color='black', linewidth=2.5),
                  whiskerprops=dict(linewidth=1.5),
                  capprops=dict(linewidth=1.5))

# Media (puntos azules oscuros)
medias = [df_pos[df_pos['tier'] == tier]['roi_pct'].mean() for tier in ['low', 'mid', 'high']]
ax1.scatter(positions, medias, s=150, color='darkblue', zorder=5, marker='D', label='Media')

ax1.set_ylabel("ROI (%)", fontsize=12, fontweight='bold')
ax1.set_xticks(positions)
ax1.set_xticklabels(['$18k\n(LOW)', '$50k\n(MID)', '$120k\n(HIGH)'], fontsize=11)
ax1.set_ylim(-5, 40)
ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
ax1.grid(alpha=0.3, axis='y')
ax1.set_title("ROI y Eficiencia de Presupuesto por Tier (Solo ROI Positivos)", fontsize=13, fontweight='bold')

# Segundo eje: Línea de uso de presupuesto
ax2 = ax1.twinx()
pct_usado = []
for tier in ['low', 'mid', 'high']:
    data = df_pos[df_pos['tier'] == tier]
    pct = (data['budget_used'] / data['budget']).mean() * 100
    pct_usado.append(pct)

ax2.plot(positions, pct_usado, 'o-', color='#ff7f0e', linewidth=2.5, markersize=10, label='% Presupuesto usado')
ax2.set_ylabel("Presupuesto usado (%)", fontsize=12, fontweight='bold', color='#ff7f0e')
ax2.tick_params(axis='y', labelcolor='#ff7f0e')
ax2.set_ylim(0, 110)

# Leyenda combinada
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig("bench_out/benchmark_remodel2/figuras/02_roi_boxplot_presupuesto.png", dpi=300)
print("✓ Gráfico 2: Box plot ROI + % presupuesto usado")
plt.close()

# ==================== GRÁFICO 2b: BoxPlot ROI por Tier (Solo Positivos) ====================
fig, ax = plt.subplots(figsize=(10, 7))

roi_data = [df_pos[df_pos['tier'] == tier]['roi_pct'].values for tier in ['low', 'mid', 'high']]
positions = [1, 2, 3]
tier_names = ['LOW\n($18k)', 'MID\n($50k)', 'HIGH\n($120k)']

# Crear box plot
bp = ax.boxplot(roi_data, positions=positions, widths=0.6, patch_artist=True,
                  boxprops=dict(facecolor='#87CEEB', alpha=0.8, linewidth=2),
                  medianprops=dict(color='#d62728', linewidth=2.5),
                  whiskerprops=dict(linewidth=1.5, color='gray'),
                  capprops=dict(linewidth=1.5, color='gray'),
                  flierprops=dict(marker='o', markerfacecolor='gray', markersize=5, alpha=0.5))

# Superponer media con puntos
medias = [df_pos[df_pos['tier'] == tier]['roi_pct'].mean() for tier in ['low', 'mid', 'high']]
ax.scatter(positions, medias, s=200, color='darkgreen', zorder=5, marker='D', label='Media', edgecolors='black', linewidth=1.5)

ax.set_ylabel("ROI (%)", fontsize=13, fontweight='bold')
ax.set_xticks(positions)
ax.set_xticklabels(tier_names, fontsize=11, fontweight='bold')
ax.set_title("Distribución de ROI por Presupuesto (Solo ROI Positivos)", fontsize=13, fontweight='bold')
ax.grid(alpha=0.3, axis='y')
ax.legend(fontsize=11, loc='upper right')

# Añadir estadísticas en el gráfico
for i, tier in enumerate(['low', 'mid', 'high']):
    data = df_pos[df_pos['tier'] == tier]['roi_pct']
    stats_text = f"n={len(data)}\nMediana={data.median():.0f}%"
    ax.text(positions[i], ax.get_ylim()[1]*0.95, stats_text, ha='center', va='top', 
            fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig("bench_out/benchmark_remodel2/figuras/02b_roi_boxplot_por_tier.png", dpi=300)
print("✓ Gráfico 2b: BoxPlot ROI por Tier")
plt.close()

# ==================== GRÁFICO 3: ROI por Neighborhood (Top 10) ====================
plt.savefig("bench_out/benchmark_remodel2/figuras/02_roi_boxplot_presupuesto.png", dpi=300)
print("✓ Gráfico 2: Box plot ROI + % presupuesto usado")
plt.close()

# ==================== GRÁFICO 3: ROI por Neighborhood (Top 10) ====================
fig, ax = plt.subplots(figsize=(12, 8))

roi_by_neighborhood = df_pos.groupby('neighborhood')['roi_pct'].mean().sort_values(ascending=False).head(10)
roi_by_neighborhood.index = roi_by_neighborhood.index.map(lambda x: neighborhood_map.get(x, x))

bars = ax.barh(range(len(roi_by_neighborhood)), roi_by_neighborhood.values, color='#1f77b4')
ax.set_yticks(range(len(roi_by_neighborhood)))
ax.set_yticklabels(roi_by_neighborhood.index, fontsize=11)
ax.set_xlabel("ROI Promedio (%)", fontsize=12, fontweight='bold')
ax.set_title("ROI Promedio por Neighborhood - Top 10 (Solo ROI Positivos)", fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Añadir valores en las barras
for i, (bar, val) in enumerate(zip(bars, roi_by_neighborhood.values)):
    ax.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig("bench_out/benchmark_remodel2/figuras/03_roi_by_neighborhood_top10.png", dpi=300)
print("✓ Gráfico 3: ROI por Neighborhood")
plt.close()

# ==================== GRÁFICO 4: Distribución de ROI en intervalos (Stacked bar) ====================
fig, ax = plt.subplots(figsize=(13, 7))

# Definir intervalos de ROI SOLO POSITIVOS
roi_bins = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 1.0]
roi_labels = ['0.00-0.05', '0.05-0.10', '0.10-0.15', '0.15-0.20', '0.20-0.25', '0.25-0.30', '0.30-0.35', '0.35-0.40', '0.40+']

# Filtrar solo ROI positivos y categorizar por bins
df_roi_pos = df[df['roi_pct'] > 0].copy()
df_roi_pos['roi_bin'] = pd.cut(df_roi_pos['roi_pct']/100, bins=roi_bins, labels=roi_labels, right=True)

# Crear tabla cruzada: intervalos vs presupuestos
roi_dist = pd.crosstab(df_roi_pos['roi_bin'], df_roi_pos['tier'], margins=False)
roi_dist = roi_dist[['low', 'mid', 'high']]  # Reordenar columnas

# Colores
colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Stacked bar
roi_dist.plot(kind='bar', stacked=True, ax=ax, color=colors_list, width=0.8, legend=True)

ax.set_xlabel("ROI (intervalos)", fontsize=12, fontweight='bold')
ax.set_ylabel("Frecuencia", fontsize=12, fontweight='bold')
ax.set_title("Distribución de ROI (Solo Positivos) por Intervalos y Presupuesto", fontsize=13, fontweight='bold')
ax.legend(title='Presupuesto', labels=['LOW ($18k)', 'MID ($50k)', 'HIGH ($120k)'], fontsize=10)
ax.grid(alpha=0.3, axis='y')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig("bench_out/benchmark_remodel2/figuras/04_roi_distribution_intervals.png", dpi=300)
print("✓ Gráfico 4: Distribución de ROI positivos en intervalos")
plt.close()

# ==================== Preparar datos de cambios por categoría ====================
def extract_variable_name(change_str):
    """Extrae el nombre de variable limpio (sin detalles de qué opción)."""
    change_str = str(change_str).strip()
    
    # Si tiene guion bajo, toma solo la parte antes (ej: "Exterior 1st_Plywood" -> "Exterior 1st")
    if '_' in change_str:
        var_name = change_str.split('_')[0].strip()
    else:
        var_name = change_str
    
    return var_name

def classify_change_type(variable_name):
    """Clasifica si es una ampliación (cambio cuantitativo) o cambio cualitativo."""
    var = str(variable_name).strip()
    
    # Ampliaciones/expansiones: variables de área, cantidad o espacio (COMPLETA)
    expansion_keywords = [
        'SF',        # Square Feet (cualquier área)
        'Area',      # Garage Area, etc
        'Bedroom',   # Bedroom AbvGr, etc
        'Bath',      # Full Bath, Half Bath, Bsmt Full Bath, etc
        'Porch',     # Open Porch, Enclosed Porch, Screen Porch
        'Deck',      # Wood Deck SF
        'Gr Liv',    # Gr Liv Area (living area)
        'Bsmt',      # Bsmt Unf SF, BsmtFin SF, Total Bsmt SF, etc
        'Flr',       # 1st Flr SF, 2nd Flr SF
        'TotRms',    # TotRms AbvGrd (total rooms)
    ]
    
    for keyword in expansion_keywords:
        if keyword.lower() in var.lower():
            return 'Ampliación'  # Con tilde para consistencia
    
    # Lo demás son cambios cualitativos (materiales, calidad, etc)
    return 'Cambio Cualitativo'

df_changes['variable'] = df_changes['change'].apply(extract_variable_name)
df_changes['tipo_cambio'] = df_changes['variable'].apply(classify_change_type)

# ==================== GRÁFICOS 5: Top cambios por Tier ====================
tiers_order = ['low', 'mid', 'high']
tier_labels = ['LOW ($18k)', 'MID ($50k)', 'HIGH ($120k)']
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, tier in enumerate(tiers_order):
    ax = axes[idx]
    
    # Agrupar cambios por VARIABLE, ordenar por tipo primero (Ampliaciones > Cambios)
    tier_changes = df_changes[df_changes['tier'] == tier].copy()
    var_counts = tier_changes.groupby(['variable', 'tipo_cambio'])['count'].sum().reset_index()
    var_counts = var_counts.sort_values(['tipo_cambio', 'count'], ascending=[False, False])
    
    # Crear etiqueta con tipo de cambio
    var_counts['label'] = var_counts.apply(
        lambda x: f"{x['variable']} (A)" if x['tipo_cambio'] == 'Ampliación' else x['variable'],
        axis=1
    )
    
    top10 = var_counts.head(10)
    bars = ax.barh(range(len(top10)), top10['count'].values, 
                   color=['#d62728' if t == 'Ampliación' else '#1f77b4' for t in top10['tipo_cambio']])
    
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10['label'].values, fontsize=10)
    ax.set_xlabel("Frecuencia", fontsize=11, fontweight='bold')
    ax.set_title(f"Top 10 Variables - {tier_labels[idx]}", fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Valores en las barras
    for i, val in enumerate(top10['count'].values):
        ax.text(val + 2, i, str(int(val)), va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig("bench_out/benchmark_remodel2/figuras/05_top_variables_por_tier.png", dpi=300)
print("✓ Gráfico 5: Top variables por tier (Ampliaciones destacadas)")
plt.close()

# ==================== Estadísticas finales ====================
print("\n" + "="*60)
print("RESUMEN ROI (SOLO POSITIVOS)")
print("="*60)
for tier in ['low', 'mid', 'high']:
    data = df_pos[df_pos['tier'] == tier]
    if len(data) > 0:
        print(f"\n{tier.upper()} Tier ({presupuesto_map[tier]}k):")
        print(f"  - Casos: {len(data)}")
        print(f"  - ROI Promedio: {data['roi_pct'].mean():.2f}%")
        print(f"  - ROI Mediana: {data['roi_pct'].median():.2f}%")
        print(f"  - ROI Min-Max: {data['roi_pct'].min():.2f}% - {data['roi_pct'].max():.2f}%")

print(f"\nOverall ROI Promedio: {df_pos['roi_pct'].mean():.2f}%")
print(f"Overall ROI Mediana: {df_pos['roi_pct'].median():.2f}%")

print("\n" + "="*60)
print("TOP 10 VARIABLES MÁS MODIFICADAS (GLOBAL)")
print("="*60)
global_vars = df_changes.groupby(['variable', 'tipo_cambio'])['count'].sum().reset_index()
global_vars = global_vars.sort_values(['tipo_cambio', 'count'], ascending=[False, False]).head(10)
for _, row in global_vars.iterrows():
    tipo_str = f"[{row['tipo_cambio'][0]}]" if row['tipo_cambio'] == 'Ampliación' else "      "
    print(f"  {tipo_str} {row['variable']:.<35} {int(row['count']):>5}")

print("\n" + "="*60)
print("RESUMEN POR TIPO DE CAMBIO")
print("="*60)
tipo_summary = df_changes.groupby('tipo_cambio')['count'].sum()
for tipo, cnt in tipo_summary.items():
    pct = 100*cnt/df_changes['count'].sum()
    print(f"  {tipo:.<40} {int(cnt):>5} ({pct:>5.1f}%)")

print("\n✓ Análisis completado. Gráficos guardados en bench_out/benchmark_remodel2/figuras/")
