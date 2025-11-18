# optimization/remodel/utils_analisis.py
"""
Utilidades para el análisis de resultados
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def crear_visualizaciones(df_resultados: pd.DataFrame, directorio_salida: str = "graficos"):
    """
    Crea visualizaciones de los resultados del análisis
    """
    import os
    os.makedirs(directorio_salida, exist_ok=True)
    
    # Filtrar ejecuciones exitosas
    df_exitosos = df_resultados[df_resultados['error'].isna()] if 'error' in df_resultados.columns else df_resultados
    
    if df_exitosos.empty:
        print("No hay datos para crear visualizaciones")
        return
    
    # Configurar estilo de gráficos
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Boxplot de aumento de utilidad por presupuesto
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_exitosos, x='presupuesto', y='aumento_utilidad')
    plt.title('Distribución del Aumento de Utilidad por Presupuesto')
    plt.xlabel('Presupuesto ($)')
    plt.ylabel('Aumento de Utilidad ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{directorio_salida}/boxplot_utilidad_presupuesto.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Scatter plot: Presupuesto vs ROI
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_exitosos, x='presupuesto', y='roi_%', alpha=0.6)
    plt.title('ROI vs Presupuesto')
    plt.xlabel('Presupuesto ($)')
    plt.ylabel('ROI (%)')
    plt.tight_layout()
    plt.savefig(f'{directorio_salida}/scatter_roi_presupuesto.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Histograma de márgenes
    plt.figure(figsize=(12, 6))
    df_exitosos['margen_%'].hist(bins=30, alpha=0.7)
    plt.title('Distribución de Márgenes de Utilidad')
    plt.xlabel('Margen (%)')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.savefig(f'{directorio_salida}/histograma_margenes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Heatmap de correlaciones
    columnas_numericas = ['presupuesto', 'aumento_utilidad', 'total_cost', 'precio_remodelada', 'roi_%', 'margen_%']
    df_corr = df_exitosos[columnas_numericas].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Matriz de Correlación de Métricas')
    plt.tight_layout()
    plt.savefig(f'{directorio_salida}/heatmap_correlaciones.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizaciones guardadas en directorio: {directorio_salida}")

def generar_reporte_analisis(df_resultados: pd.DataFrame, df_metricas: pd.DataFrame):
    """
    Genera un reporte de análisis en formato Markdown
    """
    reporte = "# Reporte de Análisis de Optimización\n\n"
    
    # Estadísticas básicas
    total_ejecuciones = len(df_resultados)
    ejecuciones_exitosas = len(df_resultados[df_resultados['error'].isna()]) if 'error' in df_resultados.columns else total_ejecuciones
    tasa_exito = (ejecuciones_exitosas / total_ejecuciones) * 100
    
    reporte += f"## Resumen Ejecutivo\n\n"
    reporte += f"- **Total de ejecuciones**: {total_ejecuciones}\n"
    reporte += f"- **Ejecuciones exitosas**: {ejecuciones_exitosas}\n"
    reporte += f"- **Tasa de éxito**: {tasa_exito:.1f}%\n\n"
    
    # Métricas por presupuesto
    reporte += "## Métricas por Presupuesto\n\n"
    for presupuesto in df_metricas.index:
        m = df_metricas.loc[presupuesto]
        reporte += f"### Presupuesto ${presupuesto:,}\n\n"
        reporte += f"- **Aumento utilidad promedio**: ${m.get('aumento_utilidad_mean', 0):,.0f}\n"
        reporte += f"- **Costo promedio**: ${m.get('total_cost_mean', 0):,.0f}\n"
        reporte += f"- **Valor final promedio**: ${m.get('precio_remodelada_mean', 0):,.0f}\n"
        reporte += f"- **ROI promedio**: {m.get('roi_%_mean', 0):.1f}%\n"
        reporte += f"- **Margen promedio**: {m.get('margen_%_mean', 0):.1f}%\n\n"
    
    # Guardar reporte
    with open("reporte_analisis.md", "w", encoding="utf-8") as f:
        f.write(reporte)
    
    print("Reporte de análisis guardado como: reporte_analisis.md")
    