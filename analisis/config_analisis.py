# optimization/remodel/config_analisis.py
"""
Configuración para el análisis de resultados
"""

class ConfigAnalisis:
    # Número de casas a analizar
    NUM_CASAS = 100
    
    # Presupuestos a evaluar
    PRESUPUESTOS = [15000, 40000, 75000, 200000]
    
    # Configuración de archivos
    ARCHIVO_RESULTADOS = "resultados_detallados.csv"
    ARCHIVO_METRICAS = "metricas_agregadas.csv"
    
    # Configuración de random seed para reproducibilidad
    RANDOM_SEED = 42
    
    # Timeout por ejecución (segundos)
    TIMEOUT_EJECUCION = 300  # 5 minutos
    
    # Mostrar progreso detallado
    VERBOSE = True