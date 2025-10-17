"""
=========
Configuración de parámetros, costos y restricciones para los modelos de
optimización de viviendas.

Basado en el modelo matemático del Grupo 5 - Capstone 2025
"""

import numpy as np

# ============================================================================
# ATRIBUTOS ACCIONABLES Y COSTOS UNITARIOS
# ============================================================================

# Estructura: {nombre: (costo_unitario, es_discreta, min, max, descripción)}

ATRIBUTOS_ACCIONABLES = {
    # 'NombreAtributo': (costo_unitario, es_discreta, min, max, descripción)
    # Áreas (continuas) - Costo por pie cuadrado
    'Gr Liv Area': (150, False, 600, 4000, 'Área habitable sobre suelo ($/sqft)'),
    'Total Bsmt SF': (100, False, 0, 2000, 'Área total sótano ($/sqft)'),
    '1st Flr SF': (120, False, 600, 3000, 'Área primer piso ($/sqft)'),
    '2nd Flr SF': (110, False, 0, 2000, 'Área segundo piso ($/sqft)'),
    
    # Porches y exteriores
    'Wood Deck SF': (50, False, 0, 500, 'Terraza de madera ($/sqft)'),
    'Open Porch SF': (45, False, 0, 400, 'Porche abierto ($/sqft)'),
    'Enclosed Porch': (60, False, 0, 300, 'Porche cerrado ($/sqft)'),
    'Screen Porch': (55, False, 0, 300, 'Porche con mosquitero ($/sqft)'),
    '3Ssn Porch': (65, False, 0, 300, 'Porche 3 estaciones ($/sqft)'),
    
    # Áreas especiales
    'Pool Area': (200, False, 0, 500, 'Área piscina ($/sqft)'),
    'Garage Area': (80, False, 0, 1000, 'Área garaje ($/sqft)'),
    
    # Cantidades discretas
    # **VALORES CORREGIDOS SEGÚN PDF**
    'Full Bath': (25000, True, 1, 4, 'Baños completos ($/unidad)'),
    'Half Bath': (10000, True, 0, 2, 'Medios baños ($/unidad)'),
    'Bedroom AbvGr': (5000, True, 1, 6, 'Dormitorios sobre suelo ($/unidad)'),
    'Kitchen AbvGr': (15000, True, 1, 2, 'Cocinas ($/unidad)'),
    'TotRms AbvGrd': (3000, True, 3, 12, 'Total habitaciones ($/unidad)'),
    'Fireplaces': (3500, True, 0, 3, 'Chimeneas ($/unidad)'),
    'Garage Cars': (10000, True, 0, 4, 'Capacidad garaje en autos ($/auto)'),
    
    # Calidades ordinales (1-10 o 1-5)
    'Overall Qual': (20000, True, 1, 10, 'Calidad general ($/nivel)'),
    'Overall Cond': (15000, True, 1, 10, 'Condición general ($/nivel)'),
    'Exter Qual': (8000, True, 1, 5, 'Calidad exterior ($/nivel)'),
    'Exter Cond': (6000, True, 1, 5, 'Condición exterior ($/nivel)'),
    'Bsmt Qual': (7000, True, 0, 5, 'Calidad sótano ($/nivel)'),
    'Bsmt Cond': (5000, True, 0, 5, 'Condición sótano ($/nivel)'),
    'Heating QC': (6000, True, 1, 5, 'Calidad calefacción ($/nivel)'),
    'Kitchen Qual': (10000, True, 1, 5, 'Calidad cocina ($/nivel)'),
    'Fireplace Qu': (4000, True, 0, 5, 'Calidad chimenea ($/nivel)'),
    'Garage Qual': (5000, True, 0, 5, 'Calidad garaje ($/nivel)'),
    'Garage Cond': (4000, True, 0, 5, 'Condición garaje ($/nivel)'),
    'Pool QC': (8000, True, 0, 5, 'Calidad piscina ($/nivel)'),
}

# Atributos que NO se pueden modificar en remodelación
ATRIBUTOS_FIJOS = [
    'Year Built',
    'Year Remod/Add',
    'Lot Frontage',
    'Lot Area',
    'MS SubClass',
    'MS Zoning',
    'Street',
    'Alley',
    'Lot Shape',
    'Land Contour',
    'Lot Config',
    'Land Slope',
    'Neighborhood',
    'Condition 1',
    'Condition 2',
    'Bldg Type',
    'House Style',
    #-------
    'Utilities',
    'Roof Style',
    'Roof Matl',
    'Exterior 1st',
    'Exterior 2nd',
    'Mas Vnr Type',
    'Mas Vnr Area',
    'Foundation',
    'Bsmt Exposure',
    'BsmtFin Type 1',
    'BsmtFin SF 1',
    'BsmtFin Type 2',
    'BsmtFin SF 2',
    'Bsmt Unf SF',
    'Heating',
    'Central Air',
    'Electrical',
    'Low Qual Fin SF',
    'Bsmt Full Bath',
    'Bsmt Half Bath',
    'FunctioNo aplical',
    'Garage Type',
    'Garage Yr Blt',
    'Garage Finish',
    'Paved Drive',
    'Fence',
    'Misc Feature',
    'Misc Val',
    'Mo Sold',
    'Yr Sold',
    'Sale Type',
    'Sale Condition',
]


# ============================================================================
# COSTOS DE MATERIALES (Variables categóricas)
# ============================================================================
# Define costos de materiales categóricos (solo puedes elegir UNO)

# **COSTOS ACTUALIZADOS Y NUEVOS**

# Costos Utilities (Actualizados según PDF)
COSTOS_UTILITIES = {
    'AllPub': 31750,
    'NoSewr': 39500,
    'NoSeWa': 22000,
    'ELO': 20000,
}

# Costos Roof Style (Mantienen valores iniciales)
COSTOS_ROOF_STYLE = {
    'Flat': 5000,
    'Gable': 8000,
    'Gambrel': 10000,
    'Hip': 12000,
    'Mansard': 15000,
    'Shed': 6000,
}

# Costos Roof Matl (Actualizados según PDF)
COSTOS_ROOF_MATERIAL = {
    'ClyTile': 15000,    # Tejas de arcilla (Coste fijo estimado)
    'CompShg': 6,        # $6/sqft (desde PDF)
    'Membran': 12000,    # Membrana (Coste fijo estimado)
    'Metal': 8.99,       # $8.99/sqft (desde PDF)
    'Roll': 3.7,         # $3.7/sqft (desde PDF)
    'Tar&Grv': 5.5,      # $5.5/sqft (desde PDF)
    'WdShake': 14000,    # Shakes de madera (Coste fijo estimado)
    'WdShngl': 6.35,     # $6.35/sqft (desde PDF)
}

# Exterior Materials (Actualizados según PDF)
COSTOS_EXTERIOR = {
    'AsbShng': 11.5,
    'AsphShn': 1.5,
    'BrkComm': 1.21,
    'BrkFace': 15,
    'CBlock': 8,       # Estimado promedio
    'CemntBd': 7,      # Estimado promedio
    'HdBoard': 5,      # Estimado promedio
    'ImStucc': 10,     # Estimado promedio
    'MetalSd': 5.48,
    'Plywood': 3,
    'PreCast': 10,     # Estimado promedio
    'Stone': 27.5,
    'Stucco': 12,
    'VinylSd': 7.46,
    'Wd Sdng': 9000,
    'WdShing': 8000,
}

# Matriz de compatibilidad: A[estilo][material] = 1 si compatible
ROOF_COMPATIBILITY = {
    'Gable':   {'CompShg': 1, 'Metal': 1, 'ClyTile': 1, 'WdShngl': 1, 'WdShake': 1, 'Membran': 0},
    'Hip':     {'CompShg': 1, 'Metal': 1, 'ClyTile': 1, 'WdShngl': 1, 'WdShake': 1, 'Membran': 0},
    'Flat':    {'CompShg': 0, 'Metal': 1, 'ClyTile': 0, 'WdShngl': 0, 'WdShake': 0, 'Membran': 1},
    'Mansard': {'CompShg': 1, 'Metal': 1, 'ClyTile': 1, 'WdShngl': 1, 'WdShake': 1, 'Membran': 0},
    'Shed':    {'CompShg': 1, 'Metal': 1, 'ClyTile': 0, 'WdShngl': 1, 'WdShake': 0, 'Membran': 1},
    'Gambrel': {'CompShg': 1, 'Metal': 1, 'ClyTile': 1, 'WdShngl': 1, 'WdShake': 1, 'Membran': 0},
}

# **NUEVOS COSTOS FALTANTES (Valores basados en PDF y estimaciones)**
COSTO_BSMT_FINISH_SQFT = 15 # $15/sqft (desde PDF)
COSTO_CENTRAL_AIR = 5362 # $5,362 (desde PDF)

# Electrical
COSTOS_ELECTRICAL = {
    'SBrkr': 5000,
    'FuseA': 3000,
    'FuseF': 2500,
    'FuseP': 2000,
    'Mix': 3500,
}

# Heating (Actualizados según PDF)
COSTOS_HEATING = {
    'Floor': 1773,
    'GasA': 5750,
    'GasW': 8500,
    'Grav': 6300,
    'OthW': 4900,
    'Wall': 3700,
}

# Kitchen Qual (Basado en valores del PDF)
COSTOS_KITCHEN_QUAL = {
    'Po': 13000,
    'Fa': 27750, # Interpolado
    'TA': 42500,
    'Gd': 111250, # Interpolado
    'Ex': 180000,
}

# Fence (Costos por categoría y construcción)
COSTOS_FENCE_CAT = {
    'NA': 0,
    'MnWw': 1000,
    'GdWo': 2000,
    'MnPrv': 3000,
    'GdPrv': 4000,
}
COSTO_FENCE_PSF = 8 # Costo de construcción por pie cuadrado (Estimado)

# Paved Drive
COSTOS_PAVEDDRIVE = {
    'N': 0,
    'P': 1000,
    'Y': 3000,
}

# Pool Qual (Basado en valores del PDF)
COSTOS_POOL_QUAL = {
    'NA': 0,
    'Po': 500,
    'Fa': 19000,
    'TA': 77000, # Interpolado
    'Gd': 77000,
    'Ex': 135000,
}

# Garage Finish
COSTOS_GARAGE_FINISH = {
    'NA': 0,
    'Unf': 2000,
    'RFn': 4000,
    'Fin': 8000,
}

# ============================================================================
# COSTOS DE DEMOLICIÓN (Para remodelación)
# ============================================================================

#Define cuánto cuesta demoler algo existente (para remodelación)
COSTO_DEMOLICION_POR_SQFT = 1.65 # $1.65/sqft


# ============================================================================
# RESTRICCIONES TÉCNICAS Y CONSTRUCTIVAS
# ============================================================================

# Áreas mínimas por tipo de habitación (en sqft)
AREA_MINIMA = {
    'Bedroom': 80,      # Dormitorio mínimo 80 sqft
    'FullBath': 50,     # Baño completo mínimo 50 sqft
    'HalfBath': 20,     # Medio baño mínimo 20 sqft
    'Kitchen': 100,     # Cocina mínima 100 sqft
}

# Restricciones de proporción
# El segundo piso NO puede ser más grande que el primer piso (físicamente imposible)
# El sótano NO puede ser más grande que el primer piso
PROPORCION_MAX_SEGUNDO_PISO = 1.0  # 2ndFlr ≤ 1stFlr
PROPORCION_MAX_SOTANO = 1.0        # Bsmt ≤ 1stFlr

# Límites de construcción
MAX_PISOS = 2
OCUPACION_MAX_TERRENO = 0.7  # Máximo 70% del LotArea

# ============================================================================
# CALIDADES POR DEFECTO PARA CONSTRUCCIÓN DESDE CERO
# ============================================================================
# Según tu PDF: "Cuando se construye desde cero, la calidad será excelente"

#------------
#Define valores por defecto cuando construyes desde cero
#------------
CALIDADES_CONSTRUCCION_NUEVA = {
    'OverallQual': 9,    # Excelente (9-10)
    'OverallCond': 9,
    'ExterQual': 5,      # Excellent (en escala 1-5)
    'ExterCond': 5,
    'BsmtQual': 5,
    'BsmtCond': 5,
    'HeatingQC': 5,
    'KitchenQual': 5,
    'GarageQual': 5,
    'GarageCond': 5,
}

# ============================================================================
# PARÁMETROS DE EXPANSIÓN PORCENTUAL (Sección 6 del PDF)
# ============================================================================

# Conjunto C de áreas exteriores sujetas a ampliación porcentual.
# El valor asociado es el costo unitario por pie cuadrado (C_c) tomado de ATRIBUTOS_ACCIONABLES.
ATRIBUTOS_EXPANSION_PORCENTUAL = {
    'Garage Area': 80, 
    'Wood Deck SF': 50,
    'Open Porch SF': 45,
    'Enclosed Porch': 60,
    '3Ssn Porch': 65,
    'Screen Porch': 55,
    'Pool Area': 200,
}

# Parámetros de áreas fijas para agregados puntuales (ft²) - Secc. 6, pág 21
AREA_AGREGADOS_PUNTUALES = {
    'Full Bath': 40,
    'Half Bath': 20,
    'Kitchen AbvGr': 75,
    'Bedroom AbvGr': 70,
} 

# ============================================================================
# CONFIGURACIÓN DE XGBOOST
# ============================================================================

XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
}

# ============================================================================
# CONFIGURACIÓN DE GUROBI
# ============================================================================
#Define parámetros del solver de optimización
GUROBI_PARAMS = {
    'LogToConsole': 0,      # No mostrar log en consola
    'TimeLimit': 300,       # 5 minutos máximo
    'MIPGap': 0.01,         # 1% de gap de optimalidad
    'NonConvex': 2,         # Permitir no-convexidad
}

# ============================================================================
# PATHS DE ARCHIVOS
# ============================================================================

DATA_PATH = './data/raw/casas_completas_con_present.csv'
# Ruta donde se persistirá el modelo entrenado
MODEL_PATH = "./models/xgb/completa_present_log_p2_1800_ELEGIDO/model_xgb.joblib"
RESULTS_PATH = '../optimizacion/results/'

print("✅ Archivo config.py creado correctamente")