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
    'GrLivArea': (150, False, 600, 4000, 'Área habitable sobre suelo ($/sqft)'),
    'TotalBsmtSF': (100, False, 0, 2000, 'Área total sótano ($/sqft)'),
    '1stFlrSF': (120, False, 600, 3000, 'Área primer piso ($/sqft)'),
    '2ndFlrSF': (110, False, 0, 2000, 'Área segundo piso ($/sqft)'),
    
    # Porches y exteriores
    'WoodDeckSF': (50, False, 0, 500, 'Terraza de madera ($/sqft)'),
    'OpenPorchSF': (45, False, 0, 400, 'Porche abierto ($/sqft)'),
    'EnclosedPorch': (60, False, 0, 300, 'Porche cerrado ($/sqft)'),
    'ScreenPorch': (55, False, 0, 300, 'Porche con mosquitero ($/sqft)'),
    '3SsnPorch': (65, False, 0, 300, 'Porche 3 estaciones ($/sqft)'),
    
    # Áreas especiales
    'PoolArea': (200, False, 0, 500, 'Área piscina ($/sqft)'),
    'GarageArea': (80, False, 0, 1000, 'Área garaje ($/sqft)'),
    
    # Cantidades discretas
    'FullBath': (12500, True, 1, 4, 'Baños completos ($/unidad)'),
    'HalfBath': (4000, True, 0, 2, 'Medios baños ($/unidad)'),
    'BedroomAbvGr': (5000, True, 1, 6, 'Dormitorios sobre suelo ($/unidad)'),
    'KitchenAbvGr': (15000, True, 1, 2, 'Cocinas ($/unidad)'),
    'TotRmsAbvGrd': (3000, True, 3, 12, 'Total habitaciones ($/unidad)'),
    'Fireplaces': (3500, True, 0, 3, 'Chimeneas ($/unidad)'),
    'GarageCars': (10000, True, 0, 4, 'Capacidad garaje en autos ($/auto)'),
    
    # Calidades ordinales (1-10 o 1-5)
    'OverallQual': (20000, True, 1, 10, 'Calidad general ($/nivel)'),
    'OverallCond': (15000, True, 1, 10, 'Condición general ($/nivel)'),
    'ExterQual': (8000, True, 1, 5, 'Calidad exterior ($/nivel)'),
    'ExterCond': (6000, True, 1, 5, 'Condición exterior ($/nivel)'),
    'BsmtQual': (7000, True, 0, 5, 'Calidad sótano ($/nivel)'),
    'BsmtCond': (5000, True, 0, 5, 'Condición sótano ($/nivel)'),
    'HeatingQC': (6000, True, 1, 5, 'Calidad calefacción ($/nivel)'),
    'KitchenQual': (10000, True, 1, 5, 'Calidad cocina ($/nivel)'),
    'FireplaceQu': (4000, True, 0, 5, 'Calidad chimenea ($/nivel)'),
    'GarageQual': (5000, True, 0, 5, 'Calidad garaje ($/nivel)'),
    'GarageCond': (4000, True, 0, 5, 'Condición garaje ($/nivel)'),
    'PoolQC': (8000, True, 0, 5, 'Calidad piscina ($/nivel)'),
}

# Atributos que NO se pueden modificar en remodelación
ATRIBUTOS_FIJOS = [
    'YearBuilt',
    'YearRemodAdd',
    'LotFrontage',
    'LotArea',
    'MSSubClass',
    'MSZoning',
    'Street',
    'Alley',
    'LotShape',
    'LandContour',
    'LotConfig',
    'LandSlope',
    'Neighborhood',
    'Condition1',
    'Condition2',
    'BldgType',
    'HouseStyle',
]


# ============================================================================
# COSTOS DE MATERIALES (Variables categóricas)
# ============================================================================
#Define costos de materiales categóricos (solo puedes elegir UNO)

#VERFICAR COSTOS EN EL PDF
COSTOS_ROOF_STYLE = {
    'Flat': 5000,
    'Gable': 8000,
    'Gambrel': 10000,
    'Hip': 12000,
    'Mansard': 15000,
    'Shed': 6000,
}

# COSTOS_UTILITIES = {         
#     'AllPub': 0,
#     'NoSewr': 15000,
#     'NoSeWa': 20000,
#     'ELO': 25000,
# }

COSTOS_MASVNRTYPE = {         
    'BrkCmn': 8000,
    'BrkFace': 12000,
    'CBlock': 6000,
    'None': 0,
    'Stone': 18000,
}

COSTOS_HEATING = {            
    'Floor': 8000,
    'GasA': 10000,
    'GasW': 9000,
    'Grav': 7000,
    'OthW': 8500,
    'Wall': 7500,
}

COSTOS_ELECTRICAL = {         
    'SBrkr': 5000,    # Standard Circuit Breakers
    'FuseA': 3000,    # Fuse Box > 60 AMP
    'FuseF': 2500,    # 60 AMP Fuse Box
    'FuseP': 2000,    # Poor Fuse Box
    'Mix': 3500,
}

COSTOS_GARAGETYPE = {        
    '2Types': 12000,
    'Attchd': 10000,   # Attached to home
    'Basment': 8000,
    'BuiltIn': 11000,
    'CarPort': 5000,
    'Detchd': 9000,    # Detached
    'NoGarage': 0,
}

COSTOS_PAVEDDRIVE = {         
    'Paved': 3000,
    'PartialPavement': 1500,
    'Dirt/Gravel': 0,
}

COSTOS_MISCFEATURE = {        
    'Elev': 50000,    # Elevador
    'Gar2': 15000,    # Garaje extra
    'Othr': 5000,
    'Shed': 3000,
    'TenC': 10000,    # Cancha de tenis
    'NoMisc': 0,
}

# Roof Material (Material de techo)
COSTOS_ROOF_MATERIAL = {
    'ClyTile': 15000,    # Tejas de arcilla
    'CompShg': 8000,     # Asfalto compuesto
    'Membran': 12000,    # Membrana (techos planos)
    'Metal': 13000,
    'Roll': 5000,
    'Tar&Grv': 6000,
    'WdShake': 14000,
    'WdShngl': 11000,
}

# Matriz de compatibilidad: A[estilo][material] = 1 si compatible

#Define qué combinaciones son técnicamente posibles
ROOF_COMPATIBILITY = {
    'Gable':   {'CompShg': 1, 'Metal': 1, 'ClyTile': 1, 'WdShngl': 1, 'WdShake': 1, 'Membran': 0},
    'Hip':     {'CompShg': 1, 'Metal': 1, 'ClyTile': 1, 'WdShngl': 1, 'WdShake': 1, 'Membran': 0},
    'Flat':    {'CompShg': 0, 'Metal': 1, 'ClyTile': 0, 'WdShngl': 0, 'WdShake': 0, 'Membran': 1},
    'Mansard': {'CompShg': 1, 'Metal': 1, 'ClyTile': 1, 'WdShngl': 1, 'WdShake': 1, 'Membran': 0},
    'Shed':    {'CompShg': 1, 'Metal': 1, 'ClyTile': 0, 'WdShngl': 1, 'WdShake': 0, 'Membran': 1},
    'Gambrel': {'CompShg': 1, 'Metal': 1, 'ClyTile': 1, 'WdShngl': 1, 'WdShake': 1, 'Membran': 0},
}


# ============================================================================
# COSTOS DE DEMOLICIÓN (Para remodelación)
# ============================================================================

#Define cuánto cuesta demoler algo existente (para remodelación)
#COSTO_DEMOLICION_POR_SQFT = 10  # $/sqft para demoler


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

DATA_PATH = '../data/ames_housing_cleaned.csv'
# Ruta donde se persistirá el modelo entrenado
MODEL_PATH = "./models/xgb_model.joblib"
RESULTS_PATH = '../results/'

print("✅ Archivo config.py creado correctamente")