import gurobipy as gp
from gurobipy import GRB
import joblib, pandas as pd

# --- 1) Cargar modelo XGB (pipeline sklearn) ---
try:
    model = joblib.load("models/xgb/Caso_bayesiano_top/model_xgb.joblib")
except Exception as e:
    raise RuntimeError(f"[LOAD MODEL] No pude cargar el modelo: {e}")

# --- 2) DEFAULTS completos (los tuyos) ---
DEFAULTS = {
    "MS SubClass": 20, "MS Zoning": "RL", "Lot Frontage": 141.0, "Lot Area": 31770,
    "Street": "Pave", "Alley": "No aplica", "Lot Shape": "IR1", "Land Contour": "Lvl",
    "Utilities": "AllPub", "Lot Config": "Corner", "Land Slope": "Gtl", "Neighborhood": "No aplicames",
    "Condition 1": "Norm", "Condition 2": "Norm", "Bldg Type": "1Fam", "House Style": "1Story",
    "Overall Qual": 6, "Overall Cond": 5, "Year Built": 1960, "Year Remod/Add": 1960,
    "Roof Style": "Hip", "Roof Matl": "CompShg", "Exterior 1st": "BrkFace", "Exterior 2nd": "Plywood",
    "Mas Vnr Type": "Stone", "Mas Vnr Area": 112.0, "Exter Qual": "TA", "Exter Cond": "TA",
    "Foundation": "CBlock", "Bsmt Qual": "TA", "Bsmt Cond": "Gd", "Bsmt Exposure": "Gd",
    "BsmtFin Type 1": "BLQ", "BsmtFin SF 1": 639.0, "BsmtFin Type 2": "Unf", "BsmtFin SF 2": 0.0,
    "Bsmt Unf SF": 441.0, "Total Bsmt SF": 1080.0, "Heating": "GasA", "Heating QC": "Fa",
    "Central Air": "Y", "Electrical": "SBrkr", "1st Flr SF": 1656, "2nd Flr SF": 0,
    "Low Qual Fin SF": 0, "Gr Liv Area": 1656, "Bsmt Full Bath": 1.0, "Bsmt Half Bath": 0.0,
    "Full Bath": 1, "Half Bath": 0, "Bedroom AbvGr": 3, "Kitchen AbvGr": 1, "Kitchen Qual": "TA",
    "TotRms AbvGrd": 7, "FunctioNo aplical": "Typ", "Fireplaces": 2, "Fireplace Qu": "Gd",
    "Garage Type": "Attchd", "Garage Yr Blt": 1960.0, "Garage Finish": "Fin", "Garage Cars": 2.0,
    "Garage Area": 528.0, "Garage Qual": "TA", "Garage Cond": "TA", "Paved Drive": "P",
    "Wood Deck SF": 210, "Open Porch SF": 62, "Enclosed Porch": 0, "3Ssn Porch": 0,
    "Screen Porch": 0, "Pool Area": 0, "Pool QC": "No aplica", "Fence": "No aplica",
    "Misc Feature": "No aplica", "Misc Val": 0, "Mo Sold": 5, "Yr Sold": 2010,
    "Sale Type": "WD", "Sale Condition": "Normal"
}

# Asegurar que estén TODAS las columnas que el pipeline espera
expected_cols = list(model.feature_names_in_)
for col in expected_cols:
    if col not in DEFAULTS:
        DEFAULTS[col] = 0 if any(x in col for x in ["SF","Area","Bath","Cars","Qual","Yr","Val"]) else "No aplica"

# --- 3) Presupuesto y costos lineales (I) ---
presupuesto = 120000
# Empieza MUY simple; iremos afinando. Cambia estos coeficientes a medida que avances.
COST_COEF = {
    "Gr Liv Area": 40,     # costo por unidad de área habitable (build)
    "Bedroom AbvGr": 8000, # costo por dormitorio
    "Full Bath": 5000,     # costo por baño completo
    # agrega aquí cuando actives más variables...
}

# --- 3E) Costos fijos por material exterior (para Exterior 1st y Exterior 2nd) ---
EXTERIOR_COST = {
    "AsbShng": 19000,
    "AsphShn": 22500,
    "BrkComm": 26000,
    "BrkFace": 22000,
    "CBlock": 22500,   # interpretado de "$22,5"
    "CemntBd": 14674,
    "HdBoard": 4240,   # interpretado de "$4,24"
    "ImStucc": 8500,   # interpretado de "$8,5"
    "MetalSd": 11196,
    "Other":   0,      # sin dato: lo dejamos 0 por ahora
    "Plywood": 3461.81,
    "PreCast": 37500,  # interpretado de "$37,5"
    "Stone":   106250,
    "Stucco":  5629,
    "VinylSd": 17410,
    "WdSdng":  12500,
    "WdShing": 21900,
}


# --- 3U) Costos fijos de Utilities (en dólares) ---
UTILITIES_COST = {
    "AllPub": 31750,
    "NoSewr": 39500,
    "NoSeWa": 22000,
    "ELO":    20000,
}

# --- 3R) Costos fijos de materiales del techo (RoofMatl) ---
ROOFMATL_COST = {
    "ClyTile": 17352,
    "CompShg": 20000,
    "Membran": 8011.95,
    "Metal":   11739,
    "Roll":    7600,
    "Tar&Grv": 8550,
    "WdShake": 22500,
    "WdShngl": 19500,
}

# --- 3MVT) Costos por pie² para Mas Vnr Type ---
# Relacionado con [4C] (MAS_VNR_TYPE_SET) y [4D] (Mas Vnr Area).
# Valores editables (supuestos promedios en $/ft²):
MAS_VNR_UNIT_COST = {
    "BrkCmn": 13.0,    # ladrillo común
    "BrkFace": 15.0,   # ladrillo cara vista
    "CBlock": 22.5,    # bloque hormigón
    "None": 0.0,       # sin revestimiento
    "Stone": 27.5,     # piedra
}

# --- 3FOUND) Costos unitarios por pie² para Foundation ---
FOUNDATION_UNIT_COST = {
    "BrkTil": 0.0,    # sin dato concreto en la tabla
    "CBlock": 12.0,   # bloque cemento
    "PConc": 10.0,    # concreto monolítico
    "Slab": 10.0,     # losa de concreto
    "Stone": 23.5,    # cimiento de piedra
    "Wood": 40.0      # cimentación madera
}

# --- 3HEAT) Costos fijos de Heating (USD) ---
HEATING_COST = {
    "Floor": 1773.0,
    "GasA":  5750.0,
    "GasW":  8500.0,
    "Grav":  6300.0,
    "OthW":  4900.0,
    "Wall":  3700.0,
}

# --- 3-HVAC&E) Costos de HeatingQC (solo guardar), Central Air (sumar), Electrical (conectar) ---

# HeatingQC: guardar para reglas futuras (NO se suman a I aún)
# Existen: Ex, Gd, TA, Fa, Po (valores conocidos en tu tabla: Ex, TA, Po; Gd/Fa se llenarán luego)
HEATING_QC_COST = {
    "Ex": 10000.0,  # alta gama
    "Gd": None,     # interpolado entre Ex y TA (definir más adelante)
    "TA": 6500.0,   # media
    "Fa": None,     # interpolado entre TA y Po (definir más adelante)
    "Po": 3750.0,   # básica
}

# Central Air (costo fijo si = 'Y')
CENTRAL_AIR_COST = 5362.0

# Electrical: conectar a 4F (valores 0 por ahora—cuando los tengas, colócalos aquí)
ELECTRICAL_COST = {
    "SBrkr": 0.0,
    "FuseA": 0.0,
    "FuseF": 0.0,
    "FuseP": 0.0,
    "Mix":   0.0,
}

# --- 3-MISC/PD/BSMT) Costos Misc Feature, Paved Drive y (guardar) Basement ---

# Misc Feature (4K)
MISC_FEATURE_COST = {
    "Elev": 48000.0,   # Elevator
    "Gar2": 32100.0,   # 2nd Garage (si no entra en otra sección)
    "Shed": 5631.0,    # Shed >100 ft² (promedio)
    "TenC": 15774.0,   # Tenis court
    "No aplica": 0.0,  # explícito
}

# Paved Drive (4J)
# Y: 4,908  |  N: 1,800  |  P: promedio simple de ambos
PAVED_DRIVE_COST = {
    "Y": 4908.0,
    "P": (4908.0 + 1800.0) / 2.0,  # 3,354.0
    "N": 1800.0,
}

# Basement acabado (unitario $/ft²) -> se usará MÁS ADELANTE (no se suma a I ahora)
BASEMENT_FINISH_UNIT_COST = 15.0

# --- 3-BSMTQUAL) Costos fijos de 'Bsmt Qual' (USD) ---
# Tabla: Ex=62,500 | TA=41,000 | Po=20,000 | Gd/Fa interpoladas | NA=0
_BQ_BASE = {"Ex": 62500.0, "TA": 41000.0, "Po": 20000.0}
_BQ_GD = (_BQ_BASE["Ex"] + _BQ_BASE["TA"]) / 2.0   # 51,750
_BQ_FA = (_BQ_BASE["TA"] + _BQ_BASE["Po"]) / 2.0   # 30,500

BSMT_QUAL_COST = {
    "Ex": _BQ_BASE["Ex"],
    "Gd": _BQ_GD,
    "TA": _BQ_BASE["TA"],
    "Fa": _BQ_FA,
    "Po": _BQ_BASE["Po"],
    "NA": 0.0,
}
# --- 3-BSMTFIN/KITCHEN/BATH) Costos fijos ---

# BsmtFinType (GLQ, ALQ, BLQ, Rec, LwQ, Unf, NA)
# Datos de tu tabla; ALQ y Rec = interpolaciones:
_BFT_GLQ = 75000.0
_BFT_BLQ = 32000.0
_BFT_LWQ = 15000.0
_BFT_UNF = 11250.0
_BFT_ALQ = (_BFT_GLQ + _BFT_BLQ) / 2.0     # 53,500
_BFT_REC = (_BFT_BLQ + _BFT_LWQ) / 2.0     # 23,500

BSMT_FIN_TYPE_COST = {
    "GLQ": _BFT_GLQ,
    "ALQ": _BFT_ALQ,
    "BLQ": _BFT_BLQ,
    "Rec": _BFT_REC,
    "LwQ": _BFT_LWQ,
    "Unf": _BFT_UNF,
    "NA":  0.0,
}

# KitchenQual (Ex, Gd, TA, Fa, Po)
# Gd y Fa interpoladas tal como indicas.
_KQ_EX = 180000.0
_KQ_TA = 42500.0
_KQ_PO = 13000.0
_KQ_GD = (_KQ_EX + _KQ_TA) / 2.0     # 111,250
_KQ_FA = (_KQ_TA + _KQ_PO) / 2.0     # 27,750

KITCHEN_QUAL_COST = {
    "Ex": _KQ_EX,
    "Gd": _KQ_GD,
    "TA": _KQ_TA,
    "Fa": _KQ_FA,
    "Po": _KQ_PO,
}

# Costo por baño (unidades) — de tus tablas
FULL_BATH_UNIT_COST = 25000.0
HALF_BATH_UNIT_COST = 10000.0

# Kitchen (construcción $/ft²) — GUARDADO para usar más tarde con un área
KITCHEN_UNIT_COST_PER_SF = 200.0

# --- 3-FIRE/GARAGE/POOL/FENCE) Costos ---

# FireplaceQu (calidad de chimenea)
# Ex=4,550 | TA=2,500 | Po=1,500 | Gd y Fa interpoladas | NA=0
_FQ_EX = 4550.0
_FQ_TA = 2500.0
_FQ_PO = 1500.0
_FQ_GD = (_FQ_EX + _FQ_TA) / 2.0   # 3,525
_FQ_FA = (_FQ_TA + _FQ_PO) / 2.0   # 2,000
FIREPLACE_QU_COST = {
    "Ex": _FQ_EX, "Gd": _FQ_GD, "TA": _FQ_TA, "Fa": _FQ_FA, "Po": _FQ_PO, "NA": 0.0
}

# GarageQual / GarageCond (calidad/condición del garaje)
# Ex=51,659 | TA=24,038 | Po=4,188 | Gd / Fa interpoladas | NA=0
_GQ_EX = 51659.0
_GQ_TA = 24038.0
_GQ_PO = 4188.0
_GQ_GD = (_GQ_EX + _GQ_TA) / 2.0   # 37,848.5
_GQ_FA = (_GQ_TA + _GQ_PO) / 2.0   # 14,113.0
GARAGE_QUAL_COST = {"Ex": _GQ_EX, "Gd": _GQ_GD, "TA": _GQ_TA, "Fa": _GQ_FA, "Po": _GQ_PO, "NA": 0.0}
GARAGE_COND_COST = {"Ex": _GQ_EX, "Gd": _GQ_GD, "TA": _GQ_TA, "Fa": _GQ_FA, "Po": _GQ_PO, "NA": 0.0}

# GarageFinish
# Fin=24,038 | Unf=17,500 | RFin interpolado | NA=0
_GF_FIN = 24038.0
_GF_UNF = 17500.0
_GF_RFIN = (_GF_FIN + _GF_UNF) / 2.0   # 20,769
GARAGE_FINISH_COST = {"Fin": _GF_FIN, "RFn": _GF_RFIN, "Unf": _GF_UNF, "NA": 0.0}

# GarageArea ($/ft²)
GARAGE_AREA_UNIT_COST = 47.5

# Pool (PoolQC y PoolArea)
POOL_AREA_UNIT_COST = 88.0
POOL_QC_COST = {
    "Ex": 135000.0,
    "Gd": 96333.0,
    "TA": 57667.0,
    "Fa": 19000.0,
    "NA": 0.0,
}

# Fence (valores de tu tabla; tratamos como costo fijo por categoría)
FENCE_COST = {
    "GdPrv": 42.0,      # privacidad buena
    "MnPrv": 31.33,     # privacidad mínima
    "GdWo": 12.0,       # madera buena
    "MnWw": 2.0,        # alambre económico
    "NA": 0.0,
    # Si quieres un costo base genérico: "Fence": 40.0  # (opcional)
}


# --- 4) Crear modelo Gurobi con pocas variables (activas) ---
m = gp.Model("casa_min")

# Activa de a poco: primero 3 variables. Más tarde agregamos otras.
vars_map = {}
try:
    vars_map["Gr Liv Area"]   = m.addVar(vtype=GRB.INTEGER, lb=600, ub=4000, name="Gr_Liv_Area")
    vars_map["Bedroom AbvGr"] = m.addVar(vtype=GRB.INTEGER, lb=1,   ub=6,    name="Bedroom_AbvGr")
    vars_map["Full Bath"]     = m.addVar(vtype=GRB.INTEGER, lb=1,   ub=3,    name="Full_Bath")
except Exception as e:
    raise RuntimeError(f"[ADD VAR] Error al crear variables: {e}")

# --- 4A) Categóricas: dominios + variables one-hot + suma==1 ---

# Dominios (según tu tabla)
MS_SUBCLASS_SET = [20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190]
UTILITIES_SET   = ["AllPub", "NoSewr", "NoSeWa", "ELO"]
BLDG_TYPE_SET   = ["1Fam", "2FmCon", "Duplx", "TwnhsE", "TwnhsI"]

cat_vars = {}  # clave: (colname, categoria) -> Var binaria

try:
    # MS SubClass (one-hot sobre las clases)
    for s in MS_SUBCLASS_SET:
        cat_vars[("MS SubClass", s)] = m.addVar(vtype=GRB.BINARY, name=f"MSSubClass__{s}")
    m.addConstr(gp.quicksum(cat_vars[("MS SubClass", s)] for s in MS_SUBCLASS_SET) == 1, "onehot_MSSubClass")

    # Utilities
    for u in UTILITIES_SET:
        cat_vars[("Utilities", u)] = m.addVar(vtype=GRB.BINARY, name=f"Utilities__{u}")
    m.addConstr(gp.quicksum(cat_vars[("Utilities", u)] for u in UTILITIES_SET) == 1, "onehot_Utilities")

    # Bldg Type
    for b in BLDG_TYPE_SET:
        cat_vars[("Bldg Type", b)] = m.addVar(vtype=GRB.BINARY, name=f"BldgType__{b}")
    m.addConstr(gp.quicksum(cat_vars[("Bldg Type", b)] for b in BLDG_TYPE_SET) == 1, "onehot_BldgType")
except Exception as e:
    raise RuntimeError(f"[ADD CAT VARS] {e}")

# --- 4B) Categóricas nuevas: House Style, Roof Style, Roof Matl ---

HOUSE_STYLE_SET = ["1Story", "1.5Fin", "1.5Unf", "2Story", "2.5Fin", "2.5Unf", "SFoyer", "SLvl"]
ROOF_STYLE_SET  = ["Flat", "Gable", "Gambrel", "Hip", "Mansard", "Shed"]
ROOF_MATL_SET   = ["ClyTile", "CompShg", "Membran", "Metal", "Roll", "TarGrv", "WdShake", "WdShngl"]

try:
    # House Style
    for hs in HOUSE_STYLE_SET:
        cat_vars[("House Style", hs)] = m.addVar(vtype=GRB.BINARY, name=f"HouseStyle__{hs}")
    m.addConstr(gp.quicksum(cat_vars[("House Style", hs)] for hs in HOUSE_STYLE_SET) == 1, "onehot_HouseStyle")

    # Roof Style
    for rs in ROOF_STYLE_SET:
        cat_vars[("Roof Style", rs)] = m.addVar(vtype=GRB.BINARY, name=f"RoofStyle__{rs}")
    m.addConstr(gp.quicksum(cat_vars[("Roof Style", rs)] for rs in ROOF_STYLE_SET) == 1, "onehot_RoofStyle")

    # Roof Matl
    for rm in ROOF_MATL_SET:
        cat_vars[("Roof Matl", rm)] = m.addVar(vtype=GRB.BINARY, name=f"RoofMatl__{rm}")
    m.addConstr(gp.quicksum(cat_vars[("Roof Matl", rm)] for rm in ROOF_MATL_SET) == 1, "onehot_RoofMatl")

except Exception as e:
    raise RuntimeError(f"[ADD CAT VARS 4B] {e}")

# --- 4C) Categóricas nuevas: Exterior 1st, Exterior 2nd, Mas Vnr Type ---

EXTERIOR_SET = [
    "AsbShng","AsphShn","BrkComm","BrkFace","CBlock","CemntBd","HdBoard","ImStucc",
    "MetalSd","Other","Plywood","PreCast","Stone","Stucco","VinylSd","WdSdng","WdShing"
]
MAS_VNR_TYPE_SET = ["BrkCmn","BrkFace","CBlock","None","Stone"]

try:
    # Exterior 1st
    for e1 in EXTERIOR_SET:
        cat_vars[("Exterior 1st", e1)] = m.addVar(vtype=GRB.BINARY, name=f"Exterior1st__{e1}")
    m.addConstr(gp.quicksum(cat_vars[("Exterior 1st", e1)] for e1 in EXTERIOR_SET) == 1, "onehot_Exterior1st")

    # Exterior 2nd (mismo dominio)
    for e2 in EXTERIOR_SET:
        cat_vars[("Exterior 2nd", e2)] = m.addVar(vtype=GRB.BINARY, name=f"Exterior2nd__{e2}")
    m.addConstr(gp.quicksum(cat_vars[("Exterior 2nd", e2)] for e2 in EXTERIOR_SET) == 1, "onehot_Exterior2nd")

    # Mas Vnr Type
    for t in MAS_VNR_TYPE_SET:
        cat_vars[("Mas Vnr Type", t)] = m.addVar(vtype=GRB.BINARY, name=f"MasVnrType__{t}")
    m.addConstr(gp.quicksum(cat_vars[("Mas Vnr Type", t)] for t in MAS_VNR_TYPE_SET) == 1, "onehot_MasVnrType")
except Exception as e:
    raise RuntimeError(f"[ADD CAT VARS 4C] {e}")

# --- 4D) Naturaleza nuevas: Mas Vnr Area, Foundation, Bsmt Exposure, BsmtFin Type 1 ---

FOUNDATION_SET    = ["BrkTil", "CBlock", "PConc", "Slab", "Stone", "Wood"]
BSMT_EXPOSURE_SET = ["Gd", "Av", "Mn", "No", "NA"]
BSMT_FIN_TYPE1_SET = ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"]

try:
    # Mas Vnr Area (entera, >= 0)
    vars_map["Mas Vnr Area"] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=1500, name="Mas_Vnr_Area")

    # Foundation
    for f in FOUNDATION_SET:
        cat_vars[("Foundation", f)] = m.addVar(vtype=GRB.BINARY, name=f"Foundation__{f}")
    m.addConstr(gp.quicksum(cat_vars[("Foundation", f)] for f in FOUNDATION_SET) == 1, "onehot_Foundation")

    # Bsmt Exposure
    for x in BSMT_EXPOSURE_SET:
        cat_vars[("Bsmt Exposure", x)] = m.addVar(vtype=GRB.BINARY, name=f"BsmtExposure__{x}")
    m.addConstr(gp.quicksum(cat_vars[("Bsmt Exposure", x)] for x in BSMT_EXPOSURE_SET) == 1, "onehot_BsmtExposure")

    # BsmtFin Type 1
    for b1 in BSMT_FIN_TYPE1_SET:
        cat_vars[("BsmtFin Type 1", b1)] = m.addVar(vtype=GRB.BINARY, name=f"BsmtFinType1__{b1}")
    m.addConstr(gp.quicksum(cat_vars[("BsmtFin Type 1", b1)] for b1 in BSMT_FIN_TYPE1_SET) == 1, "onehot_BsmtFinType1")

except Exception as e:
    raise RuntimeError(f"[ADD VARS 4D] {e}")

# --- 4E) Naturaleza: BsmtFin SF 1, BsmtFin Type 2, BsmtFin SF 2, Bsmt Unf SF ---

BSMT_FIN_TYPE2_SET = ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"]

try:
    # Numéricas (enteras >= 0). Sin cota superior artificial.
    vars_map["BsmtFin SF 1"] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=GRB.INFINITY, name="BsmtFin_SF_1")
    vars_map["BsmtFin SF 2"] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=GRB.INFINITY, name="BsmtFin_SF_2")
    vars_map["Bsmt Unf SF"]  = m.addVar(vtype=GRB.INTEGER, lb=0, ub=GRB.INFINITY, name="Bsmt_Unf_SF")

    # Categórica: BsmtFin Type 2 (one-hot puro, suma==1)
    for b2 in BSMT_FIN_TYPE2_SET:
        cat_vars[("BsmtFin Type 2", b2)] = m.addVar(vtype=GRB.BINARY, name=f"BsmtFinType2__{b2}")
    m.addConstr(gp.quicksum(cat_vars[("BsmtFin Type 2", b2)] for b2 in BSMT_FIN_TYPE2_SET) == 1, "onehot_BsmtFinType2")
except Exception as e:
    raise RuntimeError(f"[ADD VARS 4E] {e}")

# --- 4F) Naturaleza: Total Bsmt SF (int), Heating (cat), Central Air (cat), Electrical (cat) ---

HEATING_SET   = ["Floor", "GasA", "GasW", "Grav", "OthW", "Wall"]
CENTRAL_AIR_SET = ["Y", "N"]  # en Ames: "Y"/"N" (no "Yes"/"No")
ELECTRICAL_SET  = ["SBrkr", "FuseA", "FuseF", "FuseP", "Mix"]

try:
    # Total Bsmt SF (entera >= 0)
    vars_map["Total Bsmt SF"] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=GRB.INFINITY, name="Total_Bsmt_SF")

    # Heating (one-hot)
    for h in HEATING_SET:
        cat_vars[("Heating", h)] = m.addVar(vtype=GRB.BINARY, name=f"Heating__{h}")
    m.addConstr(gp.quicksum(cat_vars[("Heating", h)] for h in HEATING_SET) == 1, "onehot_Heating")

    # Central Air (one-hot)
    for a in CENTRAL_AIR_SET:
        cat_vars[("Central Air", a)] = m.addVar(vtype=GRB.BINARY, name=f"CentralAir__{a}")
    m.addConstr(gp.quicksum(cat_vars[("Central Air", a)] for a in CENTRAL_AIR_SET) == 1, "onehot_CentralAir")

    # Electrical (one-hot)
    for e in ELECTRICAL_SET:
        cat_vars[("Electrical", e)] = m.addVar(vtype=GRB.BINARY, name=f"Electrical__{e}")
    m.addConstr(gp.quicksum(cat_vars[("Electrical", e)] for e in ELECTRICAL_SET) == 1, "onehot_Electrical")

except Exception as e:
    raise RuntimeError(f"[ADD VARS 4F] {e}")

# --- 4G) Naturaleza: 1st Flr SF, 2nd Flr SF, Low Qual Fin SF, Gr Liv Area ---

try:
    vars_map["1st Flr SF"]      = m.addVar(vtype=GRB.INTEGER, lb=0, ub=GRB.INFINITY, name="First_Flr_SF")
    vars_map["2nd Flr SF"]      = m.addVar(vtype=GRB.INTEGER, lb=0, ub=GRB.INFINITY, name="Second_Flr_SF")
    vars_map["Low Qual Fin SF"] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=GRB.INFINITY, name="Low_Qual_Fin_SF")
    vars_map["Gr Liv Area"]     = m.addVar(vtype=GRB.INTEGER, lb=0, ub=GRB.INFINITY, name="Gr_Liv_Area_Full")
except Exception as e:
    raise RuntimeError(f"[ADD VARS 4G] {e}")

# --- 4H) Naturaleza: baños completos y medios (sótano y sobre nivel) ---

try:
    vars_map["Bsmt Full Bath"] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=GRB.INFINITY, name="Bsmt_Full_Bath")
    vars_map["Bsmt Half Bath"] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=GRB.INFINITY, name="Bsmt_Half_Bath")
    vars_map["Full Bath"]      = m.addVar(vtype=GRB.INTEGER, lb=0, ub=GRB.INFINITY, name="Full_Bath")
    vars_map["Half Bath"]      = m.addVar(vtype=GRB.INTEGER, lb=0, ub=GRB.INFINITY, name="Half_Bath")
except Exception as e:
    raise RuntimeError(f"[ADD VARS 4H] {e}")

# --- 4I) Naturaleza: dormitorios/cocinas/rooms/fireplaces + garage ---

GARAGE_TYPE_SET   = ["2Types", "Attchd", "Basment", "BuiltIn", "CarPort", "Detchd", "No aplica"]
GARAGE_FINISH_SET = ["Fin", "RFn", "Unf", "No aplica"]

try:
    # Enteras no negativas
    vars_map["Bedroom AbvGr"] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=GRB.INFINITY, name="Bedroom_AbvGr_full")
    vars_map["Kitchen AbvGr"] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=GRB.INFINITY, name="Kitchen_AbvGr_full")
    vars_map["TotRms AbvGrd"] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=GRB.INFINITY, name="TotRms_AbvGrd_full")
    vars_map["Fireplaces"]    = m.addVar(vtype=GRB.INTEGER, lb=0, ub=GRB.INFINITY, name="Fireplaces_full")

    # Garage Type (one-hot)
    for g in GARAGE_TYPE_SET:
        cat_vars[("Garage Type", g)] = m.addVar(vtype=GRB.BINARY, name=f"GarageType__{g}")
    m.addConstr(gp.quicksum(cat_vars[("Garage Type", g)] for g in GARAGE_TYPE_SET) == 1, "onehot_GarageType")

    # Garage Finish (one-hot)
    for gf in GARAGE_FINISH_SET:
        cat_vars[("Garage Finish", gf)] = m.addVar(vtype=GRB.BINARY, name=f"GarageFinish__{gf}")
    m.addConstr(gp.quicksum(cat_vars[("Garage Finish", gf)] for gf in GARAGE_FINISH_SET) == 1, "onehot_GarageFinish")

    # Garage Cars (entera >= 0)
    vars_map["Garage Cars"] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=GRB.INFINITY, name="Garage_Cars_full")
except Exception as e:
    raise RuntimeError(f"[ADD VARS 4I] {e}")

# --- 4J) Naturaleza: Garage Area, Paved Drive (cat), Wood Deck SF, Open Porch SF ---

# OJO: usamos los códigos del dataset Ames / tu XGB (tu DEFAULTS trae "P")
PAVED_DRIVE_SET = ["Y", "P", "N"]  # Yes / Partial / No

try:
    # Enteras no negativas
    vars_map["Garage Area"]  = m.addVar(vtype=GRB.INTEGER, lb=0, ub=GRB.INFINITY, name="Garage_Area_full")
    vars_map["Wood Deck SF"] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=GRB.INFINITY, name="Wood_Deck_SF_full")
    vars_map["Open Porch SF"]= m.addVar(vtype=GRB.INTEGER, lb=0, ub=GRB.INFINITY, name="Open_Porch_SF_full")

    # Paved Drive (one-hot)
    for p in PAVED_DRIVE_SET:
        cat_vars[("Paved Drive", p)] = m.addVar(vtype=GRB.BINARY, name=f"PavedDrive__{p}")
    m.addConstr(gp.quicksum(cat_vars[("Paved Drive", p)] for p in PAVED_DRIVE_SET) == 1, "onehot_PavedDrive")
except Exception as e:
    raise RuntimeError(f"[ADD VARS 4J] {e}")

# --- 4K) Naturaleza: porches, piscina, misc, sale type, sale condition ---

MISC_FEATURE_SET = ["Elev", "Gar2", "Othr", "Shed", "TenC", "No aplica"]
SALE_TYPE_SET = ["WD", "CWD", "VWD", "New", "COD", "Con", "ConLw", "ConLI", "ConLD", "Oth"]
SALE_CONDITION_SET = ["Normal", "Abnorml", "AdjLand", "Alloca", "Family", "Partial"]

try:
    # Enteras no negativas
    vars_map["Enclosed Porch"] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=GRB.INFINITY, name="Enclosed_Porch_full")
    vars_map["3Ssn Porch"]     = m.addVar(vtype=GRB.INTEGER, lb=0, ub=GRB.INFINITY, name="3Ssn_Porch_full")
    vars_map["Screen Porch"]   = m.addVar(vtype=GRB.INTEGER, lb=0, ub=GRB.INFINITY, name="Screen_Porch_full")
    vars_map["Pool Area"]      = m.addVar(vtype=GRB.INTEGER, lb=0, ub=GRB.INFINITY, name="Pool_Area_full")
    vars_map["HasPool"] = m.addVar(vtype=GRB.BINARY, name="HasPool")
    # One-hot categóricas
    for misc in MISC_FEATURE_SET:
        cat_vars[("Misc Feature", misc)] = m.addVar(vtype=GRB.BINARY, name=f"MiscFeature__{misc}")
    m.addConstr(gp.quicksum(cat_vars[("Misc Feature", misc)] for misc in MISC_FEATURE_SET) == 1, "onehot_MiscFeature")

    for st in SALE_TYPE_SET:
        cat_vars[("Sale Type", st)] = m.addVar(vtype=GRB.BINARY, name=f"SaleType__{st}")
    m.addConstr(gp.quicksum(cat_vars[("Sale Type", st)] for st in SALE_TYPE_SET) == 1, "onehot_SaleType")

    for sc in SALE_CONDITION_SET:
        cat_vars[("Sale Condition", sc)] = m.addVar(vtype=GRB.BINARY, name=f"SaleCondition__{sc}")
    m.addConstr(gp.quicksum(cat_vars[("Sale Condition", sc)] for sc in SALE_CONDITION_SET) == 1, "onehot_SaleCondition")

except Exception as e:
    raise RuntimeError(f"[ADD VARS 4K] {e}")


# --- 5) Costo total I como suma lineal (usa solo variables activas) ---
try:
    I = gp.LinExpr(0.0)
    for k, v in vars_map.items():
        coef = COST_COEF.get(k, 0.0)
        I += coef * v
    # Si quieres costos fijos por categorías (p.ej. Central Air = Y), agrégalos aparte:
    # I += 0 if DEFAULTS["Central Air"] == "N" else 0  # ejemplo; afinaremos luego
except Exception as e:
    raise RuntimeError(f"[COST] Error construyendo costo: {e}")

# --- 5A) (Opcional) costos por categoría (valen 0 por defecto; cámbialos cuando tengas datos) ---
try:
    # Ejemplos de cómo sumar costos fijos por elegir ciertas categorías:
    # I += 0 * cat_vars[("Utilities", "NoSewr")]
    # I += 0 * cat_vars[("Utilities", "NoSeWa")]
    # I += 0 * cat_vars[("Utilities", "ELO")]
    # I += 0 * cat_vars[("Bldg Type", "Duplx")]
    # I += 0 * cat_vars[("Bldg Type", "TwnhsE")]
    pass
except Exception as e:
    raise RuntimeError(f"[COST CATS] Error sumando costos categóricos: {e}")

# --- 5E) Añadir costos fijos de Exterior 1st y Exterior 2nd a I ---
try:
    # Usa las mismas categorías de [4C] (EXTERIOR_SET) y las binarias cat_vars
    for e in EXTERIOR_SET:
        c = EXTERIOR_COST.get(e, 0)
        if ("Exterior 1st", e) in cat_vars:
            I += c * cat_vars[("Exterior 1st", e)]
        if ("Exterior 2nd", e) in cat_vars:
            I += c * cat_vars[("Exterior 2nd", e)]
except Exception as e:
    raise RuntimeError(f"[COST EXTERIOR] {e}")

# --- 5U) Añadir costos fijos de Utilities a I ---
try:
    # Sanity check: que las llaves coincidan con lo creado en [4A]
    for u in UTILITIES_COST.keys():
        if ("Utilities", u) not in cat_vars:
            raise KeyError(f"Utilities '{u}' no existe en cat_vars. Revisa [4A].")
    # Sumar costo fijo de la opción elegida (one-hot)
    for u, c in UTILITIES_COST.items():
        I += c * cat_vars[("Utilities", u)]
except Exception as e:
    raise RuntimeError(f"[COST UTILITIES] {e}")

# --- 5R) Añadir costos fijos de Roof Matl a I ---
try:
    # Diccionario de costos (asegúrate de haberlo definido en la sección 3)
    ROOFMATL_COST = {
        "ClyTile": 17352,
        "CompShg": 20000,
        "Membran": 8011.95,
        "Metal":   11739,
        "Roll":    7600,
        # Soportar ambas etiquetas por si tu [4B] usa "TarGrv" (sin &)
        "Tar&Grv": 8550,
        "TarGrv":  8550,
        "WdShake": 22500,
        "WdShngl": 19500,
    }

    # Sumar el costo fijo de la categoría elegida (one-hot en [4B])
    for r, c in ROOFMATL_COST.items():
        if ("Roof Matl", r) in cat_vars:
            I += c * cat_vars[("Roof Matl", r)]
except Exception as e:
    raise RuntimeError(f"[COST ROOF MATL] {e}")

# --- 5MVT) Costo = (costo unitario) * (Mas Vnr Area) * (one-hot del tipo) ---
try:
    # Sanity checks mínimos
    if "Mas Vnr Area" not in vars_map:
        raise KeyError("Falta variable 'Mas Vnr Area' de [4D].")
    for t in MAS_VNR_TYPE_SET:  # viene de [4C]
        if ("Mas Vnr Type", t) not in cat_vars:
            raise KeyError(f"Falta one-hot de Mas Vnr Type '{t}' de [4C].")

    area = vars_map["Mas Vnr Area"]
    for t, cu in MAS_VNR_UNIT_COST.items():
        if ("Mas Vnr Type", t) in cat_vars:
            I += cu * area * cat_vars[("Mas Vnr Type", t)]
except Exception as e:
    raise RuntimeError(f"[COST MAS_VNR_TYPE] {e}")

# --- 5FOUND) Costo Foundation = costo unitario * Total Bsmt SF * one-hot del tipo ---
try:
    if "Total Bsmt SF" not in vars_map:
        raise KeyError("Falta variable 'Total Bsmt SF' (área base) de [4D].")
    
    area_bsm = vars_map["Total Bsmt SF"]
    for f, c in FOUNDATION_UNIT_COST.items():
        if ("Foundation", f) in cat_vars:
            I += c * area_bsm * cat_vars[("Foundation", f)]
except Exception as e:
    raise RuntimeError(f"[COST FOUNDATION] {e}")

# --- 5HEAT) Añadir costo fijo de Heating a I ---
try:
    for h, c in HEATING_COST.items():
        if ("Heating", h) in cat_vars:  # one-hot de [4F]
            I += c * cat_vars[("Heating", h)]
except Exception as e:
    raise RuntimeError(f"[COST HEATING] {e}")

# --- 5-CA&EL) Añadir costos de Central Air y Electrical a I ---
try:
    # Central Air (one-hot de [4F]: 'Y' / 'N')
    if ("Central Air", "Y") in cat_vars:
        I += CENTRAL_AIR_COST * cat_vars[("Central Air", "Y")]
    # 'N' no suma costo (implícito)

    # Electrical (one-hot de [4F]); actualmente todos 0.0 (listo para cuando tengas montos)
    for e, c in ELECTRICAL_COST.items():
        if ("Electrical", e) in cat_vars:
            I += c * cat_vars[("Electrical", e)]
except Exception as e:
    raise RuntimeError(f"[COST CA/EL] {e}")

# --- 5-MISC/PD) Añadir costos de Misc Feature y Paved Drive a I ---
try:
    # Misc Feature (usa one-hot de 4K)
    if 'MISC_FEATURE_SET' in globals():
        for misc, c in MISC_FEATURE_COST.items():
            key = ("Misc Feature", misc)
            if key in cat_vars:
                I += c * cat_vars[key]
    else:
        # Si no tienes la lista global, itera por el dict de costos
        for misc, c in MISC_FEATURE_COST.items():
            key = ("Misc Feature", misc)
            if key in cat_vars:
                I += c * cat_vars[key]

    # Paved Drive (usa one-hot de 4J)
    for p, c in PAVED_DRIVE_COST.items():
        key = ("Paved Drive", p)
        if key in cat_vars:
            I += c * cat_vars[key]

    # Basement acabado: NO se suma aquí. Se usará más adelante con un término:
    # I += BASEMENT_FINISH_UNIT_COST * (área acabada que definas) * (binaria/condición)
except Exception as e:
    raise RuntimeError(f"[COST MISC/PAVED] {e}")

# --- 5-BSMTQUAL) Añadir costo fijo de 'Bsmt Qual' a I ---
try:
    # Usa las one-hot ya creadas para 'Bsmt Qual' (si aún no existen, simplemente no suma)
    for q, c in BSMT_QUAL_COST.items():
        key = ("Bsmt Qual", q)
        if key in cat_vars:
            I += c * cat_vars[key]
except Exception as e:
    raise RuntimeError(f"[COST BSMT QUAL] {e}")

# --- 5-BSMTFIN/KITCHEN/BATH) Sumar a I ---

try:
    # BsmtFin Type 1 / 2 (usa las one-hot si ya las declaraste en [4D])
    # Nombres EXACTOS: "BsmtFin Type 1" y "BsmtFin Type 2"
    if any(key[0] == "BsmtFin Type 1" for key in cat_vars.keys()):
        for t, c in BSMT_FIN_TYPE_COST.items():
            key1 = ("BsmtFin Type 1", t)
            if key1 in cat_vars:
                I += c * cat_vars[key1]
    if any(key[0] == "BsmtFin Type 2" for key in cat_vars.keys()):
        for t, c in BSMT_FIN_TYPE_COST.items():
            key2 = ("BsmtFin Type 2", t)
            if key2 in cat_vars:
                I += c * cat_vars[key2]

    # KitchenQual × Kitchen AbvGr (si existen ambas variables)
    if "Kitchen AbvGr" in vars_map:
        k_count = vars_map["Kitchen AbvGr"]
        # preferimos usar one-hot si existe; si no, no sumamos (evitamos doble conteo futuro)
        if any(key[0] == "Kitchen Qual" for key in cat_vars.keys()):
            for q, c in KITCHEN_QUAL_COST.items():
                keyq = ("Kitchen Qual", q)
                if keyq in cat_vars:
                    I += c * k_count * cat_vars[keyq]
        # si aún no tienes one-hot de Kitchen Qual, déjalo sin sumar por ahora
        # (cuando declares la naturaleza, esto empezará a sumar automáticamente)

    # Baños (enteras ya definidas en [4H]): Full Bath y Half Bath
    if "Full Bath" in vars_map:
        I += FULL_BATH_UNIT_COST * vars_map["Full Bath"]
    if "Half Bath" in vars_map:
        I += HALF_BATH_UNIT_COST * vars_map["Half Bath"]

    # Kitchen $/ft² guardado — NO se suma aquí (falta definir área de cocina)
except Exception as e:
    raise RuntimeError(f"[COST BSMTFIN/KITCHEN/BATH] {e}")

# --- 5-FIRE/GARAGE/POOL/FENCE) Sumar a I ---

try:
    # Fireplace Qu
    for q, c in FIREPLACE_QU_COST.items():
        key = ("Fireplace Qu", q)
        if key in cat_vars:
            I += c * cat_vars[key]

    # Garage Qual
    for q, c in GARAGE_QUAL_COST.items():
        key = ("Garage Qual", q)
        if key in cat_vars:
            I += c * cat_vars[key]

    # Garage Cond
    for q, c in GARAGE_COND_COST.items():
        key = ("Garage Cond", q)
        if key in cat_vars:
            I += c * cat_vars[key]

    # Garage Finish
    for f, c in GARAGE_FINISH_COST.items():
        key = ("Garage Finish", f)
        if key in cat_vars:
            I += c * cat_vars[key]

    # Garage Area (unitario)
    if "Garage Area" in vars_map:
        I += GARAGE_AREA_UNIT_COST * vars_map["Garage Area"]

    # Pool QC (fijo por calidad)
    for q, c in POOL_QC_COST.items():
        key = ("Pool QC", q)
        if key in cat_vars:
            I += c * cat_vars[key]

    # Pool Area (unitario)
    if "Pool Area" in vars_map:
        I += POOL_AREA_UNIT_COST * vars_map["Pool Area"]

    # Fence (fijo por categoría)
    for f, c in FENCE_COST.items():
        key = ("Fence", f)
        if key in cat_vars:
            I += c * cat_vars[key]

except Exception as e:
    raise RuntimeError(f"[COST FIRE/GARAGE/POOL/FENCE] {e}")


# --- 6) Restricción de presupuesto (única al principio) ---
try:
    m.addConstr(I <= presupuesto, name="presupuesto")
except Exception as e:
    raise RuntimeError(f"[CONSTR] Error en restricción de presupuesto: {e}")

# === Restricciones de Área y Dimensiones ===
# Eq. (5): 1stFlrSF + TotalPorchSF + AreaPool <= LotArea

TotalPorchSF = (
    vars_map.get("Open Porch SF", 0)
    + vars_map.get("Enclosed Porch", 0)
    + vars_map.get("3Ssn Porch", 0)
    + vars_map.get("Screen Porch", 0)
    + vars_map.get("Wood Deck SF", 0)
)

if all(k in vars_map for k in ["1st Flr SF", "Lot Area"]):
    m.addConstr(
        vars_map["1st Flr SF"] + TotalPorchSF + vars_map.get("Pool Area", 0) <= vars_map["Lot Area"],
        name="restriccion_area_lote"
    )

# Eq. (6): 2ndFlrSF <= 1stFlrSF
if all(k in vars_map for k in ["2nd Flr SF", "1st Flr SF"]):
    m.addConstr(
        vars_map["2nd Flr SF"] <= vars_map["1st Flr SF"],
        name="restriccion_segundo_piso"
    )
# ===========================================


# --- 7) Objetivo Gurobi (placeholder). Manténlo 0 para factibilidad. ---
m.setObjective(0.0, GRB.MAXIMIZE)

# --- 8) Optimiza y evalúa ρ = V - I con XGB afuera ---
try:
    m.optimize()
except Exception as e:
    raise RuntimeError(f"[OPT] Error al optimizar: {e}")

# Construir la fila completa para XGB con los valores optimizados (lo demás = DEFAULTS)
row = DEFAULTS.copy()
# --- 8A) Escribir categóricas elegidas (argmax de la one-hot) ---
try:
    # MS SubClass (es numérica en tus datos, pero aquí la escogemos por one-hot)
    ms_sel = max(MS_SUBCLASS_SET, key=lambda s: cat_vars[("MS SubClass", s)].X)
    row["MS SubClass"] = ms_sel

    # Utilities
    util_sel = max(UTILITIES_SET, key=lambda u: cat_vars[("Utilities", u)].X)
    row["Utilities"] = util_sel

    # Bldg Type
    bldg_sel = max(BLDG_TYPE_SET, key=lambda b: cat_vars[("Bldg Type", b)].X)
    row["Bldg Type"] = bldg_sel
except Exception as e:
    raise RuntimeError(f"[WRITE CATS TO ROW] {e}")

for name, var in vars_map.items():
    # seguridad por si var no tiene solución
    try:
        row[name] = int(round(var.X)) if var.VType == GRB.INTEGER else float(var.X)
    except Exception:
        row[name] = DEFAULTS[name]

# Predicción V y cálculo de I y ρ
try:
    df = pd.DataFrame([row])[expected_cols]
    V = float(model.predict(df)[0])
except Exception as e:
    raise RuntimeError(f"[PREDICT] Error al predecir XGB: {e}")




# Recalcular I en Python con los mismos coeficientes (para imprimir)
I_val = 0.0
for k, coef in COST_COEF.items():
    I_val += coef * row[k]

rho = V - I_val

print("\n=== RESULTADO ===")
print(f"V (XGB): {round(V,2)}")
print(f"I (costo): {round(I_val,2)}  | Presupuesto: {presupuesto}")
print(f"ρ = V - I : {round(rho,2)}")
print("Decisión:")
for k in vars_map.keys():
    print(f" - {k}: {row[k]}")
# --- PRINT EXTRA: categóricas elegidas y desglose de costos ---
try:
    # Categóricas elegidas por one-hot
    ms_sel   = max(MS_SUBCLASS_SET, key=lambda s: cat_vars[("MS SubClass", s)].X)
    util_sel = max(UTILITIES_SET,   key=lambda u: cat_vars[("Utilities", u)].X)
    bldg_sel = max(BLDG_TYPE_SET,   key=lambda b: cat_vars[("Bldg Type", b)].X)

    print(" - MS SubClass:", ms_sel)
    print(" - Utilities  :", util_sel)
    print(" - Bldg Type  :", bldg_sel)
except Exception as e:
    print("[INFO] No pude leer categóricas:", e)

# --- 8B) Mostrar categóricas 4B elegidas ---
try:
    hs_sel = max(HOUSE_STYLE_SET, key=lambda hs: cat_vars[("House Style", hs)].X)
    rs_sel = max(ROOF_STYLE_SET,  key=lambda rs: cat_vars[("Roof Style", rs)].X)
    rm_sel = max(ROOF_MATL_SET,   key=lambda rm: cat_vars[("Roof Matl", rm)].X)

    print(" - House Style:", hs_sel)
    print(" - Roof Style :", rs_sel)
    print(" - Roof Matl  :", rm_sel)
except Exception as e:
    print("[INFO] No pude leer House/Roof:", e)

# --- 8C) Volcar y mostrar las nuevas categóricas 4C ---
try:
    ext1_sel = max(EXTERIOR_SET,      key=lambda e: cat_vars[("Exterior 1st", e)].X)
    ext2_sel = max(EXTERIOR_SET,      key=lambda e: cat_vars[("Exterior 2nd", e)].X)
    mvt_sel  = max(MAS_VNR_TYPE_SET,  key=lambda t: cat_vars[("Mas Vnr Type", t)].X)

    row["Exterior 1st"] = ext1_sel
    row["Exterior 2nd"] = ext2_sel
    row["Mas Vnr Type"] = mvt_sel

    print(" - Exterior 1st:", ext1_sel)
    print(" - Exterior 2nd:", ext2_sel)
    print(" - Mas Vnr Type:", mvt_sel)
except Exception as e:
    print("[INFO] No pude leer Exterior/MasVnr:", e)

# --- 8D) Volcar y mostrar las nuevas variables 4D ---
try:
    # Categóricas
    f_sel  = max(FOUNDATION_SET,    key=lambda f: cat_vars[("Foundation", f)].X)
    x_sel  = max(BSMT_EXPOSURE_SET, key=lambda x: cat_vars[("Bsmt Exposure", x)].X)
    b1_sel = max(BSMT_FIN_TYPE1_SET,key=lambda b1: cat_vars[("BsmtFin Type 1", b1)].X)

    row["Foundation"]    = f_sel
    row["Bsmt Exposure"] = x_sel
    row["BsmtFin Type 1"] = b1_sel
    row["Mas Vnr Area"]  = int(round(vars_map["Mas Vnr Area"].X))

    print(" - Foundation   :", f_sel)
    print(" - Bsmt Exposure:", x_sel)
    print(" - BsmtFin Type1:", b1_sel)
    print(" - Mas Vnr Area :", row['Mas Vnr Area'])
except Exception as e:
    print("[INFO] No pude leer Foundation/Bsmt vars:", e)

# --- 8E) Volcar y mostrar variables de 4E ---
try:
    row["BsmtFin SF 1"] = int(round(vars_map["BsmtFin SF 1"].X))
    row["BsmtFin SF 2"] = int(round(vars_map["BsmtFin SF 2"].X))
    row["Bsmt Unf SF"]  = int(round(vars_map["Bsmt Unf SF"].X))

    b2_sel = max(BSMT_FIN_TYPE2_SET, key=lambda b2: cat_vars[("BsmtFin Type 2", b2)].X)
    row["BsmtFin Type 2"] = b2_sel

    print(" - BsmtFin SF 1 :", row["BsmtFin SF 1"])
    print(" - BsmtFin Type2:", b2_sel)
    print(" - BsmtFin SF 2 :", row["BsmtFin SF 2"])
    print(" - Bsmt Unf SF  :", row["Bsmt Unf SF"])
except Exception as e:
    print("[INFO] No pude volcar/mostrar Bsmt vars (4E):", e)

# --- 8F) Volcar y mostrar 4F ---
try:
    # numérica
    row["Total Bsmt SF"] = int(round(vars_map["Total Bsmt SF"].X))

    # categóricas por argmax
    heat_sel = max(HEATING_SET,      key=lambda h: cat_vars[("Heating", h)].X)
    ac_sel   = max(CENTRAL_AIR_SET,  key=lambda a: cat_vars[("Central Air", a)].X)
    elec_sel = max(ELECTRICAL_SET,   key=lambda e: cat_vars[("Electrical", e)].X)

    row["Heating"]     = heat_sel
    row["Central Air"] = ac_sel
    row["Electrical"]  = elec_sel

    print(" - Total Bsmt SF:", row["Total Bsmt SF"])
    print(" - Heating      :", heat_sel)
    print(" - Central Air  :", ac_sel)
    print(" - Electrical   :", elec_sel)
except Exception as e:
    print("[INFO] No pude volcar/mostrar 4F]:", e)

# --- 8G) Volcar y mostrar 4G ---
try:
    row["1st Flr SF"]      = int(round(vars_map["1st Flr SF"].X))
    row["2nd Flr SF"]      = int(round(vars_map["2nd Flr SF"].X))
    row["Low Qual Fin SF"] = int(round(vars_map["Low Qual Fin SF"].X))
    row["Gr Liv Area"]     = int(round(vars_map["Gr Liv Area"].X))

    print(" - 1st Flr SF     :", row["1st Flr SF"])
    print(" - 2nd Flr SF     :", row["2nd Flr SF"])
    print(" - Low Qual Fin SF:", row["Low Qual Fin SF"])
    print(" - Gr Liv Area    :", row["Gr Liv Area"])
except Exception as e:
    print("[INFO] No pude volcar/mostrar 4G]:", e)

# --- 8H) Volcar y mostrar 4H ---
try:
    row["Bsmt Full Bath"] = int(round(vars_map["Bsmt Full Bath"].X))
    row["Bsmt Half Bath"] = int(round(vars_map["Bsmt Half Bath"].X))
    row["Full Bath"]      = int(round(vars_map["Full Bath"].X))
    row["Half Bath"]      = int(round(vars_map["Half Bath"].X))

    print(" - Bsmt Full Bath:", row["Bsmt Full Bath"])
    print(" - Bsmt Half Bath:", row["Bsmt Half Bath"])
    print(" - Full Bath     :", row["Full Bath"])
    print(" - Half Bath     :", row["Half Bath"])
except Exception as e:
    print("[INFO] No pude volcar/mostrar 4H]:", e)

# --- 8I) Volcar y mostrar 4I ---
try:
    # Numéricas
    row["Bedroom AbvGr"] = int(round(vars_map["Bedroom AbvGr"].X))
    row["Kitchen AbvGr"] = int(round(vars_map["Kitchen AbvGr"].X))
    row["TotRms AbvGrd"] = int(round(vars_map["TotRms AbvGrd"].X))
    row["Fireplaces"]    = int(round(vars_map["Fireplaces"].X))
    row["Garage Cars"]   = int(round(vars_map["Garage Cars"].X))

    # Categóricas por argmax
    gtype_sel   = max(GARAGE_TYPE_SET,   key=lambda g:  cat_vars[("Garage Type", g)].X)
    gfinish_sel = max(GARAGE_FINISH_SET, key=lambda gf: cat_vars[("Garage Finish", gf)].X)

    row["Garage Type"]   = gtype_sel
    row["Garage Finish"] = gfinish_sel

    print(" - Bedroom AbvGr :", row['Bedroom AbvGr'])
    print(" - Kitchen AbvGr :", row['Kitchen AbvGr'])
    print(" - TotRms AbvGrd :", row['TotRms AbvGrd'])
    print(" - Fireplaces    :", row['Fireplaces'])
    print(" - Garage Type   :", gtype_sel)
    print(" - Garage Finish :", gfinish_sel)
    print(" - Garage Cars   :", row['Garage Cars'])
except Exception as e:
    print("[INFO] No pude volcar/mostrar 4I]:", e)

# --- 8J) Volcar y mostrar 4J ---
try:
    row["Garage Area"]   = int(round(vars_map["Garage Area"].X))
    row["Wood Deck SF"]  = int(round(vars_map["Wood Deck SF"].X))
    row["Open Porch SF"] = int(round(vars_map["Open Porch SF"].X))

    pd_sel = max(PAVED_DRIVE_SET, key=lambda p: cat_vars[("Paved Drive", p)].X)
    row["Paved Drive"] = pd_sel

    print(" - Garage Area  :", row["Garage Area"])
    print(" - Paved Drive  :", pd_sel)
    print(" - Wood Deck SF :", row["Wood Deck SF"])
    print(" - Open Porch SF:", row["Open Porch SF"])
except Exception as e:
    print("[INFO] No pude volcar/mostrar 4J]:", e)

# --- 8K) Volcar y mostrar 4K ---
try:
    row["Enclosed Porch"] = int(round(vars_map["Enclosed Porch"].X))
    row["3Ssn Porch"]     = int(round(vars_map["3Ssn Porch"].X))
    row["Screen Porch"]   = int(round(vars_map["Screen Porch"].X))
    row["Pool Area"]      = int(round(vars_map["Pool Area"].X))

    misc_sel  = max(MISC_FEATURE_SET,     key=lambda m: cat_vars[("Misc Feature", m)].X)
    sale_sel  = max(SALE_TYPE_SET,        key=lambda s: cat_vars[("Sale Type", s)].X)
    cond_sel  = max(SALE_CONDITION_SET,   key=lambda c: cat_vars[("Sale Condition", c)].X)

    row["Misc Feature"]   = misc_sel
    row["Sale Type"]      = sale_sel
    row["Sale Condition"] = cond_sel

    print(" - Enclosed Porch :", row["Enclosed Porch"])
    print(" - 3Ssn Porch     :", row["3Ssn Porch"])
    print(" - Screen Porch   :", row["Screen Porch"])
    print(" - Pool Area      :", row["Pool Area"])
    print(" - Misc Feature   :", misc_sel)
    print(" - Sale Type      :", sale_sel)
    print(" - Sale Condition :", cond_sel)
except Exception as e:
    print("[INFO] No pude volcar/mostrar 4K]:", e)

# --- 8U) Mostrar Utilities elegido y costo asociado ---
try:
    # Elegir la categoría activa por argmax de su binaria
    UTILITIES_SET = ["AllPub", "NoSewr", "NoSeWa", "ELO"]  # mismo orden que [4A]
    util_sel = max(UTILITIES_SET, key=lambda u: cat_vars[("Utilities", u)].X)
    row["Utilities"] = util_sel
    util_cost = UTILITIES_COST.get(util_sel, 0)
    print(f" - Utilities     : {util_sel}  (costo: ${util_cost:,.0f})")
except Exception as e:
    print("[INFO] No pude volcar/mostrar Utilities]:", e)

# --- 8R) Mostrar Roof Matl elegido y costo asociado ---
try:
    ROOF_MATL_KEYS = ["ClyTile","CompShg","Membran","Metal","Roll","TarGrv","Tar&Grv","WdShake","WdShngl"]
    # Filtra a los que existen realmente en cat_vars (según tu [4B])
    roof_opts = [k for k in ROOF_MATL_KEYS if ("Roof Matl", k) in cat_vars]
    roof_sel = max(roof_opts, key=lambda k: cat_vars[("Roof Matl", k)].X)
    row["Roof Matl"] = roof_sel

    # mismo dict que en [5R]
    ROOFMATL_COST = {
        "ClyTile": 17352, "CompShg": 20000, "Membran": 8011.95, "Metal": 11739,
        "Roll": 7600, "Tar&Grv": 8550, "TarGrv": 8550, "WdShake": 22500, "WdShngl": 19500
    }
    print(f" - Roof Matl     : {roof_sel}  (costo: ${ROOFMATL_COST.get(roof_sel,0):,.0f})")
except Exception as e:
    print("[INFO] No pude volcar/mostrar Roof Matl]:", e)

# --- 8E-COST) Mostrar costos seleccionados de Exterior 1st / 2nd ---
try:
    ext1_sel = max(EXTERIOR_SET, key=lambda e: cat_vars[("Exterior 1st", e)].X)
    ext2_sel = max(EXTERIOR_SET, key=lambda e: cat_vars[("Exterior 2nd", e)].X)

    c_ext1 = EXTERIOR_COST.get(ext1_sel, 0)
    c_ext2 = EXTERIOR_COST.get(ext2_sel, 0)

    print(f" - Exterior 1st  : {ext1_sel}  (costo: ${c_ext1:,.2f})")
    print(f" - Exterior 2nd  : {ext2_sel}  (costo: ${c_ext2:,.2f})")
except Exception as e:
    print("[INFO] No pude mostrar costos de Exterior 1st/2nd]:", e)

# --- 8MVT) Mostrar costo de Mas Vnr Type ---
try:
    t_sel = max(MAS_VNR_TYPE_SET, key=lambda t: cat_vars[("Mas Vnr Type", t)].X)
    row["Mas Vnr Type"] = t_sel
    mv_area = int(round(vars_map["Mas Vnr Area"].X))
    unit = MAS_VNR_UNIT_COST.get(t_sel, 0.0)
    mv_cost = unit * mv_area
    print(f" - Mas Vnr Type  : {t_sel}")
    print(f" - Mas Vnr Area  : {mv_area} ft²")
    print(f"   Costo Mampost.: ${mv_cost:,.2f}  ({unit:.2f} $/ft²)")
except Exception as e:
    print("[INFO] No pude mostrar Mas Vnr Type/Area]:", e)

# --- 8FOUND) Mostrar costo de Foundation ---
try:
    f_sel = max(FOUNDATION_SET, key=lambda f: cat_vars[("Foundation", f)].X)
    row["Foundation"] = f_sel
    area_bsm = int(round(vars_map["Total Bsmt SF"].X))
    unit_cost = FOUNDATION_UNIT_COST.get(f_sel, 0.0)
    found_cost = unit_cost * area_bsm
    print(f" - Foundation    : {f_sel}")
    print(f"   Área Bsmt     : {area_bsm} ft²")
    print(f"   Costo ciment. : ${found_cost:,.2f}  ({unit_cost:.2f} $/ft²)")
except Exception as e:
    print("[INFO] No pude mostrar Foundation]:", e)


# --- 8HEAT) Mostrar Heating y costo asociado ---
try:
    heat_sel = max(HEATING_SET, key=lambda h: cat_vars[("Heating", h)].X)  # HEATING_SET viene de [4F]
    row["Heating"] = heat_sel
    h_cost = HEATING_COST.get(heat_sel, 0.0)
    print(f" - Heating       : {heat_sel}  (costo: ${h_cost:,.2f})")
except Exception as e:
    print("[INFO] No pude volcar/mostrar Heating]:", e)

# --- 8-CA&EL) Mostrar Central Air y Electrical con costo asociado ---
try:
    # Conjuntos de [4F]
    CENTRAL_AIR_SET = ["Y", "N"]
    ELECTRICAL_SET  = ["SBrkr", "FuseA", "FuseF", "FuseP", "Mix"]

    # Central Air
    ca_sel = max(CENTRAL_AIR_SET, key=lambda a: cat_vars[("Central Air", a)].X)
    row["Central Air"] = ca_sel
    ca_cost = CENTRAL_AIR_COST if ca_sel == "Y" else 0.0
    print(f" - Central Air   : {ca_sel}  (costo: ${ca_cost:,.2f})")

    # Electrical
    el_sel = max(ELECTRICAL_SET, key=lambda e: cat_vars[("Electrical", e)].X)
    row["Electrical"] = el_sel
    el_cost = ELECTRICAL_COST.get(el_sel, 0.0)
    print(f" - Electrical    : {el_sel}  (costo: ${el_cost:,.2f})")
except Exception as e:
    print("[INFO] No pude mostrar CA/EL]:", e)

# --- 8-MISC/PD) Mostrar Misc Feature y Paved Drive con costo asociado ---
try:
    # Conjuntos ya definidos en 4K y 4J
    MISC_FEATURE_SET = ["Elev", "Gar2", "Othr", "Shed", "TenC", "No aplica"]  # por si no está global
    PAVED_DRIVE_SET  = ["Y", "P", "N"]

    # Misc Feature
    misc_opts = [m for m in MISC_FEATURE_SET if ("Misc Feature", m) in cat_vars]
    if misc_opts:
        misc_sel = max(misc_opts, key=lambda m: cat_vars[("Misc Feature", m)].X)
        row["Misc Feature"] = misc_sel
        print(f" - Misc Feature  : {misc_sel}  (costo: ${MISC_FEATURE_COST.get(misc_sel,0):,.2f})")

    # Paved Drive
    pd_opts = [p for p in PAVED_DRIVE_SET if ("Paved Drive", p) in cat_vars]
    if pd_opts:
        pd_sel = max(pd_opts, key=lambda p: cat_vars[("Paved Drive", p)].X)
        row["Paved Drive"] = pd_sel
        print(f" - Paved Drive   : {pd_sel}  (costo: ${PAVED_DRIVE_COST.get(pd_sel,0):,.2f})")

    # Basement acabado (solo recordatorio; no se suma aún)
    print(f" - [INFO] Basement finish unit cost guardado: ${BASEMENT_FINISH_UNIT_COST:.2f} /ft² (no aplicado todavía)")
except Exception as e:
    print("[INFO] No pude mostrar Misc/Paved/BSMT]:", e)

# --- 8-BSMTQUAL) Mostrar 'Bsmt Qual' y costo asociado ---
try:
    BSMT_QUAL_SET = ["Ex", "Gd", "TA", "Fa", "Po", "NA"]  # esperado por Ames/XGB
    # filtra solo las categorías que efectivamente existen en cat_vars
    bq_opts = [q for q in BSMT_QUAL_SET if ("Bsmt Qual", q) in cat_vars]
    if bq_opts:
        bq_sel = max(bq_opts, key=lambda q: cat_vars[("Bsmt Qual", q)].X)
        row["Bsmt Qual"] = bq_sel
        print(f" - Bsmt Qual     : {bq_sel}  (costo: ${BSMT_QUAL_COST.get(bq_sel,0):,.2f})")
    else:
        # si aún no definiste la naturaleza de 'Bsmt Qual'
        bq_sel = DEFAULTS.get("Bsmt Qual", "TA")
        row["Bsmt Qual"] = bq_sel
        print(f" - Bsmt Qual     : {bq_sel}  (costo: ${BSMT_QUAL_COST.get(bq_sel,0):,.2f}) [DEFAULT]")
except Exception as e:
    print("[INFO] No pude mostrar Bsmt Qual]:", e)

# --- 8-BSMTFIN/KITCHEN/BATH) Mostrar decisiones y costos ---

try:
    # BsmtFin Type 1 y 2
    BSMT_FIN_TYPE_SET = ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"]
    def _sel_if_exists(varname):
        opts = [t for t in BSMT_FIN_TYPE_SET if (varname, t) in cat_vars]
        return max(opts, key=lambda t: cat_vars[(varname, t)].X) if opts else None

    t1 = _sel_if_exists("BsmtFin Type 1")
    t2 = _sel_if_exists("BsmtFin Type 2")

    if t1 is not None:
        print(f" - BsmtFin Type 1: {t1} (costo: ${BSMT_FIN_TYPE_COST.get(t1,0):,.2f})")
        row["BsmtFin Type 1"] = t1
    if t2 is not None:
        print(f" - BsmtFin Type 2: {t2} (costo: ${BSMT_FIN_TYPE_COST.get(t2,0):,.2f})")
        row["BsmtFin Type 2"] = t2

    # KitchenQual × Kitchens
    if "Kitchen AbvGr" in vars_map:
        k_count_val = int(round(vars_map["Kitchen AbvGr"].X))
    else:
        k_count_val = DEFAULTS.get("Kitchen AbvGr", 1)

    KITCHEN_QUAL_SET = ["Ex", "Gd", "TA", "Fa", "Po"]
    kq_sel = None
    opts_kq = [q for q in KITCHEN_QUAL_SET if ("Kitchen Qual", q) in cat_vars]
    if opts_kq:
        kq_sel = max(opts_kq, key=lambda q: cat_vars[("Kitchen Qual", q)].X)
        row["Kitchen Qual"] = kq_sel

    if kq_sel is not None:
        kq_cost = KITCHEN_QUAL_COST.get(kq_sel, 0.0) * k_count_val
        print(f" - Kitchen Qual  : {kq_sel} x {k_count_val}  (costo: ${kq_cost:,.2f})")
    else:
        print(" - Kitchen Qual  : [no one-hot aún] — costo no aplicado")

    # Baños
    full_b = int(round(vars_map["Full Bath"].X)) if "Full Bath" in vars_map else DEFAULTS.get("Full Bath", 0)
    half_b = int(round(vars_map["Half Bath"].X)) if "Half Bath" in vars_map else DEFAULTS.get("Half Bath", 0)
    print(f" - Full Bath     : {full_b} (costo: ${full_b*FULL_BATH_UNIT_COST:,.2f})")
    print(f" - Half Bath     : {half_b} (costo: ${half_b*HALF_BATH_UNIT_COST:,.2f})")

    # Kitchen $/ft² (solo recordatorio)
    print(f" - [INFO] Kitchen unit cost guardado: ${KITCHEN_UNIT_COST_PER_SF:.2f} /ft² (no aplicado todavía)")
except Exception as e:
    print("[INFO] No pude mostrar BsmtFin/Kitchen/Baths]:", e)

# --- 8-FIRE/GARAGE/POOL/FENCE) Mostrar decisiones y costos ---

try:
    # Fireplace Qu
    FQ_SET = ["Ex","Gd","TA","Fa","Po","NA"]
    opts = [x for x in FQ_SET if ("Fireplace Qu", x) in cat_vars]
    if opts:
        fq_sel = max(opts, key=lambda x: cat_vars[("Fireplace Qu", x)].X)
        row["Fireplace Qu"] = fq_sel
        print(f" - Fireplace Qu  : {fq_sel} (costo: ${FIREPLACE_QU_COST.get(fq_sel,0):,.2f})")

    # Garage Qual
    GQ_SET = ["Ex","Gd","TA","Fa","Po","NA"]
    opts = [x for x in GQ_SET if ("Garage Qual", x) in cat_vars]
    if opts:
        gq_sel = max(opts, key=lambda x: cat_vars[("Garage Qual", x)].X)
        row["Garage Qual"] = gq_sel
        print(f" - Garage Qual   : {gq_sel} (costo: ${GARAGE_QUAL_COST.get(gq_sel,0):,.2f})")

    # Garage Cond
    opts = [x for x in GQ_SET if ("Garage Cond", x) in cat_vars]
    if opts:
        gc_sel = max(opts, key=lambda x: cat_vars[("Garage Cond", x)].X)
        row["Garage Cond"] = gc_sel
        print(f" - Garage Cond   : {gc_sel} (costo: ${GARAGE_COND_COST.get(gc_sel,0):,.2f})")

    # Garage Finish
    GF_SET = ["Fin","RFn","Unf","NA"]
    opts = [x for x in GF_SET if ("Garage Finish", x) in cat_vars]
    if opts:
        gf_sel = max(opts, key=lambda x: cat_vars[("Garage Finish", x)].X)
        row["Garage Finish"] = gf_sel
        print(f" - Garage Finish : {gf_sel} (costo: ${GARAGE_FINISH_COST.get(gf_sel,0):,.2f})")

    # Garage Area
    if "Garage Area" in vars_map:
        ga = float(vars_map["Garage Area"].X)
    else:
        ga = float(DEFAULTS.get("Garage Area", 0))
    print(f" - Garage Area   : {ga:.0f} ft²  (costo: ${ga*GARAGE_AREA_UNIT_COST:,.2f})")

    # Pool QC
    PQ_SET = ["Ex","Gd","TA","Fa","NA"]
    opts = [x for x in PQ_SET if ("Pool QC", x) in cat_vars]
    if opts:
        pq_sel = max(opts, key=lambda x: cat_vars[("Pool QC", x)].X)
        row["Pool QC"] = pq_sel
        print(f" - Pool QC       : {pq_sel} (costo: ${POOL_QC_COST.get(pq_sel,0):,.2f})")

    # Pool Area
    if "Pool Area" in vars_map:
        pa = float(vars_map["Pool Area"].X)
    else:
        pa = float(DEFAULTS.get("Pool Area", 0))
    print(f" - Pool Area     : {pa:.0f} ft²  (costo: ${pa*POOL_AREA_UNIT_COST:,.2f})")

    # Fence
    FENCE_SET = ["GdPrv","MnPrv","GdWo","MnWw","NA"]
    opts = [x for x in FENCE_SET if ("Fence", x) in cat_vars]
    if opts:
        fe_sel = max(opts, key=lambda x: cat_vars[("Fence", x)].X)
        row["Fence"] = fe_sel
        print(f" - Fence         : {fe_sel} (costo: ${FENCE_COST.get(fe_sel,0):,.2f})")

except Exception as e:
    print("[INFO] No pude mostrar FIRE/GARAGE/POOL/FENCE]:", e)


# Desglose de costos con los coeficientes actuales
try:
    comp = {
        "build(Gr Liv Area)": COST_COEF.get("Gr Liv Area", 0) * row["Gr Liv Area"],
        "bedrooms": COST_COEF.get("Bedroom AbvGr", 0) * row["Bedroom AbvGr"],
        "full_bath": COST_COEF.get("Full Bath", 0) * row["Full Bath"],
        # aquí podrás añadir componentes categóricos cuando les pongas costo (>0)
        # "utilities_NoSewr": 2000 * (1 if util_sel=="NoSewr" else 0),
    }
    print("\nDesglose de costos I:")
    for k, v in comp.items():
        print(f" - {k}: {v}")
    print("Total I (check):", sum(comp.values()))
except Exception as e:
    print("[INFO] No pude calcular desglose de costos:", e)
