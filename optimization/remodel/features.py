# optimization/remodel/features.py
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np

@dataclass
class FeatureSpec:
    name: str
    lb: float
    ub: float
    vartype: str = "C"  # C continuo, I entero, B binario



# define las features que SI puede modificar la optimizacion
# ajusta los bounds segun tu PDF "modelo matematico"
MODIFIABLE = [
    # nombres EXACTOS con espacios, tal como en tu CSV/modelo
    FeatureSpec("Bedroom AbvGr", 0, 6, "I"),
    FeatureSpec("Full Bath", 0, 4, "I"),
    FeatureSpec("Garage Cars", 0, 4, "I"),
    FeatureSpec("BsmtFin SF 1", 0.0, 1000000.0, "C"),
    FeatureSpec("BsmtFin SF 2", 0.0, 100000.0, "C"),
    FeatureSpec("Bsmt Unf SF",  0.0, 100000.0, "C"),
    FeatureSpec("Gr Liv Area", 0.0, 10000.0, "C"),
    FeatureSpec("1st Flr SF", 0.0, 5000.0, "C"),
    FeatureSpec("2nd Flr SF", 0.0, 4000.0, "C"),
    FeatureSpec("Low Qual Fin SF", 0.0, 1000.0, "C"),
    # --- Mas Vnr Type (binarios) ---
    FeatureSpec("mvt_is_BrkCmn", lb=0, ub=1, vartype="B"),
    FeatureSpec("mvt_is_BrkFace", lb=0, ub=1, vartype="B"),
    FeatureSpec("mvt_is_CBlock", lb=0, ub=1, vartype="B"),
    FeatureSpec("mvt_is_No aplica", lb=0, ub=1, vartype="B"),
    FeatureSpec("mvt_is_Stone", lb=0, ub=1, vartype="B"),

]

QUALITY_COLS = [
    "Kitchen Qual", "Exter Qual", "Exter Cond",
    "Bsmt Qual", "Bsmt Cond",
    "Heating QC",
    "Fireplace Qu",
    "Garage Qual", "Garage Cond",
    "Pool QC",
]

# cada calidad: entero 0..4
for _q in QUALITY_COLS:
    MODIFIABLE.append(FeatureSpec(name=_q, lb=0, ub=4, vartype="I"))

MODIFIABLE.append(FeatureSpec("delta_KitchenQual_TA", lb=0, ub=1, vartype="B"))
MODIFIABLE.append(FeatureSpec("delta_KitchenQual_EX", lb=0, ub=1, vartype="B"))

# Utilities como entero ordinal 0..3 (ELO, NoSeWa, NoSewr, AllPub)
MODIFIABLE.append(FeatureSpec("Utilities", 0, 3, "I"))
MODIFIABLE += [
    FeatureSpec("u_util_ELO",    0, 1, "B"),
    FeatureSpec("u_util_NoSeWa", 0, 1, "B"),
    FeatureSpec("u_util_NoSewr", 0, 1, "B"),
    FeatureSpec("u_util_AllPub", 0, 1, "B"),
]

# códigos que alimentan al predictor Roof Style y Roof Matl
MODIFIABLE.append(FeatureSpec("Roof Style", 0, 5, "I"))
MODIFIABLE.append(FeatureSpec("Roof Matl",  0, 7, "I"))

# ==== ROOF: elección de estilo/material (exactamente una de cada) ====
for _nm in ["Flat","Gable","Gambrel","Hip","Mansard","Shed"]:
    MODIFIABLE.append(FeatureSpec(name=f"roof_style_is_{_nm}", lb=0, ub=1, vartype="B"))

for _nm in ["ClyTile","CompShg","Membran","Metal","Roll","Tar&Grv","WdShake","WdShngl"]:
    MODIFIABLE.append(FeatureSpec(name=f"roof_matl_is_{_nm}", lb=0, ub=1, vartype="B"))

#-----ELECTRICAL-------
for _nm in ["SBrkr","FuseA","FuseF","FuseP","Mix"]:
    MODIFIABLE.append(FeatureSpec(name=f"elect_is_{_nm}", lb=0, ub=1, vartype="B"))


    # === MATERIALES EXTERIORES ===
EXT_MATS = [
    "AsbShng","AsphShn","BrkComm","BrkFace","CBlock","CemntBd","HdBoard","ImStucc",
    "MetalSd","Other","Plywood","PreCast","Stone","Stucco","VinylSd","Wd Sdng","WdShngl",
]

for nm in EXT_MATS:
    MODIFIABLE.append(FeatureSpec(f"ex1_is_{nm}", lb=0, ub=1, vartype="B"))
for nm in EXT_MATS:
    MODIFIABLE.append(FeatureSpec(f"ex2_is_{nm}", lb=0, ub=1, vartype="B"))

# ----- HEATING: elección de tipo -----
for _nm in ["Floor","GasA","GasW","Grav","OthW","Wall"]:
    MODIFIABLE.append(FeatureSpec(name=f"heat_is_{_nm}", lb=0, ub=1, vartype="B"))

# Banderas de caminos (como en el PDF)
MODIFIABLE += [
    FeatureSpec("heat_upg_type", 0, 1, "B"),
    FeatureSpec("heat_upg_qc",   0, 1, "B"),
]

# Asegúrate de que Exter Qual y Exter Cond estén en MODIFIABLE como enteras 0..4:
# (si ya estaban, no dupliques)
# MODIFIABLE.append(Feature("Exter Qual", lb=0, ub=4, vartype="I"))
# MODIFIABLE.append(Feature("Exter Cond", lb=0, ub=4, vartype="I"))

# features fijas que el modelo necesita pero no se modifican (tomadas de la casa base)
IMMUTABLE: List[str] = [
    "MSSubClass", "Neighborhood", "OverallQual", "OverallCond",
    "YearBuilt", "YearRemodAdd", "CentralAir", "Functional"
]

# mapeo ordinal (ejemplo). Ajusta a tu encoding de entrenamiento
ORDINAL_MAP = {
    "KitchenQual": {"NA":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5}
}


def bounds_dict() -> Dict[str, Tuple[float, float]]:
    return {f.name: (f.lb, f.ub) for f in MODIFIABLE}