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

# features.py (solo muestra el patrón; no dupliques si ya existen)
MODIFIABLE += [
    FeatureSpec("Kitchen Qual",   lb=-1, ub=4, vartype="I"),
    FeatureSpec("Exter Qual",     lb=-1, ub=4, vartype="I"),
    FeatureSpec("Exter Cond",     lb=-1, ub=4, vartype="I"),
    FeatureSpec("Heating QC",     lb=-1, ub=4, vartype="I"),
    FeatureSpec("Fireplace Qu",   lb=-1, ub=4, vartype="I"),
    FeatureSpec("Bsmt Qual",      lb=-1, ub=4, vartype="I"),
    FeatureSpec("Bsmt Cond",      lb=-1, ub=4, vartype="I"),
    FeatureSpec("Garage Qual",    lb=-1, ub=4, vartype="I"),
    FeatureSpec("Garage Cond",    lb=-1, ub=4, vartype="I"),
    FeatureSpec("Pool QC",        lb=-1, ub=4, vartype="I"),
]




OHE_CATS = ["No aplica", "Po", "Fa", "TA", "Gd", "Ex"]


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


# ==== GARAGE FINISH (Ga = {Fin, RFn, Unf, No aplica}) ====
for _nm in ["Fin", "RFn", "Unf", "No aplica"]:
    MODIFIABLE.append(FeatureSpec(name=f"gfin_is_{_nm}", lb=0, ub=1, vartype="B"))

MODIFIABLE.append(FeatureSpec(name="upg_garage_finish", lb=0, ub=1, vartype="B"))

# ==== POOL QC (P = {Ex, Gd, TA, Fa, Po, No aplica}) ====
for _nm in ["Ex", "Gd", "TA", "Fa", "Po", "No aplica"]:
    MODIFIABLE.append(FeatureSpec(name=f"poolqc_is_{_nm}", lb=0, ub=1, vartype="B"))

MODIFIABLE.append(FeatureSpec(name="upg_pool_qc", lb=0, ub=1, vartype="B"))


# ==== AGREGADOS / AMPLIACIONES ====
# Binarias para nuevos ambientes
for nm in ["AddFull", "AddHalf", "AddKitch", "AddBed"]:
    MODIFIABLE.append(FeatureSpec(name=nm, lb=0, ub=1, vartype="B"))

# Binarias de ampliación por componente y escala (10/20/30 %)
COMPONENTES = ["GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
               "3SsnPorch", "ScreenPorch", "PoolArea"]
for c in COMPONENTES:
    for scale in [10, 20, 30]:
        MODIFIABLE.append(FeatureSpec(name=f"z{scale}_{c}", lb=0, ub=1, vartype="B"))

# ==== GARAGE QUAL / COND (G = {Ex, Gd, TA, Fa, Po, NA}) ====
for _nm in ["Ex", "Gd", "TA", "Fa", "Po", "NA"]:
    MODIFIABLE.append(FeatureSpec(name=f"garage_qual_is_{_nm}", lb=0, ub=1, vartype="B"))
    MODIFIABLE.append(FeatureSpec(name=f"garage_cond_is_{_nm}", lb=0, ub=1, vartype="B"))

# Variable de activación (común para ambos)
MODIFIABLE.append(FeatureSpec(name="UpgGarage", lb=0, ub=1, vartype="B"))

# ==== PAVED DRIVE (D = {Y, P, N}) ====
for _nm in ["Y", "P", "N"]:
    MODIFIABLE.append(FeatureSpec(name=f"paved_drive_is_{_nm}", lb=0, ub=1, vartype="B"))

    
# ==== FENCE (F = {GdPrv, MnPrv, GdWo, MnWw, NA}) ====
for _nm in ["GdPrv", "MnPrv", "GdWo", "MnWw", "NA"]:
    MODIFIABLE.append(FeatureSpec(name=f"fence_is_{_nm}", lb=0, ub=1, vartype="B"))


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

# ----- BSMT FIN TYPES (Type1 y Type2) -----
_BSMT_TYPES = ["GLQ","ALQ","BLQ","Rec","LwQ","Unf","No aplica"]
for _nm in _BSMT_TYPES:
    MODIFIABLE.append(FeatureSpec(name=f"b1_is_{_nm}", lb=0, ub=1, vartype="B"))
for _nm in _BSMT_TYPES:
    MODIFIABLE.append(FeatureSpec(name=f"b2_is_{_nm}", lb=0, ub=1, vartype="B"))


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