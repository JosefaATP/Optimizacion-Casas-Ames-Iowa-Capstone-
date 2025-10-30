# === FILE: optimization/construction/features.py ===
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class FeatureSpec:
    name: str
    lb: float
    ub: float
    vartype: str = "C"  # C=continuous, I=integer, B=binary

# --------------------------
# Canonical label sets (Ames)
# --------------------------
ROOF_STYLE = ["Flat","Gable","Gambrel","Hip","Mansard","Shed"]
ROOF_MATL  = ["ClyTile","CompShg","Membran","Metal","Roll","Tar&Grv","WdShake","WdShngl"]
EXT_MATS   = [
    "AsbShng","AsphShn","BrkComm","BrkFace","CBlock","CemntBd","HdBoard","ImStucc",
    "MetalSd","Other","Plywood","PreCast","Stone","Stucco","VinylSd","Wd Sdng","WdShngl",
]
BLDG_TYPES = ["1Fam","2FmCon","Duplex","TwnhsE","TwnhsI"]
HOUSE_STYLES = ["1Story","2Story"]  # por modelo: solo 1 o 2 pisos
UTILS = ["ELO","NoSeWa","NoSewr","AllPub"]
FOUNDATION = ["BrkTil","CBlock","PConc","Slab","Stone","Wood"]
BSMT_EXPOS = ["Gd","Av","Mn","No","NA"]
BSMT_TYPES = ["GLQ","ALQ","BLQ","Rec","LwQ","Unf","NA"]
HEATING = ["Floor","GasA","GasW","Grav","OthW","Wall"]
CENTRAL_AIR = ["N","Y"]
ELECTRICAL = ["SBrkr","FuseA","FuseF","FuseP","Mix"]
GARAGE_TYPE = ["2Types","Attchd","Basment","BuiltIn","CarPort","Detchd","NA"]
GARAGE_FINISH = ["Fin","RFn","Unf","NA"]
PAVED_DRIVE = ["Y","P","N"]
MISC_FEATURE = ["Elev","Gar2","Othr","Shed","TenC","NA"]

# --------------------------
# Modifiable / decision features
# --------------------------
MODIFIABLE: List[FeatureSpec] = [
    # Areas por piso + porches/piscina
    FeatureSpec("1st Flr SF", 0.0, 1_000_000.0, "C"),
    FeatureSpec("2nd Flr SF", 0.0, 1_000_000.0, "C"),
    FeatureSpec("Total Porch SF", 0.0, 1_000_000.0, "C"),
    FeatureSpec("Pool Area", 0.0, 1_000_000.0, "C"),
    # Sótano
    FeatureSpec("Total Bsmt SF", 0.0, 1_000_000.0, "C"),
    FeatureSpec("BsmtFin SF 1", 0.0, 1_000_000.0, "C"),
    FeatureSpec("BsmtFin SF 2", 0.0, 1_000_000.0, "C"),
    # Conteos clave
    FeatureSpec("Bedroom AbvGr", 0, 12, "I"),
    FeatureSpec("Full Bath", 0, 6, "I"),
    FeatureSpec("Half Bath", 0, 6, "I"),
    FeatureSpec("Kitchen AbvGr", 0, 3, "I"),
    FeatureSpec("Fireplaces", 0, 6, "I"),
    # Garage
    FeatureSpec("Garage Area", 0.0, 1_000_000.0, "C"),
    FeatureSpec("Garage Cars", 0, 6, "I"),
    # Techo: variables auxiliares
    FeatureSpec("PR1", 0.0, 1_000_000.0, "C"),
    FeatureSpec("PR2", 0.0, 1_000_000.0, "C"),
    FeatureSpec("PlanRoofArea", 0.0, 1_000_000.0, "C"),
    FeatureSpec("ActualRoofArea", 0.0, 1_000_000.0, "C"),
    # Pisos (binarios)
    FeatureSpec("Floor1", 0, 1, "B"),
    FeatureSpec("Floor2", 0, 1, "B"),
    # Ordinales que usa el XGB (y que ligamos a sus one-hot)
    FeatureSpec("Roof Style", 0, len(ROOF_STYLE)-1, "I"),
    FeatureSpec("Roof Matl", 0, len(ROOF_MATL)-1, "I"),
    FeatureSpec("Utilities", 0, len(UTILS)-1, "I"),
]

# One-hot seleccionables (exclusividades). Usamos el prefijo exacto que las
# funciones de restricciones esperan.
for s in ROOF_STYLE:
    MODIFIABLE.append(FeatureSpec(f"roof_style_is_{s}", 0, 1, "B"))
for m in ROOF_MATL:
    MODIFIABLE.append(FeatureSpec(f"roof_matl_is_{m}", 0, 1, "B"))
for u in UTILS:
    MODIFIABLE.append(FeatureSpec(f"u_util_{u}", 0, 1, "B"))

# Exterior 1 / 2
for nm in EXT_MATS:
    MODIFIABLE.append(FeatureSpec(f"ex1_is_{nm}", 0, 1, "B"))
for nm in EXT_MATS:
    MODIFIABLE.append(FeatureSpec(f"ex2_is_{nm}", 0, 1, "B"))
MODIFIABLE.append(FeatureSpec("SameMaterial", 0, 1, "B"))

# Basement exposure + finish types
for x in BSMT_EXPOS:
    MODIFIABLE.append(FeatureSpec(f"bsmt_exposure_is_{x}", 0, 1, "B"))
for t in BSMT_TYPES:
    MODIFIABLE.append(FeatureSpec(f"b1_is_{t}", 0, 1, "B"))
for t in BSMT_TYPES:
    MODIFIABLE.append(FeatureSpec(f"b2_is_{t}", 0, 1, "B"))

# Heating / Central Air / Electrical
for h in HEATING:
    MODIFIABLE.append(FeatureSpec(f"heat_is_{h}", 0, 1, "B"))
for a in CENTRAL_AIR:
    MODIFIABLE.append(FeatureSpec(f"air_is_{a}", 0, 1, "B"))
for e in ELECTRICAL:
    MODIFIABLE.append(FeatureSpec(f"elect_is_{e}", 0, 1, "B"))

# Garage type / finish / paved drive
for g in GARAGE_TYPE:
    MODIFIABLE.append(FeatureSpec(f"garage_type_is_{g}", 0, 1, "B"))
for gf in GARAGE_FINISH:
    MODIFIABLE.append(FeatureSpec(f"garage_finish_is_{gf}", 0, 1, "B"))
for p in PAVED_DRIVE:
    MODIFIABLE.append(FeatureSpec(f"paved_drive_is_{p}", 0, 1, "B"))

# Misc
for s in MISC_FEATURE:
    MODIFIABLE.append(FeatureSpec(f"misc_is_{s}", 0, 1, "B"))

# --------------------------
# Immutable/base-carried cols (no se cambian por MIP, pero existen en XGB)
# --------------------------
IMMUTABLE: List[str] = [
    "MS SubClass", "Neighborhood", "OverallQual", "OverallCond", "YearBuilt",
    "YearRemodAdd", "LotArea", "LotFrontage", "BldgType", "HouseStyle",
    "Foundation", "Mas Vnr Type", "Exterior 1st", "Exterior 2nd",
]

# --------------------------
# Helper to expose bounds as dict
# --------------------------
def bounds_dict() -> Dict[str, Tuple[float, float]]:
    return {f.name: (f.lb, f.ub) for f in MODIFIABLE}


