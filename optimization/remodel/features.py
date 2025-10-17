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
    FeatureSpec("Wood Deck SF", 0.0, 120.0, "C"),
    FeatureSpec("Total Bsmt SF", 0.0, 300.0, "C"),
    FeatureSpec("Gr Liv Area", 0.0, 10000.0, "C"),
    FeatureSpec("1st Flr SF", 0.0, 5000.0, "C"),
    FeatureSpec("2nd Flr SF", 0.0, 4000.0, "C"),
    FeatureSpec("Low Qual Fin SF", 0.0, 1000.0, "C"),
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

# features fijas que el modelo necesita pero no se modifican (tomadas de la casa base)
IMMUTABLE: List[str] = [
    "MSSubClass", "Neighborhood", "OverallQual", "OverallCond",
    "YearBuilt", "YearRemodAdd", "CentralAir", "Functional",
]

# mapeo ordinal (ejemplo). Ajusta a tu encoding de entrenamiento
ORDINAL_MAP = {
    "KitchenQual": {"NA":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5}
}


def bounds_dict() -> Dict[str, Tuple[float, float]]:
    return {f.name: (f.lb, f.ub) for f in MODIFIABLE}