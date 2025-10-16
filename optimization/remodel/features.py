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
    FeatureSpec("Gr Liv Area", 20.0, 400.0, "C"),
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