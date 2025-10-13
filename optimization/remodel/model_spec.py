from dataclasses import dataclass
from typing import Dict, List, Set

@dataclass
class BaseHouse:
    # features base de la casa (antes de remodelar)
    features: Dict[str, any]  # viene de tu dataset, una fila

@dataclass
class CostTables:
    utilities: Dict[str, float]
    roof_style: Dict[str, float]
    roof_matl: Dict[str, float]
    exterior1st: Dict[str, float]        # <- NUEVO
    exterior2nd: Dict[str, float]        # <- NUEVO
    mas_vnr_type: Dict[str, float]
    electrical: Dict[str, float]
    central_air_install: float
    heating: Dict[str, float]
    kitchen_qual: Dict[str, float]

@dataclass
class PoolRules:
    max_share: float       # p.ej 0.20 del terreno
    min_area_per_quality: int  # p.ej 100 ft2 por nivel de calidad

@dataclass
class Compat:
    roof_style_by_matl_forbidden: Set[tuple]  # {(style, matl) no permitido}
