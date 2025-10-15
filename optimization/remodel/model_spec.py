from dataclasses import dataclass, field
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
    exterior1st: Dict[str, float]
    exterior2nd: Dict[str, float]
    mas_vnr_type: Dict[str, float]
    electrical: Dict[str, float]
    heating: Dict[str, float]
    kitchen_qual: Dict[str, float]
    # existentes
    central_air_install: float = 6000.0
    # NUEVOS (para evitar el AttributeError)
    cost_finish_bsmt_ft2: float = 20.0        # CBsmt
    cost_pool_ft2: float = 70.0               # PoolCostPerFt2
    cost_addition_ft2: float = 110.0          # Cstr_floor
    cost_demolition_ft2: float = 2.0          # Cdemolition
    kitchen_remodel: Dict[str, float] = field(default_factory=dict)

@dataclass
class PoolRules:
    max_share: float       # p.ej 0.20 del terreno
    min_area_per_quality: int  # p.ej 100 ft2 por nivel de calidad

@dataclass
class Compat:
    roof_style_by_matl_forbidden: Set[tuple]  # {(style, matl) no permitido}
