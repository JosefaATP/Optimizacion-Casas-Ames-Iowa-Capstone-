from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from pathlib import Path
import yaml

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

    @classmethod
    def from_yaml(cls, path_or_dict):
        """Create CostTables from a YAML file path or an already-loaded dict.

        This centralizes the mapping between YAML keys and CostTables fields so all
        code uses the same source of truth.
        """
        if isinstance(path_or_dict, (str, Path)):
            path = Path(path_or_dict)
            mats = yaml.safe_load(open(path, "r", encoding="utf-8"))
        else:
            mats = path_or_dict or {}

        return cls(
            utilities=mats.get("Utilities", {}),
            roof_style=mats.get("RoofStyle", {}),
            roof_matl=mats.get("RoofMatl", {}),
            exterior1st=mats.get("Exterior1st", {}),
            exterior2nd=mats.get("Exterior2nd", {}),
            mas_vnr_type=mats.get("MasVnrType", {}),
            electrical=mats.get("Electrical", {}),
            heating=mats.get("Heating", {}),
            kitchen_qual=mats.get("KitchenQual", {}),
            central_air_install=mats.get("CentralAirInstall", cls.central_air_install),
            cost_finish_bsmt_ft2=mats.get("CBsmt", cls.cost_finish_bsmt_ft2),
            cost_pool_ft2=mats.get("PoolCostPerFt2", cls.cost_pool_ft2),
            cost_addition_ft2=mats.get("Cstr_floor", cls.cost_addition_ft2),
            cost_demolition_ft2=mats.get("Cdemolition", cls.cost_demolition_ft2),
            kitchen_remodel=mats.get("KitchenRemodel", {}),
        )

@dataclass
class PoolRules:
    max_share: float       # p.ej 0.20 del terreno
    min_area_per_quality: int  # p.ej 100 ft2 por nivel de calidad

@dataclass
class Compat:
    roof_style_by_matl_forbidden: Set[tuple]  # {(style, matl) no permitido}
