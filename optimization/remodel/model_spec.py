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
    # nuevos (con defaults)
    bsmt_cond: Dict[str, float] = field(default_factory=dict)           # Ex, Gd, TA, Fa, Po
    bsmt_fin_type: Dict[str, float] = field(default_factory=dict)       # GLQ, ALQ, BLQ, Rec, LwQ, Unf, NA
    fireplace_qu: Dict[str, float] = field(default_factory=dict)        # Ex, Gd, TA, Fa, Po, NA
    fence_cat: Dict[str, float] = field(default_factory=dict)           # GdPrv, MnPrv, GdWo, MnWw, NA
    fence_build_psf: float = 30.0                # costo por ft² de construir cerca (FenceBuildPerFt)
    paved_drive: Dict[str, float] = field(default_factory=dict)         # Y, P, N
    garage_qual: Dict[str, float] = field(default_factory=dict)         # Ex, Gd, TA, Fa, Po, NA
    garage_cond: Dict[str, float] = field(default_factory=dict)         # Ex, Gd, TA, Fa, Po, NA
    garage_finish: Dict[str, float] = field(default_factory=dict)       # Fin, RFn, Unf, NA
    pool_qc: Dict[str, float] = field(default_factory=dict)             # Ex, Gd, TA, Fa, Po, NA
    expansion: Dict[str, float] = field(default_factory=dict)
    foundation: Dict[str, float] = field(default_factory=dict)
    misc_feature: Dict[str, float] = field(default_factory=dict)
    # bathroom/addition costs
    full_bath_cost: float = 25000.0
    half_bath_cost: float = 10000.0
    bath_remodel_per_ft2: float = 650.0

    @classmethod
    def from_yaml(cls, path_or_dict):
        """Create CostTables from a YAML file path or an already-loaded dict.

        This centralizes the mapping between YAML keys and CostTables fields so all
        code uses the same source of truth.
        """
        def load_yaml(p):
            try:
                return yaml.safe_load(open(p, "r", encoding="utf-8")) or {}
            except Exception:
                return {}

        if isinstance(path_or_dict, (str, Path)):
            path = Path(path_or_dict)
            mats = load_yaml(path)
            # try to load a sibling materials_unit.yaml and merge unit values when missing
            unit_path = path.parent / "materials_unit.yaml"
            unit_mats = load_yaml(unit_path)
            # merge: mats has priority; fill missing entries from unit_mats with mapping rules
            def merge_unit_into_main(main, unit):
                # simple copy of keys that are missing in main
                for k, v in unit.items():
                    if k not in main:
                        main[k] = v
                    else:
                        # if both are dicts, merge nested keys conservatively
                        if isinstance(main[k], dict) and isinstance(v, dict):
                            for kk, vv in v.items():
                                if kk not in main[k]:
                                    main[k][kk] = vv
                return main

            # Normalization helpers: map unit-file sections to main-file expected keys
            mapped = {}
            # direct merge first
            mapped = merge_unit_into_main(dict(mats), dict(unit_mats))
            mats = mapped
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
            # extended mappings
            bsmt_cond=mats.get("BsmtCond", {}),
            bsmt_fin_type=mats.get("BsmtFinType", {}),
            fireplace_qu=mats.get("FireplaceQu", {}),
            fence_cat=mats.get("FenceCat", {}),
            fence_build_psf=mats.get("FenceBuildPerFt", mats.get("FenceBuildPerFt", cls.fence_build_psf) if hasattr(cls, 'fence_build_psf') else 30.0),
            paved_drive=mats.get("PavedDrive", {}),
            garage_qual=mats.get("GarageQual", {}),
            garage_cond=mats.get("GarageCond", {}),
            garage_finish=mats.get("GarageFinish", {}),
            pool_qc=mats.get("PoolQC", {}),
            expansion=mats.get("Expansion", {}),
            foundation=mats.get("Foundation", {}),
            misc_feature=mats.get("MiscFeature", {}),
            full_bath_cost=mats.get("FullBathCost", cls.full_bath_cost),
            half_bath_cost=mats.get("HalfBathCost", cls.half_bath_cost),
            bath_remodel_per_ft2=mats.get("BathRemodelPerFt2", cls.bath_remodel_per_ft2),
        )

@dataclass
class PoolRules:
    max_share: float       # p.ej 0.20 del terreno
    min_area_per_quality: int  # p.ej 100 ft2 por nivel de calidad

@dataclass
class Compat:
    roof_style_by_matl_forbidden: Set[tuple]  # {(style, matl) no permitido}
