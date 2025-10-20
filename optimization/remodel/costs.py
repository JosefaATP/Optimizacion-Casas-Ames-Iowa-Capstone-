# optimization/remodel/costs.py
from dataclasses import dataclass, field
from typing import Dict
import pandas as pd

@dataclass
class CostTables:
    # ====== COSTOS BASE (numéricos) ======
    add_bedroom: float = 230.0
    add_bathroom: float = 850.0
    deck_per_m2: float = 200.0
    garage_per_car: float = 9000.0
    finish_basement_per_f2: float = 15.0

    # ====== COCINA (paquetes) ======

    kitchenQual_upgrade_TA: float = 42000.0
    kitchenQual_upgrade_EX: float = 180000.0

    # dentro de class CostTables:

    def kitchen_level_cost(self, level) -> float:

        name_to_ord = {"Po":0, "Fa":1, "TA":2, "Gd":3, "Ex":4}
        if isinstance(level, str):
            lvl = name_to_ord.get(level.strip(), None)
        else:
            try:
                lvl = int(level)
            except Exception:
                lvl = None

        if lvl is None:
            return 0.0
        if lvl >= 4:  # Ex
            return float(self.kitchenQual_upgrade_EX)
        if lvl >= 2:  # TA o Gd
            return float(self.kitchenQual_upgrade_TA)
        return 0.0    # Po/Fa


    # ====== UTILITIES ======
    utilities_costs: Dict[str, float] = field(default_factory=lambda: {
        "AllPub": 31750.0,
        "NoSewr": 39500.0,
        "NoSeWa": 22000.0,
        "ELO":    20000.0,
    })
    def util_cost(self, name: str) -> float:
        return float(self.utilities_costs.get(str(name), 0.0))

    # ====== TECHO ======
    roof_style_costs: Dict[str, float] = field(default_factory=lambda: {
        "Flat": 3500.0, "Gable": 3000.0, "Gambrel": 4500.0,
        "Hip": 4000.0, "Mansard": 6000.0, "Shed": 2000.0,
    })
    def roof_style_cost(self, name: str) -> float:
        return float(self.roof_style_costs.get(str(name), 0.0))

    roof_matl_costs_sqft: Dict[str, float] = field(default_factory=lambda: {
        "ClyTile": 11.89, "CompShg": 6.00, "Membran": 6.00,
        "Metal": 8.99, "Roll": 3.75, "Tar&Grv": 5.50,
        "WdShake": 11.00, "WdShngl": 6.35,
    })
    def roof_matl_cost(self, name: str) -> float:
        return float(self.roof_matl_costs_sqft.get(str(name), 0.0))

    # ---- Masonry veneer (Mas Vnr) ----
    mas_vnr_costs_sqft: Dict[str, float] = field(default_factory=lambda: {
        "BrkCmn": 1.21, "BrkFace": 15.0, "CBlock": 22.5, "None": 0.0, "Stone": 27.5, "No aplica": 0.0,
    })
    def mas_vnr_cost(self, name: str) -> float:
        return float(self.mas_vnr_costs_sqft.get(str(name), 0.0))
    
    # ====== GARAGE FINISH ======
    # costos totales por categoría (USD)
    garage_finish_costs_sqft: Dict[str, float] = field(default_factory=lambda: {
        "No aplica": 0.0,
        "Unf": 17500.0,
        "RFn": 17500.0,
        "Fin": 24038.0,
    })
    def garage_finish_cost(self, name: str) -> float:
        return float(self.garage_finish_costs_sqft.get(str(name), 0.0))

        # ====== POOL QUALITY ======
    pool_area_cost: float = 88.0  # USD por ft²

    # costos totales por categoría (USD)
    poolqc_costs: Dict[str, float] = field(default_factory=lambda: {
        "No aplica": 0.0,
        "Fa": 19000.0,
        "TA": 57667.0,
        "Gd": 96333.0,
        "Po": 115000.0, #Inventado
        "Ex": 135000.0,
    })

    # ====== COSTOS DE CONSTRUCCIÓN Y AMPLIACIÓN ======
    construction_cost: float = 230.0  # USD/ft²

    ampl10_cost: float = 82.28   # ampliación pequeña
    ampl20_cost: float = 106.49  # ampliación moderada
    ampl30_cost: float = 130.70  # ampliación grande
    

    # ====== GARAGE QUALITY / CONDITION ======
    garage_qc_costs: Dict[str, float] = field(default_factory=lambda: {
        "No aplica": 0.0,
        "Po": 13000.0,   # muy mala calidad
        "Fa": 19000.0,   # fair
        "TA": 57667.0,   # typical/average
        "Gd": 96333.0,   # good
        "Ex": 135000.0,  # excellent
    })

    # ====== PAVED DRIVE ======
# costos por categoría (USD, desde el PDF)
    paved_drive_costs: Dict[str, float] = field(default_factory=lambda: {
        "Y": 4908.0,   # entrada totalmente pavimentada
        "P": 3354.0,   # entrada parcialmente pavimentada (promedio)
        "N": 1800.0,   # entrada de grava / tierra
    })

    def paved_drive_cost(self, name: str) -> float:
        return float(self.paved_drive_costs.get(str(name), 0.0))
    # ====== FENCE ======
    # costos por categoría (USD)
    fence_category_costs: Dict[str, float] = field(default_factory=lambda: {
        "GdPrv": 6300.0,    # buena privacidad (8 pies de altura)
        "MnPrv": 4700.0,    # privacidad media (6 pies de altura)
        "GdWo": 1500.0,     # madera buena ($10–$14/ft aprox.)
        "MnWw": 300.0,      # alambrada económica ($2/ft × 150ft)
        "No aplica": 0.0,          # sin cerca
    })

    # costo por pie lineal de construcción nueva (USD/ft)
    fence_build_cost_per_ft: float = 40.0

    def fence_category_cost(self, f: str) -> float:
        """Costo por categoría de cerca (remodelación)"""
        return self.fence_category_costs.get(f, 0.0)


    # ====== EXTERIOR ======
    exterior_demo_face1: float = 1.65
    exterior_demo_face2: float = 1.65

    exterior_matl_costs: Dict[str, float] = field(default_factory=lambda: {
        "AsbShng": 11.50, "AsphShn": 1.50, "BrkComm": 1.21, "BrkFace": 15.00,
        "CBlock": 22.50, "CemntBd": 12.50, "HdBoard": 11.00, "ImStucc": 8.50,
        "MetalSd": 5.48, "Other": 11.56, "Plywood": 2.00, "PreCast": 37.50,
        "Stone": 27.50, "Stucco": 12.00, "VinylSd": 7.46, "Wd Sdng": 3.64,
        "WdShngl": 12.50,
    })
    def ext_mat_cost(self, name: str) -> float:
        return float(self.exterior_matl_costs.get(str(name), 0.0))

    exter_qual_upgrade_per_level: float = 3000.0
    exter_cond_upgrade_per_level: float = 2500.0

    @staticmethod
    def exterior_area_proxy(base_row: pd.Series) -> float:
        try:
            gla = float(pd.to_numeric(base_row.get("Gr Liv Area"), errors="coerce"))
        except Exception:
            gla = 0.0
        if pd.isna(gla):
            gla = 0.0
        return 0.8 * gla

    # ====== ELECTRICAL ======
    electrical_demo_small: float = 800.0
    electrical_type_costs: Dict[str, float] = field(default_factory=lambda: {
        # Ames: SBrkr, FuseA, FuseF, FuseP, Mix (peor→mejor ~ más caro)
        "FuseP": 1800.0,
        "FuseF": 2000.0,
        "FuseA": 2500.0,
        "Mix":   3000.0,
        "SBrkr": 3500.0,
    })
    def electrical_cost(self, name: str) -> float:
        return float(self.electrical_type_costs.get(str(name), 0.0))
    
        # ====== CENTRAL AIR ======
    central_air_install: float = 5362.0  # <-- puedes ajustar este costo

    # ====== HEATING ======
    heating_type_costs: Dict[str, float] = field(default_factory=lambda: {
        "Floor": 1773.0,
        "GasA":  5750.0,
        "GasW":  8500.0,
        "Grav":  6300.0,
        "OthW":  4900.0,
        "Wall":  3700.0,
    })
    def heating_type_cost(self, name: str) -> float:
        return float(self.heating_type_costs.get(str(name), 0.0))

    # Calidad (valores del cuadro; Gd y Fa interpoladas como dice la nota)
    heating_qc_costs: Dict[str, float] = field(default_factory=lambda: {
        "Ex": 10000.0,
        "Gd":  8000.0,   # interpolado (entre TA y Ex)
        "TA":  6500.0,
        "Fa":  5000.0,   # interpolado (entre Po y TA)
        "Po":  3750.0,
    })
    def heating_qc_cost(self, name: str) -> float:
        return float(self.heating_qc_costs.get(str(name), 0.0))

    # ====== BSMT COND ======
    bsmt_cond_upgrade_costs: Dict[str, float] = field(default_factory=lambda: {
        # costo de elegir este nivel como final cuando la base es TA/Fa/Po
        "Gd": 50000.0,
        "Ex": 62500.0,
    })
    def bsmt_cond_cost(self, name: str) -> float:
        return float(self.bsmt_cond_upgrade_costs.get(str(name), 0.0))

    # ====== BSMT FIN TYPE (costos por categoría) ======
    bsmt_type_costs: Dict[str, float] = field(default_factory=lambda: {
        # Valores del cuadro (ALQ/Rec interpolados)
        "GLQ": 75000.0,
        "ALQ": 53500.0,
        "BLQ": 32000.0,
        "Rec": 23500.0,
        "LwQ": 15000.0,
        "Unf": 11250.0,
        "No aplica": 0.0,
    })
    def bsmt_type_cost(self, name: str) -> float:
        return float(self.bsmt_type_costs.get(str(name), 0.0))

    # ====== FIREPLACE ======
    fireplace_costs: Dict[str, float] = field(default_factory=lambda: {
        "Po": 1500.0,   # Royster (rangos 1–2k)
        "Fa": 2000.0,   # interpolada (TA/Po)
        "TA": 2500.0,   # 2–3k
        "Gd": 3525.0,   # interpolada (Ex/TA)
        "Ex": 4550.0,   # 3.5–5.6k (Grupa)
        "No aplica": 0.0,
    })
    def fireplace_cost(self, name: str) -> float:
        return float(self.fireplace_costs.get(str(name), 0.0))


    # ====== COSTO FIJO Y COSTO INICIAL ======
    project_fixed: float = 0.0
    def initial_cost(self, base_row) -> float:
        return float(base_row.get("InitialCost", 0.0))
