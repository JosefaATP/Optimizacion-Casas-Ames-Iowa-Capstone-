# optimization/remodel/costs.py
from dataclasses import dataclass, field
from typing import Dict
import pandas as pd

@dataclass
class CostTables:
    # ====== COSTOS BASE (numéricos) ======
    add_bedroom: float = 0.0
    add_bathroom: float = 0.0
    deck_per_m2: float = 0.0
    finish_basement_per_f2: float = 15.0

    # ====== COCINA (paquetes) ====== LISTO

    kitchenQual_upgrade_TA: float = 42500.0
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


    # ====== UTILITIES ====== LISTO
    utilities_costs: Dict[str, float] = field(default_factory=lambda: {
        "AllPub": 31750.0,
        "NoSewr": 39500.0,
        "NoSeWa": 22000.0,
        "ELO":    20000.0,
    })
    def util_cost(self, name: str) -> float:
        return float(self.utilities_costs.get(str(name), 0.0))

    # ====== TECHO ====== REVISAR
    roof_style_costs: Dict[str, float] = field(default_factory=lambda: {
        "Flat": 3500.0, "Gable": 3000.0, "Gambrel": 4500.0,
        "Hip": 4000.0, "Mansard": 6000.0, "Shed": 2000.0,
    }) # <- NO VA
    def roof_style_cost(self, name: str) -> float:
        return float(self.roof_style_costs.get(str(name), 0.0))

    roof_matl_fixed = {
        "ClyTile": 17352.0,
        "CompShg": 20000.0,
        "Membran": 8011.95,
        "Metal":   11739.0,
        "Roll":    7600.0,
        "Tar&Grv": 8550.0,
        "WdShake": 22500.0,
        "WdShngl": 19500.0,
    }
    # costo fijo de demolicion del techo (ajusta segun tu paper)
    roof_demo_cost = 10850.0  # si aplica, pon el valor > 0

    def get_roof_matl_cost(self, mat: str) -> float:
        return float(self.roof_matl_fixed.get(mat, 0.0))

    # ---- Masonry veneer (Mas Vnr) ---- LISTO
    mas_vnr_costs_sqft: Dict[str, float] = field(default_factory=lambda: {
        "BrkCmn": 1.21, "BrkFace": 15.0, "CBlock": 22.5, "None": 0.0, "Stone": 27.5, "No aplica": 0.0,
    })
    def mas_vnr_cost(self, name: str) -> float:
        return float(self.mas_vnr_costs_sqft.get(str(name), 0.0))
    
    # ====== GARAGE FINISH ====== REVISAR
    # costos totales por categoría (USD)
    garage_finish_costs_sqft: Dict[str, float] = field(default_factory=lambda: {
        "No aplica": 0.0,
        "Unf": 17500.0,
        "RFn": 20769.0,
        "Fin": 24038.0,
    })
    def garage_finish_cost(self, name: str) -> float:
        return float(self.garage_finish_costs_sqft.get(str(name), 0.0))

    # ====== POOL QUALITY ====== REVISAR
    pool_area_cost: float = 88.0  # USD por ft²

    # costos totales por categoría (USD) REVISAR
    poolqc_costs: Dict[str, float] = field(default_factory=lambda: {
        "No aplica": 0.0,
        "Fa": 19000.0,
        "TA": 57667.0,
        "Gd": 96333.0,
        "Po": 115000.0, #Inventado
        "Ex": 135000.0,
    })

    # ====== COSTOS DE CONSTRUCCIÓN Y AMPLIACIÓN ====== LISTO
    construction_cost: float = 230.0  # USD/ft²

    ampl10_cost: float = 82.28   # ampliación pequeña
    ampl20_cost: float = 106.49  # ampliación moderada
    ampl30_cost: float = 130.70  # ampliación grande
    

    # ====== GARAGE QUALITY / CONDITION ====== LISTO
    garage_qc_costs: Dict[str, float] = field(default_factory=lambda: {
        "No aplica": 0.0,
        "Po": 4188.0,   # muy mala calidad
        "Fa": 14113.0,   # fair
        "TA": 24038.0,   # typical/average
        "Gd": 37849.0,   # good
        "Ex": 51659.0,  # excellent
    })

    # ====== PAVED DRIVE ======
    # costos por categoría (USD, desde el PDF)
    paved_drive_costs: Dict[str, float] = field(default_factory=lambda: {
        "Y": 4.0,   # entrada totalmente pavimentada
        "P": 5.0,   # entrada parcialmente pavimentada (promedio)
        "N": 1.0,   # entrada de grava / tierra
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
    fence_build_cost_per_ft: float = 40.0 #ARREGLAR ESTO 

    def fence_category_cost(self, f: str) -> float:
        """Costo por categoría de cerca (remodelación)"""
        return self.fence_category_costs.get(f, 0.0)


    # ====== EXTERIOR (LUMPSUM, SIN DEMOLICIÓN) ======

    # Costos fijos por CAMBIO de material del frente 1 o 2. LISTO
    exterior_matl_lumpsum: Dict[str, float] = field(default_factory=lambda: {
        "AsbShng": 19000.0,
        "AsphShn": 22500.0,
        "BrkComm": 26000.0,
        "BrkFace": 22000.0,
        "CBlock": 10300.0,
        "CemntBd": 14674.0,
        "HdBoard": 21300.0,
        "ImStucc": 16500.0,
        "MetalSd": 9600.0,
        "Other": 23461.81,
        "Plywood": 8461.81,
        "PreCast": 17625.0,
        "Stone": 50216.50,
        "Stucco": 5629.0,
        "VinylSd": 12500.0,
        "Wd Sdng": 12500.0,
        "WdShngl": 21900.0,
    })

    def ext_mat_cost(self, name: str) -> float:
        """Costo fijo por elegir el material 'name' en un frente (demolición incluida)."""
        return float(self.exterior_matl_lumpsum.get(str(name), 0.0))


    # Costos fijos por NIVEL final de calidad/condición (se cobran sólo si el nivel
    # final es superior al nivel de la casa base).
    exter_qual_costs: Dict[str, float] = field(default_factory=lambda: {
        "Po": 7646.70,
        "Fa": 14558.00,
        "TA": 18883.75,
        "Gd": 22833.06,
        "Ex": 106250.00,
    })
    exter_cond_costs: Dict[str, float] = field(default_factory=lambda: {
        "Po": 7646.70,
        "Fa": 14558.00,
        "TA": 18883.75,
        "Gd": 22833.06,
        "Ex": 106250.00,
    })

    def exter_qual_cost(self, level: str) -> float:
        """Costo fijo por terminar con calidad exterior 'level' (Po/Fa/TA/Gd/Ex)."""
        return float(self.exter_qual_costs.get(str(level), 0.0))

    def exter_cond_cost(self, level: str) -> float:
        """Costo fijo por terminar con condición exterior 'level' (Po/Fa/TA/Gd/Ex)."""
        return float(self.exter_cond_costs.get(str(level), 0.0))

    # ====== ELECTRICAL ====== LISTO
    electrical_demo_small: float = 800.0
    electrical_type_costs: Dict[str, float] = field(default_factory=lambda: {
        # Ames: SBrkr, FuseA, FuseF, FuseP, Mix (peor→mejor ~ más caro)
        "FuseP": 850.0,
        "FuseF": 1675.0,
        "FuseA": 2500.0,
        "Mix":   1075.0,
        "SBrkr": 1587.5,
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
        "OthW":  5000.0,
        "Wall":  3700.0,
    })
    def heating_type_cost(self, name: str) -> float:
        return float(self.heating_type_costs.get(str(name), 0.0))

    # Calidad (valores del cuadro; Gd y Fa interpoladas como dice la nota)
    heating_qc_costs: Dict[str, float] = field(default_factory=lambda: {
        "Ex": 10000.0,
        "Gd":  8250.0,   # interpolado (entre TA y Ex)
        "TA":  6500.0,
        "Fa":  5125.0,   # interpolado (entre Po y TA)
        "Po":  3750.0,
    })
    def heating_qc_cost(self, name: str) -> float:
        return float(self.heating_qc_costs.get(str(name), 0.0))

    # ====== BSMT COND ======
    bsmt_cond_upgrade_costs: Dict[str, float] = field(default_factory=lambda: {
        # costo de elegir este nivel como final cuando la base es TA/Fa/Po
        "Gd": 51750.0,
        "Ex": 62500.0,
    })
    def bsmt_cond_cost(self, name: str) -> float:
        return float(self.bsmt_cond_upgrade_costs.get(str(name), 0.0))

    # ====== BSMT FIN TYPE (costos por categoría) ====== LISTO
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

    # ====== FIREPLACE ====== LISTO
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
