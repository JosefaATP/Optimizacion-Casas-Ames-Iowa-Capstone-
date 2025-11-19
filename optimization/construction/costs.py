# optimization/construction/costs.py  — LIMPIO con tus valores del PDF
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class CostTables:
    # ====== OBRA GRUESA Y AREAS ======
    construction_cost: float = 230.0           # USD/ft2
    finish_basement_per_f2: float = 15.0       # USD/ft2 (terminar sotano acabado)
    # Ambientes (USD/ft2)
    kitchen_area_cost: float = 200.0           # Appendix: C_kitchen = $200/ft2
    fullbath_area_cost: float = 500.0          # Aprox 25k/50ft2
    halfbath_area_cost: float = 500.0          # Aprox 10k/20ft2
    bedroom_area_cost: float = 325.0           # Appendix: $325/ft2
    # Area caps per room (usados como big-M para ligar áreas con conteos)
    bedroom_area_cap_first: float = 450.0       # ft2 por dormitorio en 1er piso
    bedroom_area_cap_second: float = 350.0      # ft2 por dormitorio en 2do piso
    kitchen_area_cap_first: float = 350.0       # ft2 por cocina en 1er piso
    kitchen_area_cap_second: float = 275.0      # ft2 por cocina en 2do piso
    fullbath_area_cap_first: float = 140.0      # ft2 por baño completo en 1er piso
    fullbath_area_cap_second: float = 110.0     # ft2 por baño completo en 2do piso
    halfbath_area_cap_first: float = 80.0       # ft2 por medio baño en 1er piso
    halfbath_area_cap_second: float = 60.0      # ft2 por medio baño en 2do piso
    totag_max_ratio: float = 1.1                # Max área sobre rasante respecto al lote
    roof_area_max_ratio: float = 1.4            # AreaExterior/ActualRoofArea <= ratio * lot_area
    roof_projection_max_ratio: float = 0.8      # PR1/PR2 <= ratio * lot_area
    misc_val_cap: float = 100000.0              # USD, bound para Misc Val en fallback

    # ====== COCINA (paquetes) [REMOVED for construction; keep for compatibility] ======
    # En construcción usaremos costos por área diferenciados por calidad.
    kitchenQual_upgrade_TA: float = 42500.0
    kitchenQual_upgrade_EX: float = 180000.0

    # ====== UTILITIES ======
    utilities_costs: Dict[str, float] = field(default_factory=lambda: {
        "AllPub": 31750.0,
        "NoSewr": 39500.0,
        "NoSeWa": 22000.0,
        "ELO":    20000.0,
    })
    def util_cost(self, key: str) -> float:
        return float(self.utilities_costs.get(str(key), 0.0))

    # ====== TECHO ======
    roof_matl_fixed: Dict[str, float] = field(default_factory=lambda: {
        "ClyTile": 17352.0,
        "CompShg": 20000.0,
        "Membran": 8011.95,
        "Metal":   11739.0,
        "Roll":    7600.0,
        "Tar&Grv": 8550.0,
        "WdShake": 22500.0,
        "WdShngl": 19500.0,
    })
    roof_demo_cost: float = 10850.0  # si alguna vez modelas reemplazo explicito
    def get_roof_matl_cost(self, mat: str) -> float:
        return float(self.roof_matl_fixed.get(mat, 0.0))

    # ====== MASONRY VENEER (USD/ft2) ======
    mas_vnr_costs_sqft: Dict[str, float] = field(default_factory=lambda: {
        "BrkCmn": 1.21,
        "BrkFace": 15.0,
        "CBlock": 22.5,
        "Stone": 27.5,
        "None": 0.0,
        "No aplica": 0.0,
    })
    def mas_vnr_cost(self, name: str) -> float:
        return float(self.mas_vnr_costs_sqft.get(str(name), 0.0))

    # ====== GARAGE FINISH (lumpsum por categoria) ======
    garage_finish_costs_sqft: Dict[str, float] = field(default_factory=lambda: {
        "No aplica": 0.0,
        "Unf": 17500.0,
        "RFn": 20769.0,
        "Fin": 24038.0,
    })
    def garage_finish_cost(self, name: str) -> float:
        return float(self.garage_finish_costs_sqft.get(str(name), 0.0))

    # ====== PISCINA ======
    pool_area_cost: float = 88.0  # USD/ft2
    poolqc_costs: Dict[str, float] = field(default_factory=lambda: {
        "No aplica": 0.0,
        "Fa": 19000.0,
        "TA": 57667.0,
        "Gd": 96333.0,
        "Po": 115000.0,  # inventado en tu cuadro, lo respeto
        "Ex": 135000.0,
    })

    # ====== AMPLIACIONES MODULARES (USD/ft2) ======
    ampl10_cost: float = 82.28
    ampl20_cost: float = 106.49
    ampl30_cost: float = 130.70

    # ====== GARAGE QUALITY / CONDITION (lumpsum por nivel) ======
    garage_qc_costs: Dict[str, float] = field(default_factory=lambda: {
        "No aplica": 0.0,
        "Po": 4188.0,
        "Fa": 14113.0,
        "TA": 24038.0,
        "Gd": 37849.0,
        "Ex": 51659.0,
    })
    # Costo por ft2 de garage (construcción)
    garage_area_cost: float = 60.0

    # ====== PAVED DRIVE (lumpsum categoria) ======
    paved_drive_costs: Dict[str, float] = field(default_factory=lambda: {
        "Y": 4.0,
        "P": 5.0,
        "N": 1.0,
    })
    def paved_drive_cost(self, name: str) -> float:
        return float(self.paved_drive_costs.get(str(name), 0.0))

    # ====== FENCE ======
    fence_category_costs: Dict[str, float] = field(default_factory=lambda: {
        "GdPrv": 6300.0,
        "MnPrv": 4700.0,
        "GdWo": 1500.0,
        "MnWw": 300.0,
        "No aplica": 0.0,
    })
    fence_build_cost_per_ft: float = 40.0
    def fence_category_cost(self, f: str) -> float:
        return float(self.fence_category_costs.get(f, 0.0))

    # ====== PORCHES / DECK (USD/ft2) ======
    wooddeck_cost: float = 50.0
    openporch_cost: float = 77.5
    enclosedporch_cost: float = 80.0
    threessnporch_cost: float = 157.5
    screenporch_cost: float = 72.5

    # ====== FOUNDATION (USD/ft2) ======
    foundation_cost_per_sf: Dict[str, float] = field(default_factory=lambda: {
        # Valores base por tipo de cimentación (USD/ft2)
        # Fuente: anexo costos (aprox; ajustar si el equipo actualiza el anexo)
        "BrkTil": 22.0,
        "CBlock": 12.0,
        "PConc":  10.0,
        "Slab":   10.0,
        "Stone":  23.5,
        "Wood":   40.0,
    })

    # ====== EXTERIOR: materiales (lumpsum por frente) ======
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
        return float(self.exterior_matl_lumpsum.get(str(name), 0.0))

    # calidad/condicion exterior (lumpsum por nivel final)
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
        return float(self.exter_qual_costs.get(str(level), 0.0))
    def exter_cond_cost(self, level: str) -> float:
        return float(self.exter_cond_costs.get(str(level), 0.0))

    # ====== MISC FEATURE (lumpsum por categoría) ======
    misc_feature_costs: Dict[str, float] = field(default_factory=lambda: {
        # Anexo: costos misceláneos
        "Elev": 48000.0,
        "Gar2": 32100.0,
        "Othr": 50000.0,
        "Shed": 5631.0,
        "TenC": 15774.0,
        "No aplica": 0.0,
    })

    # ====== ELECTRICA ======
    electrical_demo_small: float = 800.0
    electrical_type_costs: Dict[str, float] = field(default_factory=lambda: {
        "FuseP": 850.0,
        "FuseF": 1675.0,
        "FuseA": 2500.0,
        "Mix":   1075.0,
        "SBrkr": 1587.5,
    })
    def electrical_cost(self, name: str) -> float:
        return float(self.electrical_type_costs.get(str(name), 0.0))

    # ====== CALIDADES (multiplicadores por nivel para obra nueva) ======
    # Convención: los unitarios base (area_cost) representan EX (1.00)
    quality_baseline_is_ex: bool = True

    kitchen_area_quality_mult: Dict[str, float] = field(default_factory=lambda: {
        "TA": 0.85, "Gd": 0.95, "Ex": 1.00,
    })
    bsmt_finish_quality_mult: Dict[str, float] = field(default_factory=lambda: {
        "TA": 0.85, "Gd": 0.95, "Ex": 1.00,
    })
    garage_area_quality_mult: Dict[str, float] = field(default_factory=lambda: {
        "TA": 0.90, "Gd": 0.96, "Ex": 1.00,
    })
    pool_area_quality_mult: Dict[str, float] = field(default_factory=lambda: {
        "TA": 0.80, "Gd": 0.92, "Ex": 1.00,
    })

    # ====== EXTERIOR (param opcional) ======
    # Si True, fuerza Exterior1st == Exterior2nd en el modelo. Por defecto False.
    exterior_require_same: bool = False

    # ====== SHARE BOUNDS (porcentaje del 1er+2do piso) ======
    # Circulación/Muros/Closets (Remainder1+Remainder2) ≈ 15%–25% del área sobre rasante
    rem_share_min: float = 0.15
    rem_share_max: float = 0.25
    # Áreas comunes (AreaOther1+AreaOther2) ≈ 20%–30% del área sobre rasante
    other_share_min: float = 0.20
    other_share_max: float = 0.30
    enforce_area_share_bounds: bool = True

    def _mult_from_ord(self, table: Dict[str, float], ord_val: int) -> float:
        # ord_val: 2=TA, 3=Gd, 4=Ex (otros => 1.0)
        key = {2: "TA", 3: "Gd", 4: "Ex"}.get(int(ord_val), None)
        return float(table.get(key, 1.0)) if key is not None else 1.0

    # ====== AIRE / CALEFACCION ======
    central_air_install: float = 5362.0
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
    heating_qc_costs: Dict[str, float] = field(default_factory=lambda: {
        "Ex": 10000.0,
        "Gd":  8250.0,
        "TA":  6500.0,
        "Fa":  5125.0,
        "Po":  3750.0,
    })
    def heating_qc_cost(self, name: str) -> float:
        return float(self.heating_qc_costs.get(str(name), 0.0))

    # ====== BSMT ======
    bsmt_cond_upgrade_costs: Dict[str, float] = field(default_factory=lambda: {
        "Gd": 51750.0,
        "Ex": 62500.0,
    })
    def bsmt_cond_cost(self, name: str) -> float:
        return float(self.bsmt_cond_upgrade_costs.get(str(name), 0.0))
    bsmt_type_costs: Dict[str, float] = field(default_factory=lambda: {
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

    # ====== CHIMENEA ======
    fireplace_costs: Dict[str, float] = field(default_factory=lambda: {
        "Po": 1500.0,
        "Fa": 2000.0,
        "TA": 2500.0,
        "Gd": 3525.0,
        "Ex": 4550.0,
        "No aplica": 0.0,
    })
    def fireplace_cost(self, name: str) -> float:
        return float(self.fireplace_costs.get(str(name), 0.0))

    # ====== FIJOS ======
    project_fixed: float = 0.0
    def initial_cost(self, base_row) -> float:
        return float(base_row.get("InitialCost", 0.0))

    # ====== REGLAS OPCIONALES (gating por tamaño, guard-rails) ======
    # Permite más cocinas en 1Fam si el área de living supera estos umbrales (ft2)
    # Se usa en gurobi_model si está definido.
    kitchen_by_area_thresholds: list[float] = field(default_factory=lambda: [3200.0, 5000.0])
    # Guard-rail por defecto: Kitchen <= 1 en 1Fam cuando no aplica el gating por tamaño
    enforce_single_kitchen_1fam: bool = True
