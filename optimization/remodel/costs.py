# optimization/remodel/costs.py
from dataclasses import dataclass
from typing import Dict

@dataclass
class CostTables:
    # ===== costos unitarios existentes =====
    add_bedroom: float = 8000
    add_bathroom: float = 12000
    deck_per_m2: float = 200
    garage_per_car: float = 9000
    finish_basement_per_f2: float = 15
    kitchenQual_upgrade_TA: float = 10   # demo
    kitchenQual_upgrade_EX: float = 10   # demo

    # ===== Utilities (ya tenías) =====
    utilities_costs = {
        "AllPub": 31750.0,
        "NoSewr": 39500.0,
        "NoSeWa": 22000.0,
        "ELO":    20000.0,
    }
    def util_cost(self, name: str) -> float:
        return float(self.utilities_costs.get(str(name), 0.0))

    # costo Fijo por CAMBIO de estilo (elige números que te acomoden)
    roof_style_costs = {
        "Flat":   3500.0,
        "Gable":  3000.0,
        "Gambrel":4500.0,
        "Hip":    4000.0,
        "Mansard":6000.0,
        "Shed":   2000.0,
    }
    def roof_style_cost(self, name: str) -> float:
        return float(self.roof_style_costs.get(str(name), 0.0))

    # costo por ft² para material (usa los del PDF; aquí un ejemplo)
    roof_matl_costs_sqft = {
        "ClyTile": 11.89,
        "CompShg": 6.00,     # = Asphalt/Composite Shingle
        "Membran": 6.00,
        "Metal":   8.99,
        "Roll":    3.75,
        "Tar&Grv": 5.50,
        "WdShake": 11.00,
        "WdShngl": 6.35,
    }
    def roof_matl_cost(self, name: str) -> float:
        return float(self.roof_matl_costs_sqft.get(str(name), 0.0))


    # costo fijo de proyecto
    project_fixed: float = 0


    # costo inicial de la casa (si tienes una columna real cámbialo)
    def initial_cost(self, base_row) -> float:
        return float(base_row.get("InitialCost", 0.0))
