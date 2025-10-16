from dataclasses import dataclass
from typing import Dict

@dataclass
class CostTables:
    # costos unitarios simples, ajusta segun tu PDF
    add_bedroom: float = 8000
    add_bathroom: float = 12000
    kitchen_upgrade_ta_to_gd: float = 7000
    deck_per_m2: float = 200
    garage_per_car: float = 9000
    finish_basement_per_m2: float = 300

    # costo fijo de proyectos (permisos, etc.)
    project_fixed: float = 1500

    # costo inicial de la casa (sin cambios). cambia por columna real si la tienes
    def initial_cost(self, base_row) -> float:
        # si tienes una columna, ej: base_row["InitialCost"], usa eso
        return float(base_row.get("InitialCost", 0.0))


def remodel_cost(actions: Dict[str, float], ct: CostTables) -> float:
    cost = ct.project_fixed
    cost += actions.get("add_bedroom", 0) * ct.add_bedroom
    cost += actions.get("add_bathroom", 0) * ct.add_bathroom
    cost += actions.get("kitchen_upgrade_ta_to_gd", 0) * ct.kitchen_upgrade_ta_to_gd
    cost += actions.get("deck_m2", 0) * ct.deck_per_m2
    cost += actions.get("garage_cars_delta", 0) * ct.garage_per_car
    cost += actions.get("finish_basement_m2", 0) * ct.finish_basement_per_m2
    return cost