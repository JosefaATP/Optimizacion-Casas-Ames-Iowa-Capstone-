'''import gurobipy as gp
from typing import Dict

# En este archivo agrega las restricciones especificas de tu PDF de modelo matematico.
# Te dejo plantillas comunes; copia y ajusta nombres y parametros.

def add_area_growth_caps(m: gp.Model, x: Dict[str, gp.Var], base: Dict[str, float]):
    """Limita ampliaciones maximas (ej: no crecer mas de 25% en GLA o Basement)."""
    if "Gr Liv Area" in x:
        m.addConstr(x["Gr Liv Area"] <= 1.25 * base.get("Gr Liv Area", 0.0))
    if "Total Bsmt SF" in x:
        m.addConstr(x["Total Bsmt SF"] <= 1.30 * base.get("Total Bsmt SF", 0.0))


def add_kitchen_upgrade_logic(m: gp.Model, x: Dict[str, gp.Var], base: Dict[str, float]):
    """Si KitchenQual sube, forzar costo o decision binaria (si prefieres)."""
    pass


def add_garage_integrity(m: gp.Model, x: Dict[str, gp.Var], base: Dict[str, float]):
    """GarageCars entero y dentro de rangos logicos (ya cubierto por bounds)."""
    pass'''