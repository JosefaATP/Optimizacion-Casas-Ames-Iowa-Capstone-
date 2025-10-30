# === FILE: optimization/construction/summary_and_costs_hooks.py ===

from .xgb_predictor import ROOF_STYLE_TO_ORD, ROOF_MATL_TO_ORD

"""Lightweight hooks you can import from run_opt.py or gurobi_model.py
   - build_cost_expr(m, x, ct, params)
   - summarize_solution(m, x, base_row, ct, params)
"""
from typing import Dict
import gurobipy as gp

def build_cost_expr(m: gp.Model, x: Dict[str, gp.Var], ct, params: Dict) -> gp.LinExpr:
    cost = gp.LinExpr(0.0)
    # estructura de costos, respeta tu costs.py (rellena valores allí)
    # 1) área construida
    c_build = float(getattr(ct, "construction_cost", 0.0))
    cost += c_build * (x["1st Flr SF"] + x["2nd Flr SF"] + x["BsmtFin SF 1"] + x["BsmtFin SF 2"])  
    # 2) techo por material y área real
    for mat, unit in getattr(ct, "roof_cost_per_sf", {}).items():
        v = m._x.get(f"roof_matl_is_{mat}")
        if v is not None and unit:
            cost += unit * x["ActualRoofArea"] * v
    # 3) garage (área)
    cost += float(getattr(ct, "garage_cost_per_sf", 0.0)) * x["Garage Area"]
    # 4) porches / pool
    for nm, unit in {
        "Total Porch SF": getattr(ct, "porch_cost_per_sf", 0.0),
        "Pool Area": getattr(ct, "pool_cost_per_sf", 0.0),
    }.items():
        if unit:
            cost += unit * x[nm]
    return cost


def summarize_solution(m: gp.Model, x: Dict[str, gp.Var], base_row, ct, params: Dict) -> str:
    get = lambda k: float(x[k].X) if k in x else None
    def chosen(prefix, labels):
        for lb in labels:
            v = m._x.get(f"{prefix}{lb}")
            if v is not None and v.X > 0.5:
                return lb
        return "(none)"
    roof_s = chosen("roof_style_is_", ROOF_STYLE_TO_ORD)
    roof_m = chosen("roof_matl_is_", ROOF_MATL_TO_ORD)

    lines = []
    lines.append("\n================= RESUMEN CONSTRUCCIÓN =================")
    lines.append(f"Pisos: Floor1={get('Floor1'):.0f}, Floor2={get('Floor2'):.0f}")
    lines.append(f"Area 1st/2nd: {get('1st Flr SF'):.0f} / {get('2nd Flr SF'):.0f}")
    lines.append(f"PlanRoof={get('PlanRoofArea'):.0f}  ActualRoof={get('ActualRoofArea'):.0f}  Style={roof_s}  Matl={roof_m}")
    lines.append(f"Garage: cars={get('Garage Cars'):.0f}, area={get('Garage Area'):.0f}")
    lines.append(f"Ambientes: beds={get('Bedroom AbvGr'):.0f}, full={get('Full Bath'):.0f}, half={get('Half Bath'):.0f}, kitchens={get('Kitchen AbvGr'):.0f}, fp={get('Fireplaces'):.0f}")
    return "\n".join(lines)
