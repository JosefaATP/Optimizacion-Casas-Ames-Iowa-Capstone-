import argparse
import gurobipy as gp
from .config import PARAMS
from .io import get_base_house
from .costs import CostTables
from .xgb_predictor import XGBBundle
from .gurobi_model import build_mip_embed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--basecsv", type=str, default=None, help="ruta alternativa al CSV base")
    args = ap.parse_args()

    base = get_base_house(args.pid, base_csv=args.basecsv)
    ct = CostTables()
    bundle = XGBBundle()

    m: gp.Model = build_mip_embed(base.row, args.budget, ct, bundle)
    m.Params.MIPGap = PARAMS.mip_gap
    m.Params.TimeLimit = PARAMS.time_limit
    m.Params.LogToConsole = PARAMS.log_to_console
    m.optimize()

    status = m.Status
    if status in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT):
        try:
            y = m.getVarByName("y_price").X
        except Exception:
            y = m.getVarByName("y_log").X
        decisions = {v.VarName: v.X for v in m.getVars() if v.VarName.startswith("x_")}
        print("precio_estimado:", y)
        print("decisiones:", decisions)
        print("rentabilidad_max:", m.ObjVal)
    else:
        print("no se encontro solucion factible, status:", status)

if __name__ == "__main__":
    main()
