import argparse
import gurobipy as gp
import pandas as pd

from optimization.construction import costs
from optimization.construction.xgb_predictor import XGBBundle
from optimization.construction.gurobi_model import build_mip_embed
from optimization.construction.run_opt import (
    compute_neigh_means,
    compute_neigh_modes,
    enforce_neigh_means,
    enforce_neigh_modes,
)


def seed(neigh: str, lot: float) -> pd.Series:
    base = {
        "Neighborhood": neigh,
        "LotArea": lot,
        "MS SubClass": 20,
        "MSZoning": "RL",
        "Street": "Pave",
        "Lot Shape": "Reg",
        "LandContour": "Lvl",
        "LotConfig": "Inside",
        "LandSlope": "Gtl",
        "BldgType": "1Fam",
        "HouseStyle": "1Story",
        "YearBuilt": 2025,
        "YearRemodAdd": 2025,
        "Foundation": "PConc",
        "Condition 1": "Norm",
        "Condition 2": "Norm",
        "Roof Style": "Gable",
        "Garage Cars": 0,
        "Low Qual Fin SF": 0,
        "Bsmt Qual": "TA",
        "Bsmt Full Bath": 0,
        "Bsmt Half Bath": 0,
        "Month Sold": 6,
        "Year Sold": 2025,
        "Sale Type": "WD",
        "Sale Condition": "Normal",
        "1st Flr SF": 0.0,
        "2nd Flr SF": 0.0,
        "Gr Liv Area": 0.0,
        "Total Bsmt SF": 0.0,
        "Bsmt Unf SF": 0.0,
        "BsmtFin SF 1": 0.0,
        "BsmtFin SF 2": 0.0,
        "Garage Area": 0.0,
        "Wood Deck SF": 0.0,
        "Open Porch SF": 0.0,
        "Enclosed Porch": 0.0,
        "3Ssn Porch": 0.0,
        "Screen Porch": 0.0,
        "Pool Area": 0.0,
        "Bedroom AbvGr": 0,
        "Full Bath": 0,
        "Half Bath": 0,
        "Kitchen AbvGr": 0,
        "OverallQual": 10,
        "OverallCond": 10,
        "Exter Qual": 4,
        "ExterCond": 4,
        "Heating QC": 4,
        "Kitchen Qual": 4,
        "Utilities": 3,
    }
    return pd.Series(base)


def main():
    parser = argparse.ArgumentParser(
        description="Diagnóstico rápido de violaciones para el modelo de construcción.",
    )
    parser.add_argument("--neigh", type=str, default="GrnHill")
    parser.add_argument("--lot", type=float, default=7000.0)
    parser.add_argument("--budget", type=float, default=500_000.0)
    parser.add_argument("--time", type=float, default=90.0, help="Time limit (segundos)")
    args = parser.parse_args()

    base_row = seed(args.neigh, float(args.lot))
    ct = costs.CostTables()
    bundle = XGBBundle()
    bundle.autocalibrate_offset(None)

    m = build_mip_embed(base_row=base_row, budget=float(args.budget), ct=ct, bundle=bundle)

    means = compute_neigh_means(args.neigh)
    if means:
        enforce_neigh_means(m, means)
    modes = compute_neigh_modes(args.neigh)
    if modes:
        enforce_neigh_modes(m, modes)

    try:
        bundle.autocalibrate_offset(base_row)
    except Exception:
        bundle.autocalibrate_offset(None)

    m.Params.MIPGap = 0.001
    m.Params.TimeLimit = float(args.time)
    m.Params.LogToConsole = 0
    m.optimize()

    print("status:", m.Status)
    cons = m.getConstrs()
    viol_pairs = []
    for c in cons:
        try:
            slack = float(c.Slack)
        except AttributeError:
            continue
        sense = getattr(c, "Sense", "=")
        if sense == "<":
            viol = max(0.0, -slack)
        elif sense == ">":
            viol = max(0.0, slack)
        else:
            viol = abs(slack)
        viol_pairs.append((viol, c.ConstrName))
    viol_pairs.sort(reverse=True)
    print("Top linear violations:")
    for v, name in viol_pairs[:10]:
        print(f"{name}: {v:.2f}")

    print("\nGeneral constraints no disponibles en esta versión de Gurobi (se omiten).")


if __name__ == "__main__":
    main()
