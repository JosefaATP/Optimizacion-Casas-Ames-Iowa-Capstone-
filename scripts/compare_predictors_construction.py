# scripts/compare_predictors_construction.py
"""
Resuelve una optimización de CONSTRUCCIÓN y compara la predicción del XGB
vs una regresión base sobre la casa construida.

Ejemplo:
PYTHONPATH=. python3 scripts/compare_predictors_construction.py \
    --pid 526301100 \
    --budget 250000 \
    --reg-model models/reg/base_reg.joblib \
    --time-limit 120

Si no tienes PID, puedes pasar --neigh y --lot para generar una fila semilla.
"""

import argparse
import joblib
import numpy as np
import pandas as pd
import gurobipy as gp

from optimization.construction.io import get_base_house
from optimization.construction import costs
from optimization.construction.xgb_predictor import XGBBundle
from optimization.construction.gurobi_model import build_mip_embed
from optimization.construction.config import PARAMS, PATHS


def _make_seed_row(neigh: str, lot: float) -> pd.Series:
    """Construye una fila base “semilla” mínima para construcción nueva."""
    base_defaults = {
        "Neighborhood": neigh or "NAmes",
        "LotArea": float(lot or 7000.0),
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
        # calidades altas por default (puedes cambiarlas si quieres)
        "OverallQual": 10,
        "OverallCond": 10,
        "Exter Qual": 4,
        "ExterCond": 4,
        "Bsmt Qual": 4,
        "Heating QC": 4,
        "Kitchen Qual": 4,
        "Fireplace Qu": 4,
        "Garage Qual": 4,
        "Garage Cond": 4,
        "Basement": 1,
        "Has Garage": 1,
        "Has Pool": 0,
        "Has Fireplace": 0,
        "Central Air": "Y",
        "LotFrontage": float(lot or 7000.0) ** 0.5,
    }
    return pd.Series(base_defaults)


def _materialize_x_input(m: gp.Model) -> pd.DataFrame:
    Xi = getattr(m, "_X_input", None)
    if Xi is None:
        raise RuntimeError("No _X_input disponible para reconstruir la fila óptima")
    if isinstance(Xi, dict) and "order" in Xi and "x" in Xi:
        order = list(Xi["order"])
        xvars = Xi["x"]
        Z = pd.DataFrame([[0.0] * len(order)], columns=order)
        for c in order:
            v = xvars.get(c)
            try:
                Z.loc[0, c] = float(v.X) if hasattr(v, "X") else float(v)
            except Exception:
                Z.loc[0, c] = 0.0
        return Z
    if hasattr(Xi, "copy"):
        Z = Xi.copy()
        for c in getattr(Z, "columns", []):
            v = Z.loc[0, c]
            if hasattr(v, "X"):
                try:
                    Z.loc[0, c] = float(v.X)
                except Exception:
                    Z.loc[0, c] = 0.0
        return Z
    raise RuntimeError("Formato inesperado de _X_input")


def main(pid: int | None, neigh: str | None, lot: float | None, budget: float,
         reg_model_path: str, time_limit: float | None = None, xgbdir: str | None = None,
         basecsv: str | None = None):
    # Base row
    if pid is not None:
        base_house = get_base_house(pid, base_csv=basecsv)
        base_row = base_house.row
    else:
        base_row = _make_seed_row(neigh or "NAmes", lot or 7000.0)

    ct = costs.CostTables()
    if xgbdir:
        # permitir usar un modelo XGB alternativo
        from pathlib import Path
        from optimization.construction.xgb_predictor import PATHS as PCONST
        PCONST.model_dir = Path(xgbdir)
        PCONST.xgb_model_file = Path(xgbdir) / "model_xgb.joblib"
    bundle = XGBBundle()

    m = build_mip_embed(base_row, budget, ct, bundle, base_price=None)
    m.Params.TimeLimit = float(time_limit if time_limit is not None else PARAMS.time_limit)
    m.optimize()

    if m.Status not in {gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL, gp.GRB.TIME_LIMIT}:
        raise RuntimeError(f"Modelo no resuelto (status={m.Status})")

    X_in = _materialize_x_input(m)

    precio_xgb = float(bundle.predict(X_in).iloc[0])

    reg_model = joblib.load(reg_model_path)
    precio_reg = float(reg_model.predict(X_in)[0])

    uplift_pct = (precio_xgb - precio_reg) / precio_reg * 100 if precio_reg != 0 else np.nan

    print("\n=== COMPARACIÓN CONSTRUCCIÓN: XGB vs REG ===")
    if pid is not None:
        print(f"PID base: {pid}")
    else:
        print(f"Semilla: Neighborhood={base_row.get('Neighborhood')} | LotArea={base_row.get('LotArea')}")
    print(f"Presupuesto: ${budget:,.0f}")
    print(f"Precio XGB (óptimo):     ${precio_xgb:,.0f}")
    print(f"Precio Regresión (ópt):  ${precio_reg:,.0f}")
    print(f"Diferencia absoluta:     ${precio_xgb - precio_reg:,.0f}")
    print(f"Diferencia % XGB vs Reg: {uplift_pct:.2f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, default=None, help="PID de casa base (opcional)")
    ap.add_argument("--neigh", type=str, default=None, help="Barrio para semilla si no hay PID")
    ap.add_argument("--lot", type=float, default=None, help="LotArea para semilla si no hay PID")
    ap.add_argument("--budget", type=float, required=True, help="Presupuesto de construcción")
    ap.add_argument("--reg-model", required=True, help="Ruta al joblib de la regresión base")
    ap.add_argument("--time-limit", type=float, default=None, help="TimeLimit del solver (segundos)")
    ap.add_argument("--xgbdir", type=str, default=None, help="Carpeta alternativa que contenga model_xgb.joblib")
    ap.add_argument("--basecsv", type=str, default=None, help="CSV base alternativo para PID")
    args = ap.parse_args()

    main(args.pid, args.neigh, args.lot, args.budget,
         reg_model_path=args.reg_model,
         time_limit=args.time_limit,
         xgbdir=args.xgbdir,
         basecsv=args.basecsv)
