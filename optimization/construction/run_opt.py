# optimization/construction/run_opt.py (CLEAN)
# -------------------------------------------------------------
# Limpio para CONSTRUCCION desde cero
# - sin referencias a "remodel"
# - objetivo: max precio_predicho - costo_total
# - deja hooks para ir agregando restricciones por bloques
# -------------------------------------------------------------

import argparse
import pandas as pd
import numpy as np
import gurobipy as gp

from .config import PARAMS
from .io import get_base_house
from . import costs
from .xgb_predictor import XGBBundle
from .gurobi_model import build_mip_embed, summarize_solution

# =====================
# helpers simples
# =====================

def money(v: float) -> str:
    try:
        f = float(pd.to_numeric(v, errors="coerce"))
    except Exception:
        return str(v)
    if pd.isna(f):
        return "-"
    return f"${f:,.0f}"


def audit_cost_breakdown_vars(m: gp.Model, top: int = 20):
    expr = getattr(m, "_lin_cost_expr", None)
    if expr is None:
        print("[COST-BREAKDOWN] no _lin_cost_expr en el modelo")
        return
    try:
        vs = expr.getVars(); cs = expr.getCoeffs()
        X  = m.getAttr("X", vs)
    except Exception:
        print("[COST-BREAKDOWN] no se pudo leer terminos")
        return
    rows = []
    for v, c, x in zip(vs, cs, X):
        try:
            contrib = float(c) * float(x)
        except Exception:
            continue
        if abs(contrib) > 1e-6:
            rows.append((v.VarName, float(c), float(x), contrib))
    rows.sort(key=lambda t: abs(t[3]), reverse=True)
    print("\n[COST-BREAKDOWN] top terminos de costo:")
    for name, c, x, contr in rows[:top]:
        print(f"  {name:<35s} coef={c:>10.4f} * X={x:>10.4f}  => {contr:>10.2f}")


def audit_predict_outside(m: gp.Model, bundle: XGBBundle):
    """Predice fuera de Gurobi con el mismo X_in que vio el embed (si existe)."""
    X_in = getattr(m, "_X_input", None)
    if X_in is None or getattr(X_in, "empty", True):
        print("[AUDIT] no hay _X_input para predecir fuera")
        return
    # materializa Vars -> escalares
    Z = X_in.copy()
    for c in Z.columns:
        v = Z.loc[0, c]
        if hasattr(v, "X"):
            try:
                Z.loc[0, c] = float(v.X)
            except Exception:
                Z.loc[0, c] = 0.0
    try:
        y_hat = float(bundle.predict(Z).iloc[0])
        print(f"[AUDIT] predict fuera = {y_hat:,.2f}")
    except Exception as e:
        print(f"[AUDIT] fallo predict fuera: {e}")


# =====================
# main
# =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, default=None)
    ap.add_argument("--neigh", type=str, default=None)
    ap.add_argument("--lot", type=float, default=None)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--basecsv", type=str, default=None, help="ruta alternativa al CSV base")
    ap.add_argument("--debug-xgb", action="store_true")
    args = ap.parse_args()

    # datos del terreno/barrio, etc (parametros fijos)
    def _make_seed_row(neigh: str, lot: float) -> pd.Series:
        # valores neutros/seguros para columnas no modificables que el pipeline puede requerir
        base_defaults = {
            "Neighborhood": neigh or "NAmes",
            "LotArea": float(lot or 7000.0),
            # opciones tipicas que no deberian bloquear el pipeline
            "MS SubClass": 20,            # 1Story 1946+
            "MSZoning": "RL",
            "Street": "Pave",
            "Lot Shape": "Reg",
            "LandContour": "Lvl",
            "LotConfig": "Inside",
            "LandSlope": "Gtl",
            "BldgType": "1Fam",
            "HouseStyle": "1Story",
            "YearBuilt": 2005,
            "YearRemodAdd": 2005,
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
            "Year Sold": 2008,
            "Sale Type": "WD",
            "Sale Condition": "Normal",
            # areas por defecto 0 para que el MIP las decida
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
        }
        base_defaults.update({
            "OverallQual": 10,
            "OverallCond": 10,
            "Exter Qual": "Ex",
            "ExterCond": "Ex",
            "Bsmt Qual": "Ex",
            "Heating QC": "Ex",
            "Kitchen Qual": "Ex",
            "Low Qual Fin SF": 0.0,
        })

        return pd.Series(base_defaults)

    if args.pid is not None:
        base = get_base_house(args.pid, base_csv=args.basecsv)
        try:
            base_row = base.row
        except AttributeError:
            base_row = base if isinstance(base, pd.Series) else pd.Series(base)
    else:
        base_row = _make_seed_row(args.neigh, args.lot)

    # costos + bundle ML
    ct = costs.CostTables()
    bundle = XGBBundle()

    # construir mip
    m: gp.Model = build_mip_embed(base_row=base_row, budget=args.budget, ct=ct, bundle=bundle)

    # parametros de solucion
    m.Params.MIPGap         = PARAMS.mip_gap
    m.Params.TimeLimit      = PARAMS.time_limit
    m.Params.LogToConsole   = PARAMS.log_to_console
    m.Params.FeasibilityTol = 1e-7
    m.Params.IntFeasTol     = 1e-7
    m.Params.OptimalityTol  = 1e-7
    m.Params.NumericFocus   = 3

    m.optimize()

    
    status = int(getattr(m, "Status", -1))
    print(f"[STATUS] gurobi Status = {status}")

    # debug infeasibilidad o inf_or_unbd
    if status in (gp.GRB.INFEASIBLE, gp.GRB.INF_OR_UNBD):
        try:
            from .gurobi_model import dump_infeasibility_report
            tag = f"construction_neigh_{args.neigh}_lot_{args.lot}_budget_{int(args.budget)}"
            dump_infeasibility_report(m, tag=tag)
        except Exception as e:
            print("[DEBUG] fallo dump_infeasibility_report:", e)
        # si quieres, puedes salir aqui
        return

    try:
        picks, areas = summarize_solution(m)
        print("\n[HOUSE SUMMARY]")
        for k,v in picks.items():
            if v is not None:
                print(f"  {k:12s}: {v}")
        if areas:
            print("  areas (ft2):")
            for k,v in areas.items():
                print(f"    {k:15s} = {v:,.0f}")
    except Exception as e:
        print(f"[HOUSE SUMMARY] no disponible: {e}")

    st = m.Status
    if st in (gp.GRB.INF_OR_UNBD, gp.GRB.INFEASIBLE, gp.GRB.UNBOUNDED):
        print("\n❌ modelo infeasible/unbounded. puedes correr m.computeIIS() desde gurobi_model si quieres")
        return
    if st not in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT) or m.SolCount == 0:
        print("\n⚠️ no hay solucion valida")
        return

    # leer solucion
    y_var = getattr(m, "_y_price_var", None)
    c_var = m.getVarByName("cost_model") or getattr(m, "_cost_var", None)
    y_price = float(y_var.X) if y_var is not None else float("nan")
    total_cost = float(c_var.X) if c_var is not None else float("nan")
    obj = float(getattr(m, "objVal", float("nan")))

    print("\n" + "="*60)
    print("             RESULTADOS OPTIMIZACION CONSTRUCCION")
    print("="*60 + "\n")

    print(f"📍 PID: {base_row.get('PID', 'N/A')} – {base_row.get('Neighborhood', 'N/A')} | Presupuesto: {money(args.budget)}")
    print(f"⏱️ Tiempo: {getattr(m, 'Runtime', 0.0):.2f}s | Gap: {getattr(m, 'MIPGap', 0.0)*100:.4f}%\n")

    print("💰 resumen economico")
    print(f"  Precio predicho (post):  {money(y_price)}")
    print(f"  Costos totales modelo:   {money(total_cost)}")
    print(f"  Objetivo (utilidad):     {money(obj)}  (= precio - costo)")
    try:
        budget = float(getattr(m, "_budget_usd", args.budget))
        print(f"  Slack presupuesto:       {money(budget - total_cost)}")
    except Exception:
        pass

    # breakdown simple
    audit_cost_breakdown_vars(m, top=25)
    audit_predict_outside(m, bundle)

    print("\n" + "="*60)
    print("                     FIN RESULTADOS")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()


