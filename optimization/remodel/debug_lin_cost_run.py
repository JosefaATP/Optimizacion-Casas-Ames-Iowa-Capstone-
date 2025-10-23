"""debug_lin_cost_run.py
Reconstruye el MIP para un PID y budget dados, resuelve y muestra:
 - status del modelo
 - cost_model.X
 - evaluación numérica de lin_cost (sum coef*var.X)
 - listado de todas las parejas (var, coef, var.X, contrib)
 - impresión de expr string y cualquier constante
Uso: python -m optimization.remodel.debug_lin_cost_run --pid 527425060 --budget 1
"""
import argparse
import pandas as pd
import gurobipy as gp
from .gurobi_model import build_mip_embed, build_base_input_row
from .io import get_base_house
from .costs import CostTables
from .xgb_predictor import XGBBundle


def inspect_model(m: gp.Model):
    print("MODEL STATUS:", m.Status)
    # cost_model
    cm = m.getVarByName("cost_model")
    print("cost_model var:", cm.VarName if cm is not None else None, "->", float(cm.X) if cm is not None else None)
    expr = getattr(m, "_lin_cost_expr", None)
    if expr is None:
        print("No m._lin_cost_expr found")
        return
    print('\n--- lin_cost expression (str) ---')
    try:
        print(str(expr))
    except Exception:
        pass
    # Try to get constant if available
    const = None
    try:
        const = float(expr.getConstant())
    except Exception:
        try:
            # fallback parsing
            s = str(expr)
            const = None
        except Exception:
            const = None
    print('constant term (if any):', const)

    print('\n--- Enumerating linear terms (var, coef, var.X, contrib) ---')
    terms = []
    try:
        n = expr.size()
    except Exception:
        n = 0
    if n and hasattr(expr, 'getVar'):
        for i in range(n):
            try:
                v = expr.getVar(i)
                c = float(expr.getCoeff(i))
                x = float(getattr(v, 'X', 0.0))
                contrib = c * x
                terms.append((v.VarName if v is not None else None, c, x, contrib))
            except Exception as e:
                terms.append((f'err_{i}', str(e), None, None))
    # print sorted by absolute contrib desc
    terms_sorted = sorted(terms, key=lambda t: -abs(t[3]) if t[3] is not None else 0)
    for name, c, x, contrib in terms_sorted:
        print(f"{name:40s} coef={c:12.2f} x={x:8.2f} contrib={contrib:12.2f}")

    # numeric sum
    numeric_sum = 0.0
    for _, c, x, contrib in terms:
        try:
            numeric_sum += float(contrib)
        except Exception:
            pass
    if const is not None:
        try:
            numeric_sum += float(const)
        except Exception:
            pass
    print('\nNumeric lin_cost sum from terms + const =', numeric_sum)
    # Also print cost_model var vs numeric_sum
    print('cost_model.X =', float(cm.X) if cm is not None else None)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid', type=int, required=True)
    ap.add_argument('--budget', type=float, required=True)
    args = ap.parse_args()

    base = get_base_house(args.pid)
    try:
        base_row = base.row
    except Exception:
        base_row = base if isinstance(base, pd.Series) else pd.Series(base)

    ct = CostTables()
    bundle = XGBBundle()

    X_base = build_base_input_row(bundle, base_row)
    precio_base = float(bundle.predict(X_base).iloc[0])

    print('Building MIP...')
    m = build_mip_embed(base_row, args.budget, ct, bundle, base_price=precio_base)

    # Set small time limits / tolerances same as run_opt
    m.Params.TimeLimit = 60
    m.Params.MIPGap = 0.001
    print('Optimizing...')
    m.optimize()

    inspect_model(m)

    print('\nDone')
