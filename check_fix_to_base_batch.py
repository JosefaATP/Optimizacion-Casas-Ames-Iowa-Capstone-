# tools/check_fix_to_base_batch.py
# Ejecuta build_mip_embed(..., fix_to_base=True) sobre las primeras N filas
# y guarda un reporte CSV con estado, y_price vs base_price, y términos negativos.

import argparse, time, json
import pandas as pd
from optimization.remodel.gurobi_model import build_mip_embed
from optimization.remodel.xgb_predictor import XGBBundle
from optimization.remodel.costs import CostTables

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="data/processed/base_completa_sin_nulos.csv")
parser.add_argument("--rows", type=int, default=50, help="Número de filas a testear")
parser.add_argument("--timelimit", type=float, default=4.0, help="TimeLimit (s) por modelo")
parser.add_argument("--budget", type=float, default=40000.0)
parser.add_argument("--out", default="debug_fix_to_base_report.csv")
args = parser.parse_args()

df = pd.read_csv(args.csv)
N = min(args.rows, len(df))
bundle = XGBBundle()
ct = CostTables()

rows = []
t0 = time.time()
for i in range(N):
    r = df.iloc[i].to_dict()
    raw_df = pd.DataFrame([r], columns=list(bundle.feature_names_in()))
    base_price = float(bundle.predict(raw_df).iloc[0])
    entry = {"row": i, "status": None, "base_price": float(base_price), "y_price": None, "y_matches": None, "neg_terms": None, "note": None}
    try:
        m = build_mip_embed(r, budget=args.budget, ct=ct, bundle=bundle, base_price=base_price, fix_to_base=True)
    except Exception as e:
        entry["status"] = "build_error"
        entry["note"] = str(e)
        rows.append(entry); continue

    m.Params.OutputFlag = 0
    m.Params.TimeLimit = args.timelimit
    try:
        m.optimize()
    except Exception as e:
        entry["status"] = "opt_error"
        entry["note"] = str(e)
        rows.append(entry); continue

    st = m.Status
    # Gurobi status codes: 2 optimal, 4 suboptimal/loaded, 3 infeasible, etc.
    if st == 3:
        entry["status"] = "infeasible"
        try:
            m.computeIIS()
            cons = [c.ConstrName for c in m.getConstrs() if c.IISConstr]
            entry["note"] = json.dumps(cons)
        except Exception as e:
            entry["note"] = "IIS_error:" + str(e)
        rows.append(entry); continue

    # try to read y_price (may fail if var missing)
    try:
        y = m.getVarByName("y_price")
        if y is not None:
            entry["y_price"] = float(y.X)
            entry["y_matches"] = abs(entry["y_price"] - entry["base_price"]) <= 1e-2
        else:
            entry["y_price"] = None
            entry["y_matches"] = None
    except Exception:
        entry["y_price"] = None
        entry["y_matches"] = None

    # inspect lin_cost for negative coefficients (LinExpr path)
    neg = []
    lin = getattr(m, "_lin_cost_expr", None)
    if lin is None:
        entry["neg_terms"] = None
    else:
        try:
            n = lin.size()
            for j in range(n):
                try:
                    v = lin.getVar(j); c = float(lin.getCoeff(j))
                    if c < -1e-9:
                        neg.append((v.VarName, c))
                except Exception:
                    # skip non-lin terms
                    pass
            entry["neg_terms"] = json.dumps(neg)
        except Exception as e:
            entry["neg_terms"] = "inspect_error:" + str(e)

    entry["status"] = "ok"
    rows.append(entry)

td = time.time() - t0
print(f"Checked {N} rows in {td:.1f}s. Writing {args.out}")
pd.DataFrame(rows).to_csv(args.out, index=False)
print("Done.")