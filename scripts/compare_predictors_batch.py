# scripts/compare_predictors_batch.py
"""
Corre la comparación XGB vs regresión para varias casas (típicamente las del
benchmark de remodelación) y guarda un CSV con los resultados agregados.

Ejemplo:
PYTHONPATH=. venv/bin/python3 scripts/compare_predictors_batch.py \
    --benchmark bench_out/remodel_benchmark.csv \
    --reg-model models/reg/base_reg.joblib \
    --time-limit 120 \
    --max-rows 50
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import gurobipy as gp
from pathlib import Path

from optimization.remodel.io import get_base_house
from optimization.remodel import costs
from optimization.remodel.xgb_predictor import XGBBundle
from optimization.remodel.gurobi_model import build_mip_embed, build_base_input_row
from optimization.remodel.run_opt import rebuild_embed_input_df
from optimization.remodel.config import PARAMS


def precio_base_from_csv(pid: int):
    try:
        df = pd.read_csv("data/raw/df_final_regresion.csv")
        df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
        row = df.loc[df["PID"] == pid]
        if not row.empty and "SalePrice_Present" in row.columns:
            return float(row["SalePrice_Present"].iloc[0])
        if not row.empty and "SalePrice" in row.columns:
            return float(row["SalePrice"].iloc[0])
    except Exception:
        return None
    return None


def build_reg_input(pid: int, reg_model, base_row: pd.Series, m, X_in: pd.DataFrame) -> pd.DataFrame:
    try:
        reg_cols = list(getattr(reg_model, "feature_names_in_", []))
    except Exception:
        reg_cols = []
    if not reg_cols:
        return X_in.copy()

    try:
        df_reg = pd.read_csv("data/raw/df_final_regresion.csv")
        df_reg.columns = [c.replace("\ufeff", "").strip() for c in df_reg.columns]
        row_reg = df_reg.loc[df_reg["PID"] == pid].iloc[0]
    except Exception:
        row_reg = pd.Series({c: base_row.get(c, np.nan) for c in reg_cols})

    opt_map = {}
    if m is not None and hasattr(m, "_x_vars") and getattr(m, "_x_vars", None):
        for name, var in m._x_vars.items():
            try:
                opt_map[name] = float(var.X) if hasattr(var, "X") else float(var)
            except Exception:
                continue
    if (not opt_map) and X_in is not None:
        try:
            row0 = X_in.iloc[0]
            for name in X_in.columns:
                if name in reg_cols:
                    try:
                        opt_map[name] = float(row0[name])
                    except Exception:
                        continue
        except Exception:
            pass

    new_row = {}
    for c in reg_cols:
        if c in opt_map:
            new_row[c] = opt_map[c]
        else:
            new_row[c] = row_reg.get(c, np.nan)
    return pd.DataFrame([new_row], columns=reg_cols)


def run_one(pid: int, budget: float, reg_model_path: str, time_limit: float | None) -> dict:
    base = get_base_house(pid)
    base_row = base.row
    ct = costs.CostTables()
    bundle = XGBBundle()

    precio_base = precio_base_from_csv(pid)
    if precio_base is None:
        X_base = build_base_input_row(bundle, base_row)
        precio_base = float(bundle.predict(X_base).iloc[0])

    m = build_mip_embed(base_row, budget, ct, bundle, base_price=precio_base)
    tl = time_limit if time_limit is not None else PARAMS.time_limit
    m.Params.TimeLimit = float(tl)
    m.Params.Presolve = 0
    m.Params.InfUnbdInfo = 1
    m.Params.NumericFocus = 3
    m.Params.ScaleFlag = 1
    m.Params.FeasibilityTol = 1e-7
    m.Params.IntFeasTol = 1e-7
    m.optimize()

    status = int(getattr(m, "Status", -1))
    X_in = rebuild_embed_input_df(m, m._X_base_numeric)
    precio_xgb = float(bundle.predict(X_in).iloc[0])

    reg_model = joblib.load(reg_model_path)
    X_reg = build_reg_input(pid, reg_model, base_row, m, X_in)
    try:
        reg_pred = reg_model.predict(X_reg)
        precio_reg = float(np.exp(reg_pred[0]))
    except Exception:
        precio_reg = float("nan")

    max_slack = float("nan")
    y_price_opt = float("nan")
    try:
        max_slack = max(abs(c.Slack) for c in m.getConstrs())
    except Exception:
        pass
    try:
        y_price_opt = float(getattr(m, "_y_price_var", None).X) if getattr(m, "_y_price_var", None) is not None else float("nan")
    except Exception:
        pass

    return {
        "pid": pid,
        "budget": budget,
        "status": status,
        "price_base": precio_base,
        "price_xgb": precio_xgb,
        "price_reg": precio_reg,
        "delta": precio_xgb - precio_reg if np.isfinite(precio_reg) else float("nan"),
        "delta_pct": ((precio_xgb - precio_reg)/precio_reg*100) if (precio_reg and np.isfinite(precio_reg) and precio_reg!=0) else float("nan"),
        "y_price_opt": y_price_opt,
        "max_slack": max_slack,
        "runtime_s": float(getattr(m, "Runtime", float("nan"))),
        "mip_gap": float(getattr(m, "MIPGap", float("nan"))),
    }


def main(benchmark_path: str, reg_model_path: str, time_limit: float | None, max_rows: int | None, out_csv: str):
    df = pd.read_csv(benchmark_path)
    rows = []
    for i, r in df.iterrows():
        if max_rows is not None and len(rows) >= max_rows:
            break
        pid = int(r.get("pid"))
        budget = float(r.get("budget"))
        try:
            res = run_one(pid, budget, reg_model_path, time_limit)
            res["ok"] = True
            res["error"] = ""
        except Exception as e:
            res = {
                "pid": pid,
                "budget": budget,
                "ok": False,
                "error": str(e),
                "price_base": float("nan"),
                "price_xgb": float("nan"),
                "price_reg": float("nan"),
                "delta": float("nan"),
                "delta_pct": float("nan"),
                "y_price_opt": float("nan"),
                "max_slack": float("nan"),
                "runtime_s": float("nan"),
                "mip_gap": float("nan"),
                "status": -1,
            }
        rows.append(res)
        print(f"[{len(rows)}/{len(df)}] pid={pid} budget={budget:,.0f} ok={res.get('ok')}")

    out = pd.DataFrame(rows)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Guardado {len(out)} filas en {out_path}")

    # Resumen por presupuesto (promedios de diferencias)
    try:
        gb = out.groupby("budget")
        summary = gb.agg(
            n=("pid", "count"),
            delta_mean=("delta", "mean"),
            delta_pct_mean=("delta_pct", "mean"),
            price_xgb_mean=("price_xgb", "mean"),
            price_reg_mean=("price_reg", "mean"),
        ).reset_index()
        print("\n-- Promedio de diferencias por presupuesto --")
        for _, r in summary.iterrows():
            print(f"budget={r['budget']:>8,.0f} | n={int(r['n'])} | Δ$={r['delta_mean']:>10.1f} | Δ%={r['delta_pct_mean']:>7.2f}%")
    except Exception as e:
        print(f"[WARN] No se pudo calcular resumen por presupuesto: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=str, default="bench_out/remodel_benchmark.csv", help="CSV con resultados de benchmark (pid,budget,...)")
    ap.add_argument("--reg-model", required=True, help="Ruta al joblib de la regresión base")
    ap.add_argument("--time-limit", type=float, default=None, help="TimeLimit para cada corrida MIP")
    ap.add_argument("--max-rows", type=int, default=None, help="Máximo de filas a procesar")
    ap.add_argument("--out-csv", type=str, default="bench_out/remodel_compare_batch.csv", help="Ruta de salida para los resultados")
    args = ap.parse_args()

    main(args.benchmark, args.reg_model, args.time_limit, args.max_rows, args.out_csv)
