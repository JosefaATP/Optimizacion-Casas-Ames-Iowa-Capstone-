# scripts/compare_predictors.py
"""
Compara la predicción del XGB productivo vs una regresión base externa sobre la casa remodelada.

Modos:
- Por defecto resuelve la remodelación (MIP) y compara.
- Si pasas --xin-csv (ej. X_input_after_opt.csv) solo usa esa fila óptima y NO resuelve.
"""

import argparse
import joblib
import numpy as np
import pandas as pd
import gurobipy as gp

from optimization.remodel.io import get_base_house
from optimization.remodel import costs
from optimization.remodel.xgb_predictor import XGBBundle
from optimization.remodel.gurobi_model import build_mip_embed, build_base_input_row
from optimization.remodel.run_opt import rebuild_embed_input_df
from optimization.remodel.config import PARAMS


def _load_x_input_from_csv(path: str) -> pd.DataFrame:
    """Carga X_input_after_opt.csv (formato idx,feature,value,LB,UB) y devuelve DF 1xN."""
    df = pd.read_csv(path)
    if {"feature", "value"}.issubset(df.columns):
        cols = df["feature"].tolist()
        vals = df["value"].tolist()
        return pd.DataFrame([vals], columns=cols)
    # fallback: si ya viene como 1 fila con columnas
    if len(df) == 1:
        return df
    raise ValueError(f"Formato inesperado en {path}")


def main(pid: int, budget: float, reg_model_path: str,
         time_limit: float | None = None, xin_csv: str | None = None):
    # Cargar insumos
    base = get_base_house(pid)
    base_row = base.row
    ct = costs.CostTables()
    bundle = XGBBundle()

    # Precio base: primero intenta usar el de la base de datos; si no, cae al XGB
    def _precio_base_from_csv(pid: int):
        try:
            df = pd.read_csv("data/raw/df_final_regresion.csv")
            df.columns = [c.replace("\ufeff","").strip() for c in df.columns]
            row = df.loc[df["PID"] == pid]
            if not row.empty and "SalePrice_Present" in row.columns:
                return float(row["SalePrice_Present"].iloc[0])
            if not row.empty and "SalePrice" in row.columns:
                return float(row["SalePrice"].iloc[0])
        except Exception:
            return None
        return None

    precio_base = _precio_base_from_csv(pid)
    if precio_base is None:
        X_base = build_base_input_row(bundle, base_row)
        precio_base = float(bundle.predict(X_base).iloc[0])

    # Si se pasa un X_input precalculado, no resolvemos el MIP
    if xin_csv:
        X_in = _load_x_input_from_csv(xin_csv)
        m = None
    else:
        # Armar y resolver el MIP
        if budget is None:
            raise ValueError("budget es obligatorio si no usas --xin-csv")
        m = build_mip_embed(base_row, budget, ct, bundle, base_price=precio_base)
        if time_limit is None:
            time_limit = PARAMS.time_limit
        m.Params.TimeLimit = float(time_limit)
        # endurecer tolerancias y numerics
        m.Params.Presolve = 0
        m.Params.InfUnbdInfo = 1
        m.Params.NumericFocus = 3
        m.Params.ScaleFlag = 1
        m.Params.FeasibilityTol = 1e-7
        m.Params.IntFeasTol = 1e-7
        m.optimize()

        if m.Status not in {gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL, gp.GRB.TIME_LIMIT}:
            raise RuntimeError(f"Modelo no resuelto (status={m.Status})")

        # Reconstruir X_in posteado al embed
        X_in = rebuild_embed_input_df(m, m._X_base_numeric)

    # Predicciones
    precio_xgb = float(bundle.predict(X_in).iloc[0])

    reg_model = joblib.load(reg_model_path)

    # --- Construir fila cruda alineada a las columnas de la regresión (si expone feature_names_in_) ---
    try:
        reg_cols = list(getattr(reg_model, "feature_names_in_", []))
    except Exception:
        reg_cols = []

    if reg_cols:
        # Usa el row del CSV de regresión como base, y sobreescribe con valores óptimos cuando el nombre coincide.
        try:
            df_reg = pd.read_csv("data/raw/df_final_regresion.csv")
            df_reg.columns = [c.replace("\ufeff", "").strip() for c in df_reg.columns]
            row_reg = df_reg.loc[df_reg["PID"] == pid].iloc[0]
        except Exception:
            row_reg = pd.Series({c: base_row.get(c, np.nan) for c in reg_cols})

        # valores óptimos del MIP (solo si el nombre coincide)
        opt_map = {}
        if m is not None and hasattr(m, "_x_vars") and getattr(m, "_x_vars", None):
            for name, var in m._x_vars.items():
                try:
                    opt_map[name] = float(var.X) if hasattr(var, "X") else float(var)
                except Exception:
                    continue
        # Si no hay modelo (xin-csv) o no hay _x_vars, intenta usar valores de X_in
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
        X_reg = pd.DataFrame([new_row], columns=reg_cols)
    else:
        # fallback: intentar con X_in directo (puede fallar si faltan columnas)
        X_reg = X_in.copy()

    reg_pred = reg_model.predict(X_reg)
    # La regresión del equipo se entrenó en log(SalePrice_Present) -> llevar a dólares.
    try:
        precio_reg = float(np.exp(reg_pred[0]))
    except Exception:
        precio_reg = float(reg_pred[0])

    # Chequeos de factibilidad y precio mínimo
    if m is not None:
        try:
            max_slack = max(abs(c.Slack) for c in m.getConstrs())
        except Exception:
            max_slack = float("nan")
        if max_slack is not None and max_slack > 1e-3:
            try:
                from optimization.remodel.run_opt import debug_infeas
                debug_infeas(m, tag="remodel_compare")
            except Exception:
                pass
            raise RuntimeError(f"Solución con violación alta ({max_slack:.3e}); chequea remodel_compare.ilp/.lp")

        y_price_opt = float(getattr(m, "_y_price_var", None).X) if getattr(m, "_y_price_var", None) is not None else None
        if y_price_opt is not None and y_price_opt < precio_base - 1e-3:
            raise RuntimeError(f"La solución baja el precio base (base={precio_base:.2f}, opt={y_price_opt:.2f}). Revisa el modelo.")

    # Porcentajes
    uplift_vs_reg = (precio_xgb - precio_reg) / precio_reg * 100 if precio_reg != 0 else np.nan

    print("\n=== COMPARACIÓN DE PREDICTORES SOBRE CASA REMODELADA ===")
    budget_txt = f"${budget:,.0f}" if budget is not None else "(no especificado)"
    print(f"PID: {pid} | Presupuesto: {budget_txt}")
    print(f"Precio base (XGB):        ${precio_base:,.0f}")
    print(f"Precio remodelado XGB:    ${precio_xgb:,.0f}")
    print(f"Precio remodelado Regresión: ${precio_reg:,.0f}")
    print(f"Diferencia absoluta:      ${precio_xgb - precio_reg:,.0f}")
    print(f"Diferencia % (XGB vs Reg): {uplift_vs_reg:.2f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True, help="PID de la casa base")
    ap.add_argument("--budget", type=float, required=False, default=None, help="Presupuesto de remodelación (solo si resuelves MIP)")
    ap.add_argument("--reg-model", required=True, help="Ruta al joblib de la regresión base")
    ap.add_argument("--time-limit", type=float, default=None, help="TimeLimit del solver (segundos)")
    ap.add_argument("--xin-csv", type=str, default=None, help="CSV con X_input ya óptimo (ej. X_input_after_opt.csv) para saltar el MIP")
    args = ap.parse_args()

    main(args.pid, args.budget, args.reg_model, time_limit=args.time_limit, xin_csv=args.xin_csv)
