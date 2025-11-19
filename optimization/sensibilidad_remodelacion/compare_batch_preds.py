"""
Compara predicciones de XGB vs Regresión lineal para las casas generadas
en el análisis de sensibilidad (lee `detalles.jsonl`).

Para cada fila OPTIMAL:
 - Usa `extra.opt_row` (atributos "raw" de la casa óptima) para predecir con:
     * XGBBundle (mismo modelo de remodel)
     * Regresión lineal base (entrenada al vuelo con df_final_regresion)
 - Genera un CSV con columnas: pid, budget, percentile, quality_floor, status,
   xgb_pred, reg_pred, diff_reg_minus_xgb.

Uso:
  python3 optimization/sensibilidad_remodelacion/compare_batch_preds.py \
      --detalles optimization/sensibilidad_remodelacion/detalles.jsonl \
      --basecsv data/processed/base_completa_sin_nulos.csv \
      --regcsv data/raw/df_final_regresion.csv \
      --out optimization/sensibilidad_remodelacion/pred_comparison.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# asegurar repo en sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from optimization.remodel.xgb_predictor import (
    XGBBundle,
    _coerce_quality_ordinals_inplace,
    _coerce_utilities_ordinal_inplace,
)
from optimization.remodel.run_opt import _row_with_dummies


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detalles", type=Path, required=True)
    ap.add_argument("--basecsv", type=Path, default=Path("data/processed/base_completa_sin_nulos.csv"))
    ap.add_argument("--regcsv", type=Path, default=Path("data/raw/df_final_regresion.csv"))
    ap.add_argument("--out", type=Path, default=Path("optimization/sensibilidad_remodelacion/pred_comparison.csv"))
    return ap.parse_args()


def build_regression_template(X: pd.DataFrame) -> Pipeline:
    cat_cols = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),
                                   ("scaler", StandardScaler())]), num_cols),
            ("cat", Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                   ("encoder", ohe)]), cat_cols),
        ]
    )
    return Pipeline(steps=[("preprocess", pre), ("model", LinearRegression())])


def align_sources(csv_xgb: Path, csv_reg: Path, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_xgb = pd.read_csv(csv_xgb)
    df_reg = pd.read_csv(csv_reg)
    if "PID" not in df_xgb or "PID" not in df_reg:
        raise ValueError("Ambos CSV deben tener PID")
    mask = df_xgb["PID"].isin(df_reg["PID"])
    df_xgb = df_xgb.loc[mask].reset_index(drop=True)
    df_reg = df_reg.set_index("PID").loc[df_xgb["PID"]].reset_index()
    if target in df_xgb.columns:
        df_reg[target] = df_xgb[target].to_numpy()
    return df_xgb, df_reg


def main():
    args = parse_args()
    target = "SalePrice_Present"

    # Entrenar regresión base
    df_xgb, df_reg = align_sources(args.basecsv, args.regcsv, target)
    y_reg = df_reg[target]
    X_reg = df_reg.drop(columns=[c for c in ["PID", "Order", "SalePrice", target, "\ufeffOrder"] if c in df_reg.columns])
    reg_pipe = build_regression_template(X_reg)
    reg_pipe.fit(X_reg, y_reg)

    # Bundle XGB
    bundle = XGBBundle()
    feat_order = bundle.feature_names_in()

    rows: List[dict] = []
    with args.detalles.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            meta = rec.get("meta", {})
            extra = rec.get("extra", {})
            if meta.get("status") != "OPTIMAL":
                continue
            opt_row = extra.get("opt_row", {})
            if not opt_row:
                continue

            # ---- Pred XGB con features dummy ----
            X_opt = pd.DataFrame([_row_with_dummies(pd.Series(opt_row), feat_order)], columns=feat_order)
            _coerce_quality_ordinals_inplace(X_opt, bundle.quality_cols)
            _coerce_utilities_ordinal_inplace(X_opt)
            xgb_pred = float(bundle.predict(X_opt).iloc[0])

            # ---- Pred Regresión con raw opt_row ----
            row_reg = pd.DataFrame([opt_row])
            # asegurar mismas columnas que X_reg (faltantes -> NaN)
            for col in X_reg.columns:
                if col not in row_reg.columns:
                    row_reg[col] = pd.NA
            row_reg = row_reg[X_reg.columns]
            # Asegurar tipos: strings para categorías, num para numéricas
            for c in X_reg.columns:
                if X_reg[c].dtype == object:
                    row_reg[c] = row_reg[c].astype(str)
                else:
                    row_reg[c] = pd.to_numeric(row_reg[c], errors="coerce")
            row_reg = row_reg.fillna(value=pd.NA)
            reg_pred = float(reg_pipe.predict(row_reg)[0])

            rows.append({
                "pid": meta.get("pid"),
                "neighborhood": meta.get("neighborhood"),
                "percentile_label": meta.get("percentile_label"),
                "budget": meta.get("budget"),
                "quality_floor": meta.get("quality_floor"),
                "status": meta.get("status"),
                "base_price": meta.get("base_price"),
                "xgb_pred": xgb_pred,
                "reg_pred": reg_pred,
                "diff_reg_minus_xgb": reg_pred - xgb_pred,
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"Guardado resumen en {args.out} ({len(out_df)} filas).")


if __name__ == "__main__":
    main()
