"""
Predice el valor de una casa puntual con:
 - El XGBoost entrenado (bundle de remodel).
 - Una regresión lineal base (como en analysis/xgb_vs_regression).

Uso:
  python3 optimization/sensibilidad_remodelacion/compare_xgb_vs_reg.py \
      --pid 907290230 \
      --basecsv data/processed/base_completa_sin_nulos.csv \
      --regcsv data/raw/df_final_regresion.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# añadir repo root al path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from optimization.remodel.xgb_predictor import (
    XGBBundle,
    _coerce_quality_ordinals_inplace,
    _coerce_utilities_ordinal_inplace,
)
from optimization.remodel.run_opt import _row_with_dummies
from optimization.remodel.io import get_base_house
from optimization.remodel.config import PATHS


def build_regression_template(X: pd.DataFrame) -> Pipeline:
    cat_cols = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),
                                   ("scaler", StandardScaler())]), num_cols),
            ("cat", Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                   ("encoder", ohe)]), cat_cols),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocessor),
                           ("model", LinearRegression())])


def align_sources(csv_xgb: Path, csv_reg: Path, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_xgb = pd.read_csv(csv_xgb)
    df_reg = pd.read_csv(csv_reg)
    if "PID" not in df_xgb or "PID" not in df_reg:
        raise ValueError("Ambos CSV deben tener PID")

    common_mask = df_xgb["PID"].isin(df_reg["PID"])
    df_xgb = df_xgb.loc[common_mask].reset_index(drop=True)
    df_reg = df_reg.set_index("PID").loc[df_xgb["PID"]].reset_index()

    if target in df_xgb.columns:
        df_reg[target] = df_xgb[target].to_numpy()
    return df_xgb, df_reg


def predict_for_pid(pid: int, basecsv: Path, regcsv: Path) -> None:
    target = "SalePrice_Present"

    # --- XGB (bundle ya entrenado) ---
    bundle = XGBBundle()
    base_house = get_base_house(pid, base_csv=basecsv)
    feat_order = bundle.feature_names_in()
    X_row = pd.DataFrame([_row_with_dummies(base_house.row, feat_order)], columns=feat_order)
    _coerce_quality_ordinals_inplace(X_row, bundle.quality_cols)
    _coerce_utilities_ordinal_inplace(X_row)
    xgb_pred = float(bundle.predict(X_row).iloc[0])

    # --- Regresión lineal entrenada al vuelo (dataset de regresión) ---
    df_xgb, df_reg = align_sources(basecsv, regcsv, target)
    if pid not in df_reg["PID"].values:
        raise ValueError(f"PID {pid} no está en el CSV de regresión")

    y_reg = df_reg[target]
    X_reg = df_reg.drop(columns=[c for c in ["PID", "Order", "SalePrice", target, "\ufeffOrder"] if c in df_reg.columns])
    reg_pipeline = build_regression_template(X_reg)
    reg_pipeline.fit(X_reg, y_reg)

    row_reg = X_reg[df_reg["PID"] == pid]
    reg_pred = float(reg_pipeline.predict(row_reg)[0])

    print(f"PID {pid}")
    print(f"  XGB model (bundle):      ${xgb_pred:,.2f}")
    print(f"  Linear regression base:  ${reg_pred:,.2f}")
    print(f"  Precio real (SalePrice_Present): ${float(base_house.row[target]):,.2f}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--basecsv", type=Path, default=PATHS.base_csv)
    ap.add_argument("--regcsv", type=Path, default=Path("data/raw/df_final_regresion.csv"))
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict_for_pid(args.pid, args.basecsv, args.regcsv)
