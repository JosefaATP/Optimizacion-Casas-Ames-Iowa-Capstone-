#!/usr/bin/env python3
"""
Comparación detallada entre el XGBoost entrenado con los mejores hiperparámetros
y una regresión lineal base en los 100 folds (10 repeticiones x 10 folds) del
esquema de validación cruzada usado en entrenamiento.

El script:
  * Reproduce los mismos splits con `KFold` y semilla desplazada por repetición.
  * Entrena ambos modelos en cada fold y guarda las predicciones por vivienda.
  * Calcula R2, MAPE, RMSE, MAE y el porcentaje de casos en que XGBoost gana.
  * Genera reportes agregados por fold, por repetición y un resumen global.

Uso:
    python analysis/xgb_vs_regression/scripts/cv10_compare_baselines.py \\
        --csv-xgb data/processed/base_completa_sin_nulos.csv \\
        --csv-reg data/raw/df_final_regresion.csv \\
        --bayes-summary models/xgb_bayes_search/bayes_summary.json \\
        --target SalePrice_Present \\
        --outdir analysis/xgb_vs_regression
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

SEED = 42
DROP_COLS = ["PID", "Order", "SalePrice", "\ufeffOrder"]
DEFAULT_TARGET = "SalePrice_Present"
DEFAULT_XGB_CSV = "data/processed/base_completa_sin_nulos.csv"
DEFAULT_REG_CSV = "data/raw/df_final_regresion.csv"
DEFAULT_BAYES_SUMMARY = "models/xgb_bayes_search/bayes_summary.json"


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-12
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(
        np.mean(
            np.abs((y_true - y_pred) / np.clip(np.abs(y_true), eps, None))
        )
        * 100.0
    )


def load_best_params(bayes_summary_path: Path) -> Dict:
    summary = json.loads(Path(bayes_summary_path).read_text())
    best = summary["best"]["params"]
    params = {
        **best,
        "objective": "reg:squarederror",
        "random_state": SEED,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    return params


def align_sources(
    csv_xgb: Path, csv_reg: Path, target: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_xgb = pd.read_csv(csv_xgb)
    df_reg = pd.read_csv(csv_reg)

    if "PID" not in df_xgb.columns or "PID" not in df_reg.columns:
        raise ValueError("Ambas fuentes deben contener la columna PID para alinear.")

    common = df_xgb["PID"].isin(df_reg["PID"])
    if not common.all():
        missing = df_xgb.loc[~common, "PID"]
        print(
            f"⚠️  Se excluirán {len(missing)} filas de {csv_xgb} porque no "
            "aparecen en df_final_regresion."
        )
        df_xgb = df_xgb.loc[common].reset_index(drop=True)

    df_reg = df_reg.set_index("PID").loc[df_xgb["PID"]]
    df_reg = df_reg.reset_index()

    if not np.allclose(df_xgb[target], df_reg[target]):
        print(
            "⚠️  'SalePrice_Present' difiere entre las fuentes. "
            "Se usará el valor de base_completa_sin_nulos.csv para ambos modelos."
        )
        df_reg[target] = df_xgb[target].to_numpy()

    return df_xgb.reset_index(drop=True), df_reg.reset_index(drop=True)


def prepare_xgb_features(df: pd.DataFrame, target: str, params: Dict) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    working = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore").copy()
    X = working.drop(columns=[target])
    y = working[target]

    params = dict(params)
    non_numeric = X.select_dtypes(exclude=["number", "bool"]).columns
    if len(non_numeric) > 0:
        X[non_numeric] = X[non_numeric].astype("category")
        params["enable_categorical"] = True
    else:
        params.pop("enable_categorical", None)

    return X, y, params


def build_regression_template(X: pd.DataFrame) -> Pipeline:
    cat_cols = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()

    def build_ohe():
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            build_ohe(),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LinearRegression()),
        ]
    )


def prepare_regression_features(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    working = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore").copy()
    y = working[target]
    X = working.drop(columns=[target])
    return X, y


def evaluate_models(
    y_true: np.ndarray,
    y_pred_xgb: np.ndarray,
    y_pred_reg: np.ndarray,
) -> Dict[str, float]:
    mse_xgb = mean_squared_error(y_true, y_pred_xgb)
    mse_reg = mean_squared_error(y_true, y_pred_reg)
    metrics = {
        "R2_xgb": r2_score(y_true, y_pred_xgb),
        "R2_reg": r2_score(y_true, y_pred_reg),
        "RMSE_xgb": float(mse_xgb ** 0.5),
        "RMSE_reg": float(mse_reg ** 0.5),
        "MAE_xgb": mean_absolute_error(y_true, y_pred_xgb),
        "MAE_reg": mean_absolute_error(y_true, y_pred_reg),
        "MAPE_xgb": mape(y_true, y_pred_xgb),
        "MAPE_reg": mape(y_true, y_pred_reg),
    }
    return metrics


def run_experiment(
    csv_xgb: Path,
    csv_reg: Path,
    bayes_summary: Path,
    target: str,
    outdir: Path,
    n_splits: int = 10,
    repeats: int = 10,
) -> None:
    outdir = Path(outdir)
    reports_dir = outdir / "reports"
    preds_dir = reports_dir / "fold_predictions"
    reports_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    df_xgb, df_reg = align_sources(csv_xgb, csv_reg, target)
    xgb_params = load_best_params(bayes_summary)
    X_xgb, y, xgb_params = prepare_xgb_features(df_xgb, target, xgb_params)
    X_reg, _ = prepare_regression_features(df_reg, target)
    pid_series = df_xgb["PID"].reset_index(drop=True)

    reg_template = build_regression_template(X_reg)

    fold_rows: List[Dict] = []
    total_cases = 0
    total_xgb_wins = 0
    total_reg_wins = 0

    for rep in range(1, repeats + 1):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED + rep)
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_xgb), start=1):
            model_xgb = XGBRegressor(**xgb_params)
            model_xgb.fit(X_xgb.iloc[train_idx], y.iloc[train_idx])

            model_reg = clone(reg_template)
            model_reg.fit(X_reg.iloc[train_idx], y.iloc[train_idx])

            y_true = y.iloc[test_idx].to_numpy()
            preds_xgb = model_xgb.predict(X_xgb.iloc[test_idx])
            preds_reg = model_reg.predict(X_reg.iloc[test_idx])

            abs_xgb = np.abs(y_true - preds_xgb)
            abs_reg = np.abs(y_true - preds_reg)
            xgb_wins = abs_xgb < abs_reg
            reg_wins = abs_reg < abs_xgb
            win_rate = xgb_wins.mean() * 100.0

            total_cases += len(test_idx)
            total_xgb_wins += int(xgb_wins.sum())
            total_reg_wins += int(reg_wins.sum())

            metrics = evaluate_models(y_true, preds_xgb, preds_reg)
            metrics.update(
                {
                    "rep": rep,
                    "fold": fold,
                    "pct_xgb_wins": win_rate,
                    "mean_abs_err_xgb": float(abs_xgb.mean()),
                    "mean_abs_err_reg": float(abs_reg.mean()),
                    "mean_abs_err_diff": float(abs_reg.mean() - abs_xgb.mean()),
                }
            )
            fold_rows.append(metrics)

            fold_pred_df = pd.DataFrame(
                {
                    "PID": pid_series.iloc[test_idx].to_numpy(),
                    "y_true": y_true,
                    "y_pred_xgb": preds_xgb,
                    "y_pred_regression": preds_reg,
                    "abs_err_xgb": abs_xgb,
                    "abs_err_regression": abs_reg,
                    "xgb_wins": xgb_wins,
                }
            )
            fold_pred_df.to_csv(
                preds_dir / f"rep_{rep:02d}_fold_{fold:02d}.csv", index=False
            )

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(reports_dir / "fold_scores.csv", index=False)

    agg_cols = {
        col: "mean"
        for col in fold_df.columns
        if col not in {"rep", "fold"}
    }
    repeat_df = (
        fold_df.groupby("rep", as_index=False)
        .agg(agg_cols)
        .rename(columns={"pct_xgb_wins": "pct_xgb_wins_mean"})
    )
    repeat_df.to_csv(reports_dir / "repeat_means.csv", index=False)

    summary = {
        "setup": {
            "n_splits": n_splits,
            "repeats": repeats,
            "seed_base": SEED,
            "csv_xgb": str(csv_xgb),
            "csv_reg": str(csv_reg),
            "bayes_summary": str(bayes_summary),
        },
        "metrics": {},
    }

    metric_names = ["R2", "MAPE", "RMSE", "MAE", "mean_abs_err"]
    for metric in metric_names:
        if metric == "mean_abs_err":
            xgb_col = "mean_abs_err_xgb"
            reg_col = "mean_abs_err_reg"
        else:
            xgb_col = f"{metric}_xgb"
            reg_col = f"{metric}_reg"

        summary["metrics"][metric] = {
            "xgboost": {
                "mean": float(repeat_df[xgb_col].mean()),
                "sd": float(repeat_df[xgb_col].std(ddof=1)),
            },
            "regression": {
                "mean": float(repeat_df[reg_col].mean()),
                "sd": float(repeat_df[reg_col].std(ddof=1)),
            },
        }

    summary["xgb_win_rate"] = {
        "mean_per_fold": float(fold_df["pct_xgb_wins"].mean()),
        "std_per_fold": float(fold_df["pct_xgb_wins"].std(ddof=1)),
        "overall_weighted": (total_xgb_wins / total_cases) * 100.0,
    }
    summary["regression_win_rate"] = {
        "overall_weighted": (total_reg_wins / total_cases) * 100.0,
    }

    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("✅ Comparación completada.")
    print(f"- Guardado {len(fold_df)} filas en {reports_dir/'fold_scores.csv'}")
    print(f"- Guardado {len(repeat_df)} filas en {reports_dir/'repeat_means.csv'}")
    print(f"- Guardado resumen en {reports_dir/'summary.json'}")
    print(json.dumps(summary["xgb_win_rate"], indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara XGBoost vs regresión lineal en CV 10x10."
    )
    parser.add_argument(
        "--csv-xgb",
        default=DEFAULT_XGB_CSV,
        help="CSV usado por XGBoost (base_completa).",
    )
    parser.add_argument(
        "--csv-reg",
        default=DEFAULT_REG_CSV,
        help="CSV usado por la regresión (df_final).",
    )
    parser.add_argument(
        "--bayes-summary",
        default=DEFAULT_BAYES_SUMMARY,
        help="Resumen JSON con los mejores hiperparámetros del XGB.",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help="Nombre de la columna objetivo compartida.",
    )
    parser.add_argument(
        "--outdir",
        default="analysis/xgb_vs_regression",
        help="Carpeta raíz donde se guardarán reportes y docs.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=10,
        help="Número de folds por repetición.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Número de repeticiones del CV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        csv_xgb=Path(args.csv_xgb),
        csv_reg=Path(args.csv_reg),
        bayes_summary=Path(args.bayes_summary),
        target=args.target,
        outdir=Path(args.outdir),
        n_splits=args.n_splits,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    main()
