import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_pipeline(numeric_cols, categorical_cols):
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    base_reg = LinearRegression()
    reg = TransformedTargetRegressor(
        regressor=base_reg,
        func=np.log,
        inverse_func=np.exp,
        check_inverse=False,
    )
    pipe = Pipeline(steps=[("pre", pre), ("reg", reg)])
    return pipe


def main(csv_path: str, out_path: str, test_size: float, random_state: int):
    df = pd.read_csv(csv_path)
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]

    target = "SalePrice_Present"
    drop_cols = [col for col in ("PID", "SalePrice", target) if col in df.columns]
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {csv_path}")

    y = df[target].astype(float)
    X = df.drop(columns=drop_cols)

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipe = build_pipeline(numeric_cols, categorical_cols)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"[REG] test R2={r2:.4f} | MAE={mae:,.2f}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(pipe, out_path)
    print(f"[REG] modelo guardado en {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Entrena regresión lineal para construcción")
    ap.add_argument("--csv", default="data/raw/df_final_regresion.csv", help="Ruta al CSV limpio")
    ap.add_argument("--out", default="models/reg/base_reg.joblib", help="Ruta de salida del joblib")
    ap.add_argument("--test-size", type=float, default=0.2, help="Proporción para test split")
    ap.add_argument("--seed", type=int, default=42, help="Semilla para train_test_split")
    args = ap.parse_args()
    main(args.csv, args.out, args.test_size, args.seed)
