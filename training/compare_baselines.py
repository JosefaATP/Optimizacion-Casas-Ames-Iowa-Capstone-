import argparse, json, os, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from config import Config
from preprocess import infer_feature_types
from metrics import regression_report

def build_lin_preprocessor(numeric_cols, categorical_cols):
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--target", default=None)
    parser.add_argument("--outdir", default="models/xgb")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg = Config()
    if args.target: cfg.target = args.target

    df = pd.read_csv(args.csv)
    numeric_cols, categorical_cols = infer_feature_types(
        df, target=cfg.target, drop_cols=cfg.drop_cols,
        numeric_cols=cfg.numeric_cols, categorical_cols=cfg.categorical_cols
    )

    X = df.drop(columns=[cfg.target] + [c for c in cfg.drop_cols if c in df.columns], errors="ignore")
    y = df[cfg.target].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.random_state)

    pre = build_lin_preprocessor(numeric_cols, categorical_cols)

    models = {
        "Linear": LinearRegression(),
        "Lasso": Lasso(alpha=0.001, max_iter=10000)
    }

    results = {}
    for name, mdl in models.items():
        pipe = Pipeline([("pre", pre), (name, mdl)])
        pipe.fit(X_tr, y_tr)
        yhat = pipe.predict(X_te)
        results[name] = regression_report(y_te, yhat)

    print(results)
    with open(os.path.join(args.outdir, "baselines.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
