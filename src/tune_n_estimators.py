import json, argparse, numpy as np, pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from config import Config
from preprocess import infer_feature_types, build_preprocessor
from metrics import regression_report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--grid", default="1200,1800,2400,3000,3600,4200")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--log_target", action="store_true")
    args = ap.parse_args()

    cfg = Config(); cfg.target = args.target
    df = pd.read_csv(args.csv, sep=None, engine="python")
    df.columns = [c.replace("\ufeff","").strip() for c in df.columns]
    df[cfg.target] = pd.to_numeric(df[cfg.target], errors="coerce")
    df = df.dropna(subset=[cfg.target])

    num, cat = infer_feature_types(df, target=cfg.target, drop_cols=cfg.drop_cols,
                                   numeric_cols=cfg.numeric_cols, categorical_cols=cfg.categorical_cols)

    X = df.drop(columns=[cfg.target] + [c for c in cfg.drop_cols if c in df.columns], errors="ignore")
    y = df[cfg.target].values

    grid = [int(x) for x in args.grid.split(",")]
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)

    results = []
    for n in grid:
        params = cfg.xgb_params.copy()
        params["n_estimators"] = n
        mape_scores = []

        for tr, va in kf.split(X):
            X_tr, X_va = X.iloc[tr], X.iloc[va]
            y_tr, y_va = y[tr], y[va]

            pre = build_preprocessor(num, cat)
            pre.fit(X_tr)
            X_tr_pre = pre.transform(X_tr)
            X_va_pre = pre.transform(X_va)

            model = XGBRegressor(**params)

            if args.log_target:
                y_tr_fit = np.log1p(y_tr)
                model.fit(X_tr_pre, y_tr_fit, verbose=False)
                yhat = np.expm1(model.predict(X_va_pre))
            else:
                model.fit(X_tr_pre, y_tr, verbose=False)
                yhat = model.predict(X_va_pre)

            rep = regression_report(y_va, yhat)
            mape_scores.append(rep["MAPE_pct"])

        res = {"n_estimators": n,
               "MAPE_cv_mean": float(np.mean(mape_scores)),
               "MAPE_cv_std": float(np.std(mape_scores))}
        results.append(res)
        print(res)

    best = min(results, key=lambda r: r["MAPE_cv_mean"])
    print("\nBEST:", best)
    with open("models/xgb/tuning_n_estimators.json", "w") as f:
        json.dump({"grid": grid, "folds": args.folds, "results": results, "best": best}, f, indent=2)

if __name__ == "__main__":
    main()
