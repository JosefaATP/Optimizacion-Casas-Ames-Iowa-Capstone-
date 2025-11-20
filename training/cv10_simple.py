
import argparse, json, numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from pathlib import Path

SEED = 42

BEST_PARAMS = {
    "n_estimators": 3758,
    "learning_rate": 0.03683247703306933,
    "max_depth": 4,
    "min_child_weight": 5.0,
    "gamma": 0.008345270600629286,
    "subsample": 0.6783424963567025,
    "colsample_bytree": 0.43051657642562313,
    "reg_lambda": 3.364253417277695,
    "reg_alpha": 0.052422719530854305,
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "n_jobs": -1,
    "random_state": SEED,
}

DROP_COLS = ["PID", "Order", "SalePrice", "\ufeffOrder"]

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    eps = 1e-12
    return float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), eps, None))) * 100.0)

def load_data(csv_path: str, target_col: str, one_hot: bool):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if one_hot:
        X = pd.get_dummies(X, drop_first=False)
        params = dict(BEST_PARAMS)
        params.pop("enable_categorical", None)
    else:
        non_numeric = X.select_dtypes(exclude=["number", "bool"]).columns
        X[non_numeric] = X[non_numeric].astype("category")
        params = dict(BEST_PARAMS)
        params["enable_categorical"] = True

    return X, y, params

def run_cv(csv, target, outdir, n_splits=10, repeats=10, one_hot=False):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    X, y, params = load_data(csv, target, one_hot=one_hot)

    fold_rows = []


    for rep in range(1, repeats + 1):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED + rep)
        for fold, (tr, te) in enumerate(kf.split(X), start=1):
            model = XGBRegressor(**params)
            model.fit(X.iloc[tr], y.iloc[tr])
            yp = model.predict(X.iloc[te])


            rmse = mean_squared_error(y.iloc[te], yp) ** 0.5

            fold_rows.append({
                "rep": rep,
                "fold": fold,
                "R2": r2_score(y.iloc[te], yp),
                "RMSE": rmse,
                "MAE": mean_absolute_error(y.iloc[te], yp),
                "MAPE": mape(y.iloc[te], yp),
            })

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(f"{outdir}/fold_scores.csv", index=False)


    rep_df = fold_df.groupby("rep", as_index=False).agg({
        "R2": "mean", "RMSE": "mean", "MAE": "mean", "MAPE": "mean"
    }).rename(columns={"R2": "R2_mean", "RMSE": "RMSE_mean", "MAE": "MAE_mean", "MAPE": "MAPE_mean"})
    rep_df.to_csv(f"{outdir}/repeat_means.csv", index=False)


    summary = {
        "R2":   {"mean": float(rep_df["R2_mean"].mean()),   "sd": float(rep_df["R2_mean"].std(ddof=1))},
        "RMSE": {"mean": float(rep_df["RMSE_mean"].mean()), "sd": float(rep_df["RMSE_mean"].std(ddof=1))},
        "MAE":  {"mean": float(rep_df["MAE_mean"].mean()),  "sd": float(rep_df["MAE_mean"].std(ddof=1))},
        "MAPE": {"mean": float(rep_df["MAPE_mean"].mean()), "sd": float(rep_df["MAPE_mean"].std(ddof=1))},
        "n_splits": int(n_splits),
        "repeats": int(repeats),
        "seed_base": SEED,
        "one_hot": bool(one_hot),
    }
    with open(f"{outdir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("✅ Listo")
    print(f"- Guardado: {outdir}/fold_scores.csv  (filas = {len(fold_df)})")
    print(f"- Guardado: {outdir}/repeat_means.csv (filas = {len(rep_df)})")
    print(f"- Guardado: {outdir}/summary.json")
    print(pd.DataFrame(summary).T)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", default="SalePrice_Present")
    ap.add_argument("--outdir", default="models/xgb_bayes_search_2/cv10x10")
    ap.add_argument("--n-splits", type=int, default=10, help="Folds por repetición (K).")
    ap.add_argument("--repeats", type=int, default=10, help="Número de repeticiones (M).")
    ap.add_argument("--one-hot", action="store_true", help="Usa one-hot en vez de soporte categórico nativo.")
    args = ap.parse_args()
    run_cv(args.csv, args.target, args.outdir, n_splits=args.n_splits, repeats=args.repeats, one_hot=args.one_hot)

if __name__ == "__main__":
    main()
