
import os, json, argparse, numpy as np, pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

SEED = 42

def _mape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    eps = 1e-12
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), eps, None))) * 100.0

def load_params(params_json=None, bayes_summary_json=None):
    if params_json:
        with open(params_json, "r") as f: return json.load(f)
    if bayes_summary_json:
        with open(bayes_summary_json, "r") as f:
            best = json.load(f)["best"]["params"]
            return best
    raise ValueError("Debes pasar --params-json o --bayes-summary.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Ruta al CSV limpio.")
    ap.add_argument("--target", default="SalePrice", help="Nombre de la columna target.")
    ap.add_argument("--features", default=None, help="Ruta a un .txt con nombres de features (opcional).")
    ap.add_argument("--params-json", default=None, help="JSON con hiperparámetros (dict).")
    ap.add_argument("--bayes-summary", default=None, help="bayes_summary.json para tomar 'best.params'.")
    ap.add_argument("--outdir", default="models/xgb_bayes_search_2/repeated_cv", help="Salida.")
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--n-repeats", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    if args.features:
        feats = [l.strip() for l in open(args.features).read().splitlines() if l.strip()]
        X = df[feats].copy()
    else:
        X = df.drop(columns=[args.target]).copy()
    y = df[args.target].values

    params = load_params(args.params_json, args.bayes_summary)

    params.setdefault("objective", "reg:squarederror")
    params.setdefault("random_state", SEED)
    params.setdefault("tree_method", "hist")
    params.setdefault("n_jobs", -1)

    rkf = RepeatedKFold(n_splits=args.n_splits, n_repeats=args.n_repeats, random_state=SEED)

    fold_rows = []
    rep_means = []
    rep_idx = -1
    last_rep_seen = -1

    fold_counter = 0
    for tr_idx, te_idx in rkf.split(X, y):
        rep = fold_counter // args.n_splits  
        fold = fold_counter %  args.n_splits  
        fold_counter += 1

        model = XGBRegressor(**params)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        yhat = model.predict(X.iloc[te_idx])

        r2   = r2_score(y[te_idx], yhat)
        rmse = mean_squared_error(y[te_idx], yhat, squared=False)
        mae  = mean_absolute_error(y[te_idx], yhat)
        mape = _mape(y[te_idx], yhat)

        fold_rows.append({"repeat": rep+1, "fold": fold+1,
                          "R2": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape})

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(os.path.join(args.outdir, "fold_scores.csv"), index=False)

    rep_df = fold_df.groupby("repeat").agg({"R2":"mean","RMSE":"mean","MAE":"mean","MAPE":"mean"}) \
                    .reset_index().rename(columns={"R2":"R2_mean","RMSE":"RMSE_mean",
                                                   "MAE":"MAE_mean","MAPE":"MAPE_mean"})
    rep_df.to_csv(os.path.join(args.outdir, "repeat_means.csv"), index=False)

    summary = {
        "params": params,
        "n_splits": args.n_splits,
        "n_repeats": args.n_repeats,
        "R2":   {"mean": float(rep_df["R2_mean"].mean()),   "sd": float(rep_df["R2_mean"].std(ddof=1))},
        "RMSE": {"mean": float(rep_df["RMSE_mean"].mean()), "sd": float(rep_df["RMSE_mean"].std(ddof=1))},
        "MAE":  {"mean": float(rep_df["MAE_mean"].mean()),  "sd": float(rep_df["MAE_mean"].std(ddof=1))},
        "MAPE": {"mean": float(rep_df["MAPE_mean"].mean()), "sd": float(rep_df["MAPE_mean"].std(ddof=1))}
    }
    with open(os.path.join(args.outdir, "repeated_cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("✅ Guardados:")
    print(" - fold_scores.csv")
    print(" - repeat_means.csv (10 promedios, uno por repetición)")
    print(" - repeated_cv_summary.json (media ± sd de esos 10 promedios)")

if __name__ == "__main__":
    main()
