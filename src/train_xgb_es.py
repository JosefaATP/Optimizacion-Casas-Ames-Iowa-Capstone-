import argparse, json, os, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping
from config import Config
from preprocess import infer_feature_types, build_preprocessor
from metrics import regression_report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--target", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--log_target", action="store_true")
    ap.add_argument("--patience", type=int, default=200)   # nº de rondas sin mejora
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    cfg = Config(); cfg.target = args.target
    df = pd.read_csv(args.csv, sep=None, engine="python")
    df.columns = [c.replace("\ufeff","").strip() for c in df.columns]
    df[cfg.target] = pd.to_numeric(df[cfg.target], errors="coerce")
    df = df.dropna(subset=[cfg.target])

    num, cat = infer_feature_types(df, target=cfg.target, drop_cols=cfg.drop_cols,
                                   numeric_cols=cfg.numeric_cols, categorical_cols=cfg.categorical_cols)

    X = df.drop(columns=[cfg.target] + [c for c in cfg.drop_cols if c in df.columns], errors="ignore")
    y = df[cfg.target].values

    # split train/test y validación interna
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.random_state)
    X_tr2, X_va, y_tr2, y_va = train_test_split(X_tr, y_tr, test_size=0.15, random_state=cfg.random_state)

    pre = build_preprocessor(num, cat); pre.fit(X_tr2)
    X_tr2p, X_vap, X_tep = pre.transform(X_tr2), pre.transform(X_va), pre.transform(X_te)

    params = cfg.xgb_params.copy()
    # pon un n_estimators GRANDE; ES lo recortará al óptimo:
    params["n_estimators"] = max(params.get("n_estimators", 3000), 5000)

    model = XGBRegressor(**params)

    if args.log_target:
        y_tr_fit = np.log1p(y_tr2)
        y_va_fit = np.log1p(y_va)
    else:
        y_tr_fit, y_va_fit = y_tr2, y_va

    model.fit(
        X_tr2p, y_tr_fit,
        eval_set=[(X_vap, y_va_fit)],
        callbacks=[EarlyStopping(rounds=args.patience, save_best=True)],
        verbose=False
    )

    # predecir con la mejor iteración
    if hasattr(model, "best_iteration") and model.best_iteration is not None:
        yhat_tr = model.predict(X_tr2p, iteration_range=(0, model.best_iteration + 1))
        yhat_te = model.predict(X_tep,   iteration_range=(0, model.best_iteration + 1))
    else:
        yhat_tr = model.predict(X_tr2p); yhat_te = model.predict(X_tep)

    if args.log_target:
        yhat_tr, yhat_te = np.expm1(yhat_tr), np.expm1(yhat_te)

    rep_tr = regression_report(y_tr2, yhat_tr)
    rep_te = regression_report(y_te,  yhat_te)
    print("train:", rep_tr); print("test :", rep_te)
    print(f"[best_iteration] {getattr(model, 'best_iteration', None)}")

    joblib.dump({"pre": pre, "xgb": model, "log_target": bool(args.log_target)},
                os.path.join(args.outdir, "model_xgb.joblib"))
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump({"train": rep_tr, "test": rep_te,
                   "best_iteration": int(getattr(model, "best_iteration", -1)),
                   "log_target": bool(args.log_target)}, f, indent=2)

if __name__ == "__main__":
    main()
