import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------- helpers ----------
def safe_expm1(z):  # solo para LINEAL
    return np.expm1(np.clip(z, -50.0, 20.0))

def regression_report(y_true, y_pred, model_name=""):
    status = "ok"
    y_pred = np.asarray(y_pred)
    if not np.all(np.isfinite(y_pred)):
        status = f"non_finite_predictions_in_{model_name}"
        # métrica con máscara finitos para no crashear
        mask = np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = (np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-9, None))) * 100.0)
    r2   = r2_score(y_true, y_pred)
    resid = pd.Series(y_true - y_pred)
    return {
        "RMSE": rmse, "MAE": mae, "MAPE_pct": mape, "R2": r2,
        "residual_skew": float(resid.skew()), "residual_kurtosis": float(resid.kurtosis()),
        "status": status,
    }

def mape_by_decile(y_true, y_pred, q=10):
    dec = pd.qcut(y_true, q=q, labels=False, duplicates="drop")
    return [float(np.mean(np.abs((y_true[dec==d]-y_pred[dec==d]) /
                                 np.clip(np.abs(y_true[dec==d]), 1e-9, None))) * 100)
            for d in np.unique(dec)]

def get_or_make_split_indices(name, n_rows, test_size, random_state, outdir):
    outdir = Path(outdir) / "splits"; outdir.mkdir(parents=True, exist_ok=True)
    ftr, fte = outdir / f"{name}_train_idx.csv", outdir / f"{name}_test_idx.csv"
    if ftr.exists() and fte.exists():
        return (pd.read_csv(ftr, header=None)[0].to_numpy(),
                pd.read_csv(fte, header=None)[0].to_numpy())
    idx = np.arange(n_rows)
    tr, te = train_test_split(idx, test_size=test_size, random_state=random_state, shuffle=True)
    pd.Series(tr).to_csv(ftr, index=False, header=False); pd.Series(te).to_csv(fte, index=False, header=False)
    return tr, te

# ---------- utils locales ----------
from .preprocess import infer_feature_types, build_preprocessor, QUALITY_CANDIDATE_NAMES, UTIL_TO_ORD
from .config import Config
cfg = Config()

BASES = {
    "base_filtrada": "data/raw/df_final_regresion.csv",
    "base_completa_sin_nulos": "data/processed/base_completa_sin_nulos.csv",
    "base_completa_con_nulos": "data/raw/casas_completas_con_present.csv",  # NaN intactos
}
OUTDIR = Path("models/comparacion/"); OUTDIR.mkdir(parents=True, exist_ok=True)
RESULTS = []

def preparar_dataframe(ruta, tocar_nans: bool):
    df = pd.read_csv(ruta, sep=None, engine="python")
    df.columns = [c.replace("\ufeff","").strip() for c in df.columns]
    if cfg.target not in df.columns:
        raise RuntimeError(f"No se encontró {cfg.target} en {ruta}")

    # casteo importante para alinear con tu retrain
    for col in ["MS SubClass", "Mo Sold"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    df[cfg.target] = pd.to_numeric(df[cfg.target], errors="coerce")
    df = df.dropna(subset=[cfg.target])

    # solo en escenarios "sin nulos"
    if tocar_nans and "Mas Vnr Area" in df.columns:
        df["Mas Vnr Area"] = pd.to_numeric(df["Mas Vnr Area"], errors="coerce").fillna(0.0)

    # calidades -> ordinal
    MAP_Q_ORD = {"Po":0,"Fa":1,"TA":2,"Gd":3,"Ex":4}
    quality_cols = [c for c in QUALITY_CANDIDATE_NAMES if c in df.columns]
    for col in quality_cols:
        as_num = pd.to_numeric(df[col], errors="coerce")
        mask_ok = as_num.isin([0,1,2,3,4])
        df[col] = as_num.where(mask_ok, df[col].map(MAP_Q_ORD)).fillna(-1).astype(int)

    # Utilities -> ordinal
    if "Utilities" in df.columns:
        df["Utilities"] = df["Utilities"].map(UTIL_TO_ORD).fillna(-1).astype(int)

    # OHE resto categórico
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    skip = set(quality_cols + ["Utilities"])
    cat_cols = [c for c in cat_cols if c not in skip]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=float)

    X = df.drop(columns=[cfg.target] + [c for c in cfg.drop_cols if c in df.columns], errors="ignore")
    y = df[cfg.target].values

    # preprocesador: si tocar_nans -> con imputación; si no -> passthrough (XGB tolera NaN)
    num_cols, cat_cols2 = infer_feature_types(df, cfg.target, cfg.drop_cols, cfg.numeric_cols, cfg.categorical_cols)
    if tocar_nans:
        pre = build_preprocessor(num_cols, cat_cols2, quality_cols=quality_cols,
                                 utilities_cols=["Utilities"] if "Utilities" in df.columns else [])
    else:
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline as SKPipeline
        cols_num = sorted(set(num_cols + quality_cols + (["Utilities"] if "Utilities" in df.columns else [])))
        pre = ColumnTransformer([("num", SKPipeline([("passthrough","passthrough")]), cols_num)], remainder="drop")
    return df, X, y, pre

def correr_lineal(nombre_base, X, y, pre):
    lin = Ridge(alpha=1.0, fit_intercept=True, random_state=cfg.random_state)
    lin_reg = TransformedTargetRegressor(regressor=lin, func=np.log1p, inverse_func=safe_expm1)
    pipe = Pipeline([("pre", pre), ("lin", lin_reg)])

    tr, te = get_or_make_split_indices(nombre_base, len(X), cfg.test_size, cfg.random_state, OUTDIR)
    Xtr, Xte = X.iloc[tr], X.iloc[te]; ytr, yte = y[tr], y[te]

    pipe.fit(Xtr, ytr)
    yhat = pipe.predict(Xte)
    rep = regression_report(yte, yhat, model_name="lineal")
    rep["MAPE_by_decile"] = mape_by_decile(yte, yhat)
    rep["modelo"] = "lineal_log"; rep["base"] = nombre_base
    return rep

def correr_xgb(nombre_base, X, y, pre, etiqueta=""):
    xgb = XGBRegressor(**cfg.xgb_params)              # exactamente tus params
    xgb_reg = TransformedTargetRegressor(             # inverse igual a tu retrain
        regressor=xgb, func=np.log1p, inverse_func=np.expm1
    )
    pipe = Pipeline([("pre", pre), ("xgb", xgb_reg)])

    tr, te = get_or_make_split_indices(nombre_base, len(X), cfg.test_size, cfg.random_state, OUTDIR)
    Xtr, Xte = X.iloc[tr], X.iloc[te]; ytr, yte = y[tr], y[te]

    pipe.fit(Xtr, ytr)
    yhat = pipe.predict(Xte)
    rep = regression_report(yte, yhat, model_name="xgboost")
    rep["MAPE_by_decile"] = mape_by_decile(yte, yhat)
    rep["modelo"] = "xgboost_log" + etiqueta; rep["base"] = nombre_base
    return rep

# ---------- runs ----------
OUTDIR.mkdir(parents=True, exist_ok=True)

for nombre_base, ruta in BASES.items():
    print(f"\n=== {nombre_base} ===")
    tocar_nans = (nombre_base != "base_completa_con_nulos")  # NaN intactos en ese escenario
    _, X, y, pre = preparar_dataframe(ruta, tocar_nans=tocar_nans)

    has_nan = X.isna().any().any()
    # lineal solo si no hay NaN y no es el escenario "con nulos"
    if not has_nan and tocar_nans:
        rep_lin = correr_lineal(nombre_base, X, y, pre)
        RESULTS.append(rep_lin)
        (OUTDIR / f"metrics_{nombre_base}_lineal.json").write_text(json.dumps(rep_lin, indent=2))
    else:
        print("(saltando lineal: contiene NaN)")

    # xgb siempre (con y sin nulos)
    etiqueta = "" if tocar_nans else "_con_nulos"
    rep_xgb = correr_xgb(nombre_base, X, y, pre, etiqueta=etiqueta)
    RESULTS.append(rep_xgb)
    (OUTDIR / f"metrics_{nombre_base}_xgb.json").write_text(json.dumps(rep_xgb, indent=2))

# ---------- resumen ----------
cols = ["base","modelo","R2","MAPE_pct","RMSE","MAE","residual_skew","residual_kurtosis"]
df_out = pd.DataFrame(RESULTS)[cols]
df_out.to_csv(OUTDIR / "metrics_comparison.csv", index=False)
print("\n==== RESULTADOS ===="); print(df_out); print(f"\nGuardado en {OUTDIR}")
