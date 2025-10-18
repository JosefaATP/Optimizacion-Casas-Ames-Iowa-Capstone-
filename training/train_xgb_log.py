import argparse, json, os, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor


from config import Config
from preprocess import infer_feature_types, build_preprocessor
from metrics import regression_report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="ruta al csv, ej data/raw/casas_completas_con_present.csv")
    parser.add_argument("--target", default=None, help="nombre de la columna objetivo (por defecto usa config.target)")
    parser.add_argument("--outdir", default="models/xgb", help="carpeta de salida")
    parser.add_argument("--log_target", action="store_true",
                        help="aplica log1p al target al entrenar y expm1 al predecir (comparable a tu RL en log)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cfg = Config()
    if args.target:
        cfg.target = args.target

    # 1) leer csv con autodetección de separador y limpiar BOM
    df = pd.read_csv(args.csv, sep=None, engine="python")
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]

    #2) Forzar categóricas
    # Forzar que estas dos se traten como categóricas
    for col in ["MS SubClass", "Mo Sold"]:
        if col in df.columns:
            df[col] = df[col].astype("category")  # o .astype(str) si prefieres


    # 2) asegurar target numérico y eliminar filas inválidas
    df[cfg.target] = pd.to_numeric(df[cfg.target], errors="coerce")
    bad_before = df[cfg.target].isna().sum()
    if bad_before:
        print(f"[clean] filas eliminadas por target inválido (NaN/inf): {bad_before}")
        df = df.dropna(subset=[cfg.target])

    # 3) inferir tipos de variables
    numeric_cols, categorical_cols = infer_feature_types(
        df, target=cfg.target, drop_cols=cfg.drop_cols,
        numeric_cols=cfg.numeric_cols, categorical_cols=cfg.categorical_cols
    )

    # 4) split train/test
    X = df.drop(columns=[cfg.target] + [c for c in cfg.drop_cols if c in df.columns], errors="ignore")
    y = df[cfg.target].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    # 5) preprocesador
    pre = build_preprocessor(numeric_cols, categorical_cols)

    # 6) modelo XGB
    xgb = XGBRegressor(**cfg.xgb_params)

    # 7) OPCIONAL: target en log con TransformedTargetRegressor (maneja log/expm1 automáticamente)
    reg = TransformedTargetRegressor(regressor=xgb, func=np.log1p, inverse_func=np.expm1) if args.log_target else xgb

    # 8) pipeline final y fit
    pipe = Pipeline(steps=[("pre", pre), ("xgb", reg)])
    pipe.fit(X_train, y_train)

    # 9) evaluar
    yhat_tr = pipe.predict(X_train)
    yhat_te = pipe.predict(X_test)
    rep_tr = regression_report(y_train, yhat_tr)
    rep_te = regression_report(y_test, yhat_te)


    res_tr = pd.Series(y_train - yhat_tr)
    res_te = pd.Series(y_test  - yhat_te)
    extra_tr = {"residual_skew": float(res_tr.skew()), "residual_kurtosis": float(res_tr.kurtosis())}
    extra_te = {"residual_skew": float(res_te.skew()), "residual_kurtosis": float(res_te.kurtosis())}


    print("train:", rep_tr)
    print("test :", rep_te)

    # 10) guardar artefactos
    joblib.dump(pipe, os.path.join(args.outdir, "model_xgb.joblib"))
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump({"train": {**rep_tr, **extra_tr}, "test": {**rep_te, **extra_te}, "log_target": True}, f, indent=2)


    meta = {
        "target": cfg.target,
        "drop_cols": cfg.drop_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "xgb_params": cfg.xgb_params,
        "log_target": bool(args.log_target),
    }
    with open(os.path.join(args.outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
