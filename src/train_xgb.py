import argparse, json, os, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from config import Config
from preprocess import infer_feature_types, build_preprocessor
from metrics import regression_report
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="ruta al csv, ej data/raw/casas_completa.csv")
    parser.add_argument("--target", default=None, help="nombre de la columna objetivo, sobreescribe config")
    parser.add_argument("--outdir", default="models/xgb", help="carpeta de salida")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cfg = Config()
    if args.target:
        cfg.target = args.target

    df = pd.read_csv(args.csv)

    # inferencia de tipos
    numeric_cols, categorical_cols = infer_feature_types(
        df, target=cfg.target, drop_cols=cfg.drop_cols,
        numeric_cols=cfg.numeric_cols, categorical_cols=cfg.categorical_cols
    )

    # split
    X = df.drop(columns=[cfg.target] + [c for c in cfg.drop_cols if c in df.columns], errors="ignore")
    y = df[cfg.target].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    # preprocesador
    pre = build_preprocessor(numeric_cols, categorical_cols)

    # xgboost
    xgb_params = cfg.xgb_params.copy()

    # restricciones monotoniacas opcionales
    if cfg.monotone:
        # construir vector de monotone_constraints en el orden final de features numericas + categoricas onehot
        # simplificacion: aplicamos solo a numericas conocidas, 0 por defecto
        mono_map = cfg.monotone
        mono_list = []
        for col in numeric_cols:
            mono_list.append(mono_map.get(col, 0))
        # para categoricas (onehot) usamos 0
        from preprocess import OneHotEncoder  # solo para contar categorias indirectamente
        xgb_params["monotone_constraints"] = "(" + ",".join(str(v) for v in mono_list) + ")"

    model = XGBRegressor(**xgb_params)

    pipe = Pipeline(steps=[
        ("pre", pre),
        ("xgb", model)
    ])

    pipe.fit(X_train, y_train)

    # eval
    yhat_tr = pipe.predict(X_train)
    yhat_te = pipe.predict(X_test)

    rep_tr = regression_report(y_train, yhat_tr)
    rep_te = regression_report(y_test, yhat_te)

    print("train:", rep_tr)
    print("test :", rep_te)

    # guardar artefactos
    joblib.dump(pipe, os.path.join(args.outdir, "model_xgb.joblib"))
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump({"train": rep_tr, "test": rep_te}, f, indent=2)

    # guardar meta de features
    meta = {
        "target": cfg.target,
        "drop_cols": cfg.drop_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "xgb_params": xgb_params
    }
    with open(os.path.join(args.outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
