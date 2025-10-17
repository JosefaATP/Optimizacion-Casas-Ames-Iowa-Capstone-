# training/retrain_xgb_same_env.py
from optimization.remodel.compat_xgboost import patch_get_booster
patch_get_booster()

import os, json, joblib, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor

# importa tus utilidades
from .config import Config
from .preprocess import (
    infer_feature_types,
    build_preprocessor,
    QUALITY_CANDIDATE_NAMES,
)

from .metrics import regression_report

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="ruta al CSV limpio (ej: data/raw/casas_completas_con_present.csv)")
    p.add_argument("--outdir", default="models/xgb/completa_present_log_p2_1800_ELEGIDO", help="carpeta destino")
    args = p.parse_args()

    cfg = Config()
    os.makedirs(args.outdir, exist_ok=True)

    # 1) leer csv (quitando BOM)
    df = pd.read_csv(args.csv, sep=None, engine="python")
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]


    # forzar categóricas que mencionaste
    for col in ["MS SubClass", "Mo Sold"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    
    # 3) tipos de variables como tu entrenamiento original
    df["Mas Vnr Area"] = pd.to_numeric(df["Mas Vnr Area"], errors="coerce")  # nada de "No aplica"
    df["Mas Vnr Area"] = df["Mas Vnr Area"].fillna(0.0)  # o deja NaN y usa imputador numérico

    # 2) target numeric & drop NA en target
    df[cfg.target] = pd.to_numeric(df[cfg.target], errors="coerce")
    df = df.dropna(subset=[cfg.target])

    # 1) Normalizar columnas de calidad a texto (Po, Fa, TA, Gd, Ex)
    MAP_Q = {0: "Po", 1: "Fa", 2: "TA", 3: "Gd", 4: "Ex"}
    quality_cols = [c for c in QUALITY_CANDIDATE_NAMES if c in df.columns]
    for col in quality_cols:
        # si ya viene como string Po/Fa/TA/Gd/Ex, esto no la daña;
        # si viene como números 0..4, la mapea; si viene otra cosa, la deja como str
        df[col] = df[col].map(MAP_Q).fillna(df[col].astype(str))

    # 2) Tipos como antes
    numeric_cols, categorical_cols = infer_feature_types(
        df, target=cfg.target, drop_cols=cfg.drop_cols,
        numeric_cols=cfg.numeric_cols, categorical_cols=cfg.categorical_cols
    )

    # 4) split
    X = df.drop(columns=[cfg.target] + [c for c in cfg.drop_cols if c in df.columns], errors="ignore")
    y = df[cfg.target].values
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )
    # 5) preprocesador
    pre = build_preprocessor(numeric_cols, categorical_cols)

    # 6) modelo
    xgb = XGBRegressor(**cfg.xgb_params)

    # 7) target en log como antes
    reg = TransformedTargetRegressor(regressor=xgb, func=np.log1p, inverse_func=np.expm1)

    # 8) pipeline y fit
    pipe = Pipeline(steps=[("pre", pre), ("xgb", reg)])
    pipe.fit(Xtr, ytr)

    from sklearn.compose import TransformedTargetRegressor as TTR

    xgb_step = pipe.named_steps["xgb"]
    # >>> OJO: después de fit, el estimador entrenado está en `regressor_`
    xgb_reg = xgb_step.regressor_ if isinstance(xgb_step, TTR) else xgb_step

    # ====== OBTENER BOOSTER DE FORMA SEGURA (sin forzar get_booster) ======
    bst = getattr(xgb_reg, "_Booster", None)   # muchos builds lo tienen aquí tras fit
    if bst is None:
        # como respaldo, intenta get_booster, pero protegido
        try:
            bst = xgb_reg.get_booster()
        except Exception as e:
            bst = None

    # Si logramos el booster, guardamos bytes crudos para que sobreviva a joblib
    if bst is not None:
        try:
            xgb_reg._Booster_raw = bst.save_raw()
        except Exception:
            pass
    # ======================================================================


    # 9) evaluación
    yhat_tr = pipe.predict(Xtr)
    yhat_te = pipe.predict(Xte)
    rep_tr = regression_report(ytr, yhat_tr)
    rep_te = regression_report(yte, yhat_te)

    # extras de residuos
    res_tr = pd.Series(ytr - yhat_tr)
    res_te = pd.Series(yte - yhat_te)
    extra_tr = {"residual_skew": float(res_tr.skew()), "residual_kurtosis": float(res_tr.kurtosis())}
    extra_te = {"residual_skew": float(res_te.skew()), "residual_kurtosis": float(res_te.kurtosis())}

    print("train:", rep_tr)
    print("test :", rep_te)

    # 10) guardar artefactos
    model_path = Path(args.outdir) / "model_xgb.joblib"
    joblib.dump(pipe, model_path)

    if bst is not None:
        (Path(args.outdir) / "booster.json").unlink(missing_ok=True)
        bst.save_model(str(Path(args.outdir) / "booster.json"))


    with open(Path(args.outdir) / "metrics.json", "w") as f:
        json.dump({"train": {**rep_tr, **extra_tr}, "test": {**rep_te, **extra_te}, "log_target": True}, f, indent=2)

    meta = {
        "target": cfg.target,
        "drop_cols": cfg.drop_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "xgb_params": cfg.xgb_params,
        "log_target": True,
    }
    with open(Path(args.outdir) / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # 11) opcional: export Booster a JSON (por si quieres usarlo directo alguna vez)
    # OJO: esto requiere acceder al regressor interno ya fitted:
    '''booster = pipe.named_steps["xgb"].regressor.get_booster()
    booster.save_model(str(Path(args.outdir) / "booster.json"))'''
    print(f"Guardado:\n - {model_path}\n - booster.json\n - metrics.json\n - meta.json")

if __name__ == "__main__":
    main()

