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

from .config import Config
from .preprocess import (
    infer_feature_types,
    build_preprocessor,
    QUAL_ORD,      # columnas de calidad que quedarán ordinales (-1..4)
    UTIL_TO_ORD,   # Utilities ordinal
)
from .metrics import regression_report

def _to_ord_series(s: pd.Series) -> pd.Series:
    """
    Convierte Po..Ex a 0..4; si ya viene 0..4 lo respeta; “No aplica” y raros -> -1.
    """
    MAP_Q_ORD = {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4, "No aplica": -1}
    as_num = pd.to_numeric(s, errors="coerce")
    mask_ok = as_num.isin([-1, 0, 1, 2, 3, 4])
    return as_num.where(mask_ok, s.map(MAP_Q_ORD)).fillna(-1).astype(int)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="ruta al CSV limpio (ej: data/processed/base_completa_sin_nulos.csv)")
    p.add_argument("--outdir", default="models/xgb/ordinal_p2_1800", help="carpeta destino")
    args = p.parse_args()

    cfg = Config()
    os.makedirs(args.outdir, exist_ok=True)

    # 1) Cargar CSV
    df = pd.read_csv(args.csv, sep=None, engine="python")
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]

    for col in ["MS SubClass", "Mo Sold"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # 2) Target numérico y sin NA en target
    df[cfg.target] = pd.to_numeric(df[cfg.target], errors="coerce")
    df = df.dropna(subset=[cfg.target])

    # 3) Quality → ordinal (-1..4)
    qual_ord_present = [c for c in QUAL_ORD if c in df.columns]
    for col in qual_ord_present:
        df[col] = _to_ord_series(df[col])

    # 4) Utilities → ordinal (0..3; raros -> -1)
    if "Utilities" in df.columns:
        df["Utilities"] = df["Utilities"].map(UTIL_TO_ORD).fillna(-1).astype(int)

    # 5) OHE horneado:
    #    Incluir todas las categóricas (object/category) EXCEPTO las ya convertidas a ordinal (quality) y Utilities
    cats_in_df = df.select_dtypes(include=["object", "category"]).columns.tolist()
    ohe_bases = [c for c in cats_in_df if c not in qual_ord_present and c != "Utilities"]

    df = pd.get_dummies(df, columns=ohe_bases, drop_first=False, dtype=float)
    dummy_cols = [c for c in df.columns if any(c.startswith(f"{base}_") for base in ohe_bases)]

    # 6) Inferir numéricas (tras OHE casi todo es numérico)
    numeric_cols, categorical_cols = infer_feature_types(
        df, target=cfg.target, drop_cols=cfg.drop_cols,
        numeric_cols=cfg.numeric_cols, categorical_cols=cfg.categorical_cols
    )

    # 7) Preprocesador (solo pasa numéricas)
    pre = build_preprocessor(numeric_cols=numeric_cols)

    # 8) Modelo (log-target)
    xgb = XGBRegressor(**cfg.xgb_params)
    # Mover early stopping al constructor/params para evitar el warning deprecado
    try:
        xgb.set_params(early_stopping_rounds=100)
    except Exception:
        pass
    reg = TransformedTargetRegressor(regressor=xgb, func=np.log1p, inverse_func=np.expm1)

    X = df.drop(columns=[cfg.target] + [c for c in cfg.drop_cols if c in df.columns], errors="ignore")
    y = df[cfg.target].values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.random_state)
    pipe = Pipeline(steps=[("pre", pre), ("xgb", reg)])

    # Early stopping: evalúa en el target transformado (log1p)
    # ColumnTransformer aquí no aprende parámetros, sólo reordena/selecciona, por lo que es seguro.
    pre_tmp = build_preprocessor(numeric_cols=numeric_cols)
    Xtr_tmp = pre_tmp.fit_transform(Xtr)
    Xte_tmp = pre_tmp.transform(Xte)
    yte_log = np.log1p(yte)

    # Entrena con parada temprana: eval_set recibe matrices ya transformadas (X) y y en escala log
    pipe.fit(
        Xtr, ytr,
        xgb__eval_set=[(Xte_tmp, yte_log)],
        xgb__verbose=False,
    )

    # Booster robusto para joblib
    from sklearn.compose import TransformedTargetRegressor as TTR
    xgb_step = pipe.named_steps["xgb"]
    xgb_reg = xgb_step.regressor_ if isinstance(xgb_step, TTR) else xgb_step
    try:
        bst = xgb_reg.get_booster()
        xgb_reg._Booster_raw = bst.save_raw()
    except Exception:
        pass

    # 9) Métricas y guardado
    yhat_tr = pipe.predict(Xtr); yhat_te = pipe.predict(Xte)
    rep_tr = regression_report(ytr, yhat_tr); rep_te = regression_report(yte, yhat_te)
    print("train:", rep_tr); print("test :", rep_te)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_path = outdir / "model_xgb.joblib"
    joblib.dump(pipe, model_path)

    # Booster json (opcional)
    try:
        bst = xgb_reg.get_booster()
        (outdir / "booster.json").unlink(missing_ok=True)
        bst.save_model(str(outdir / "booster.json"))
    except Exception:
        pass

    # Extras de residuales
    res_tr = pd.Series(ytr - yhat_tr); res_te = pd.Series(yte - yhat_te)
    extra_tr = {"residual_skew": float(res_tr.skew()), "residual_kurtosis": float(res_tr.kurtosis())}
    extra_te = {"residual_skew": float(res_te.skew()), "residual_kurtosis": float(res_te.kurtosis())}

    with open(outdir / "metrics.json", "w") as f:
        json.dump({"train": {**rep_tr, **extra_tr}, "test": {**rep_te, **extra_te}, "log_target": True}, f, indent=2)

    # Si hay mejor iteración (early stopping), guárdala en meta
    best_iter = None
    try:
        from sklearn.compose import TransformedTargetRegressor as TTR
        xgb_step = pipe.named_steps["xgb"]
        xgb_reg = xgb_step.regressor_ if isinstance(xgb_step, TTR) else xgb_step
        best_iter = int(getattr(xgb_reg, "best_iteration", -1))
    except Exception:
        best_iter = None

    meta = {
        "target": cfg.target,
        "drop_cols": cfg.drop_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "xgb_params": cfg.xgb_params,
        "log_target": True,
        "best_iteration": best_iter,

        # IMPORTANTE para inferencia/MIP:
        "quality_cols": qual_ord_present,                   # ordinales -1..4
        "utilities_cols": ["Utilities"] if "Utilities" in df.columns else [],

        "dummy_cols": dummy_cols,                           # OHE reales
        "ohe_baked_bases": ohe_bases,                       # bases OHE usadas
    }
    with open(outdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Guardado:\n - {model_path}\n - booster.json\n - metrics.json\n - meta.json")

if __name__ == "__main__":
    main()
