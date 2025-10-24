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
    QUAL_ORD,
    UTIL_TO_ORD,
)
from .metrics import regression_report


def _to_ord_series(s: pd.Series) -> pd.Series:
    """Convierte Po..Ex a 0..4; si ya viene 0..4 lo respeta; “No aplica” y raros -> -1."""
    MAP_Q_ORD = {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4, "No aplica": -1}
    as_num = pd.to_numeric(s, errors="coerce")
    mask_ok = as_num.isin([-1, 0, 1, 2, 3, 4])
    return as_num.where(mask_ok, s.map(MAP_Q_ORD)).fillna(-1).astype(int)


def retrain_xgb(
    csv_path: str,
    outdir: str = "models/xgb/ordinal_p2_1800",
    cfg: Config | None = None,
    test_size: float | None = None,
    random_state: int | None = None,
    verbose: bool = True,
) -> dict:
    """
    Entrena/re-entrena el XGB usando el mismo preprocesamiento del script original.

    Parámetros
    ----------
    csv_path : str
        Ruta al CSV limpio (ej: 'data/processed/base_completa_sin_nulos.csv').
    outdir : str
        Carpeta de salida donde se guardan modelo y metadatos.
    cfg : Config | None
        Objeto Config opcional. Si None, se instancia por defecto.
    test_size : float | None
        Para override de cfg.test_size.
    random_state : int | None
        Para override de cfg.random_state.
    verbose : bool
        Si True, imprime métricas breves.

    Returns
    -------
    dict con:
        - 'pipe': Pipeline entrenado
        - 'metrics': {'train': {...}, 'test': {...}}
        - 'paths':  rutas a archivos guardados
        - 'meta':   diccionario meta.json escrito
    """
    cfg = cfg or Config()
    if test_size is None:
        test_size = cfg.test_size
    if random_state is None:
        random_state = cfg.random_state

    os.makedirs(outdir, exist_ok=True)

    # 1) Cargar CSV
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]

    for col in ["MS SubClass", "Mo Sold"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # 2) Target numérico y sin NA
    df[cfg.target] = pd.to_numeric(df[cfg.target], errors="coerce")
    df = df.dropna(subset=[cfg.target])

    # 3) Quality → ordinal (-1..4)
    qual_ord_present = [c for c in QUAL_ORD if c in df.columns]
    for col in qual_ord_present:
        df[col] = _to_ord_series(df[col])

    # 4) Utilities → ordinal (0..3; raros -> -1)
    if "Utilities" in df.columns:
        df["Utilities"] = df["Utilities"].map(UTIL_TO_ORD).fillna(-1).astype(int)

    # 5) OHE horneado (excepto quality y Utilities)
    cats_in_df = df.select_dtypes(include=["object", "category"]).columns.tolist()
    ohe_bases = [c for c in cats_in_df if c not in qual_ord_present and c != "Utilities"]

    df = pd.get_dummies(df, columns=ohe_bases, drop_first=False, dtype=float)
    dummy_cols = [c for c in df.columns if any(c.startswith(f"{base}_") for base in ohe_bases)]

    # 6) Inferir numéricas
    numeric_cols, categorical_cols = infer_feature_types(
        df, target=cfg.target, drop_cols=cfg.drop_cols,
        numeric_cols=cfg.numeric_cols, categorical_cols=cfg.categorical_cols
    )

    # 7) Preprocesador (solo numéricas)
    pre = build_preprocessor(numeric_cols=numeric_cols)

    # 8) Modelo (log-target)
    xgb = XGBRegressor(**cfg.xgb_params)
    reg = TransformedTargetRegressor(regressor=xgb, func=np.log1p, inverse_func=np.expm1)

    X = df.drop(columns=[cfg.target] + [c for c in cfg.drop_cols if c in df.columns], errors="ignore")
    y = df[cfg.target].values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state)
    pipe = Pipeline(steps=[("pre", pre), ("xgb", reg)])
    pipe.fit(Xtr, ytr)

    # Booster robusto para joblib
    from sklearn.compose import TransformedTargetRegressor as TTR
    xgb_step = pipe.named_steps["xgb"]
    xgb_reg = xgb_step.regressor_ if isinstance(xgb_step, TTR) else xgb_step
    try:
        bst = xgb_reg.get_booster()
        xgb_reg._Booster_raw = bst.save_raw()
    except Exception:
        pass

    # 9) Métricas
    yhat_tr = pipe.predict(Xtr); yhat_te = pipe.predict(Xte)
    rep_tr = regression_report(ytr, yhat_tr)
    rep_te = regression_report(yte, yhat_te)
    if verbose:
        print("train:", rep_tr); print("test :", rep_te)

    # 10) Guardados
    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    model_path = outdir_p / "model_xgb.joblib"
    joblib.dump(pipe, model_path)

    # Booster json (opcional)
    booster_path = outdir_p / "booster.json"
    try:
        bst = xgb_reg.get_booster()
        booster_path.unlink(missing_ok=True)
        bst.save_model(str(booster_path))
    except Exception:
        booster_path = None

    # Extras de residuales
    res_tr = pd.Series(ytr - yhat_tr); res_te = pd.Series(yte - yhat_te)
    extra_tr = {"residual_skew": float(res_tr.skew()), "residual_kurtosis": float(res_tr.kurtosis())}
    extra_te = {"residual_skew": float(res_te.skew()), "residual_kurtosis": float(res_te.kurtosis())}

    metrics = {"train": {**rep_tr, **extra_tr}, "test": {**rep_te, **extra_te}, "log_target": True}
    with open(outdir_p / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    meta = {
        "target": cfg.target,
        "drop_cols": cfg.drop_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "xgb_params": cfg.xgb_params,
        "log_target": True,
        "quality_cols": qual_ord_present,
        "utilities_cols": ["Utilities"] if "Utilities" in df.columns else [],
        "dummy_cols": dummy_cols,
        "ohe_baked_bases": ohe_bases,
    }
    with open(outdir_p / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"Guardado:\n - {model_path}\n - booster.json\n - metrics.json\n - meta.json")

    return {
        "pipe": pipe,
        "metrics": metrics,
        "paths": {
            "model": str(model_path),
            "booster": str(booster_path) if booster_path else None,
            "metrics": str(outdir_p / "metrics.json"),
            "meta": str(outdir_p / "meta.json"),
        },
        "meta": meta,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True,
                   help="ruta al CSV limpio (ej: data/processed/base_completa_sin_nulos.csv)")
    p.add_argument("--outdir", default="models/xgb/ordinal_p2_1800", help="carpeta destino")
    args = p.parse_args()

    retrain_xgb(csv_path=args.csv, outdir=args.outdir, verbose=True)


if __name__ == "__main__":
    main()
