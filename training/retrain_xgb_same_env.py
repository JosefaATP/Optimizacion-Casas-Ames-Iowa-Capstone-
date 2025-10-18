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
    UTIL_TO_ORD,        # 👈 mapeo ordinal para Utilities
)

from .metrics import regression_report

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="ruta al CSV limpio (ej: data/processed/base_completa_sin_nulos.csv)")
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

    # 2) target numeric & drop NA en target
    df[cfg.target] = pd.to_numeric(df[cfg.target], errors="coerce")
    df = df.dropna(subset=[cfg.target])


    # === A) Calidades → ordinal (0..4) =======================================
    MAP_Q_ORD = {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4}
    quality_cols = [c for c in QUALITY_CANDIDATE_NAMES if c in df.columns]

    def to_ord_series(s: pd.Series) -> pd.Series:
        # si ya viene como 0..4, respétalo; si viene como string Po..Ex, mapéalo; raro -> -1
        as_num = pd.to_numeric(s, errors="coerce")
        mask_ok = as_num.isin([0, 1, 2, 3, 4])
        return as_num.where(mask_ok, s.map(MAP_Q_ORD)).fillna(-1).astype(int)

    for col in quality_cols:
        df[col] = to_ord_series(df[col])

    # === B) Utilities → ordinal con tu orden ================================
    utilities_cols = [c for c in ["Utilities"] if c in df.columns]
    for col in utilities_cols:
        df[col] = df[col].map(UTIL_TO_ORD).fillna(-1).astype(int)

    # ====== BAKED ONE-HOT ======
    # 1) Define qué columnas vas a convertir a dummies (las que antes iban a OHE)
    #    Si ya tienes listas en tu Config, usa esas; si no, toma del DataFrame:
    cands_for_ohe = df.select_dtypes(include=["object","category"]).columns.tolist()

    # remueve las que ya convertiste a ordinal (quality_cols) y Utilities (ahora es int)
    skip = set(quality_cols + ["Utilities"])
    cands_for_ohe = [c for c in cands_for_ohe if c not in skip]

    # 2) get_dummies con todas las columnas (drop_first=False para no perder información)
    df = pd.get_dummies(df, columns=cands_for_ohe, drop_first=False, dtype=float)

    # 3) Guarda la lista de dummies creadas (para usarla luego en inferencia/MIP)
    dummy_cols = [c for c in df.columns if any(c.startswith(f"{base}_") for base in cands_for_ohe)]

    # 4) tipos
    numeric_cols, categorical_cols = infer_feature_types(
        df, target=cfg.target, drop_cols=cfg.drop_cols,
        numeric_cols=cfg.numeric_cols, categorical_cols=cfg.categorical_cols
    )

    # 5) preprocesador (excluye Q y Utilities del OHE)
    pre = build_preprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        quality_cols=quality_cols,
        utilities_cols=utilities_cols
    )

    # 6) modelo + target en log
    xgb = XGBRegressor(**cfg.xgb_params)
    reg = TransformedTargetRegressor(regressor=xgb, func=np.log1p, inverse_func=np.expm1)

    # 7) pipeline y fit
    X = df.drop(columns=[cfg.target] + [c for c in cfg.drop_cols if c in df.columns], errors="ignore")
    y = df[cfg.target].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.random_state)

    pipe = Pipeline(steps=[("pre", pre), ("xgb", reg)])
    pipe.fit(Xtr, ytr)

    # Booster robusto para joblib
    from sklearn.compose import TransformedTargetRegressor as TTR
    xgb_step = pipe.named_steps["xgb"]
    xgb_reg = xgb_step.regressor_ if isinstance(xgb_step, TTR) else xgb_step
    bst = getattr(xgb_reg, "_Booster", None)
    if bst is None:
        try: bst = xgb_reg.get_booster()
        except Exception: bst = None
    if bst is not None:
        try: xgb_reg._Booster_raw = bst.save_raw()
        except Exception: pass

    # 8) evaluación y guardado
    yhat_tr = pipe.predict(Xtr); yhat_te = pipe.predict(Xte)
    rep_tr = regression_report(ytr, yhat_tr); rep_te = regression_report(yte, yhat_te)
    print("train:", rep_tr); print("test :", rep_te)

    model_path = Path(args.outdir) / "model_xgb.joblib"
    joblib.dump(pipe, model_path)

    # Booster opcional
    from sklearn.compose import TransformedTargetRegressor as TTR
    xgb_step = pipe.named_steps["xgb"]
    xgb_reg = xgb_step.regressor_ if isinstance(xgb_step, TTR) else xgb_step
    bst = getattr(xgb_reg, "_Booster", None)
    if bst is not None:
        (Path(args.outdir) / "booster.json").unlink(missing_ok=True)
        bst.save_model(str(Path(args.outdir) / "booster.json"))

    # ---- ¡OJO! Usar el pd del import de arriba (no reimportar dentro de la función) ----
    res_tr = pd.Series(ytr - yhat_tr); res_te = pd.Series(yte - yhat_te)
    extra_tr = {"residual_skew": float(res_tr.skew()), "residual_kurtosis": float(res_tr.kurtosis())}
    extra_te = {"residual_skew": float(res_te.skew()), "residual_kurtosis": float(res_te.kurtosis())}

    with open(Path(args.outdir) / "metrics.json", "w") as f:
        import json
        json.dump({"train": {**rep_tr, **extra_tr}, "test": {**rep_te, **extra_te}, "log_target": True}, f, indent=2)

    meta = {
        "target": cfg.target,
        "drop_cols": cfg.drop_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "xgb_params": cfg.xgb_params,
        "log_target": True,
        "quality_cols": quality_cols,
        "utilities_cols": utilities_cols,
        "dummy_cols": dummy_cols,
        "ohe_baked_bases": cands_for_ohe,   # por si quieres reconstruir
    }
    with open(Path(args.outdir) / "meta.json", "w") as f:
        import json
        json.dump(meta, f, indent=2)

    print(f"Guardado:\n - {model_path}\n - booster.json\n - metrics.json\n - meta.json")

if __name__ == "__main__":
    main()
