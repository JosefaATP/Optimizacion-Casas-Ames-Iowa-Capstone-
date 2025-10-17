# xgb_predictor.py
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.exceptions import NotFittedError

from xgboost import XGBRegressor, Booster

from .config import PATHS


def _patch_ohe_categories_inplace(ct: ColumnTransformer) -> None:
    """
    Parche defensivo: para cada OneHotEncoder ya 'fitted',
    fuerza categories_ a ser listas de 'str' sin NaNs para
    evitar el np.isnan(...) de sklearn en arrays object mixtos.
    """
    trs = ct.transformers_ if hasattr(ct, "transformers_") else ct.transformers
    for item in trs:
        # item: (name, transformer, cols, *extras)
        name, transformer, cols = item[0], item[1], item[2]
        est = transformer.steps[-1][1] if isinstance(transformer, SKPipeline) else transformer
        if isinstance(est, OneHotEncoder) and hasattr(est, "categories_"):
            new_cats = []
            for arr in est.categories_:
                arr_list = []
                for v in arr:
                    if isinstance(v, float) and np.isnan(v):
                        continue
                    arr_list.append(str(v))
                new_cats.append(np.array(arr_list, dtype=object))
            est.categories_ = new_cats


def _flatten_column_transformer_inplace(ct: ColumnTransformer) -> None:
    """
    - Reemplaza en ct.transformers_ (y named_transformers_) cualquier sklearn.Pipeline
      por su último paso (estimador final), conservando objetos FITTED.
    - Reemplaza cualquier SimpleImputer (directo o como último paso) por 'passthrough'
      (identidad), asumiendo que tu base no tiene nulos para esas columnas.
    """
    from sklearn.impute import SimpleImputer as _SI

    has_fitted = hasattr(ct, "transformers_")
    trs = ct.transformers_ if has_fitted else ct.transformers

    new_trs = []
    for item in trs:
        name, transformer, cols = item[0], item[1], item[2]
        extras = item[3:] if len(item) > 3 else ()

        final_est = transformer
        if isinstance(transformer, SKPipeline):
            final_est = transformer.steps[-1][1]  # último paso del pipeline interno

        if isinstance(final_est, _SI):
            final_est = "passthrough"

        new_item = (name, final_est, cols, *extras)
        new_trs.append(new_item)

        if hasattr(ct, "named_transformers_"):
            ct.named_transformers_[name] = final_est

    if has_fitted:
        ct.transformers_ = new_trs
    else:
        ct.transformers = new_trs


def _align_ohe_input_dtypes(X: pd.DataFrame, pre: ColumnTransformer) -> pd.DataFrame:
    """
    Fuerza dtype str en las columnas que alimentan a OneHotEncoder para evitar
    errores de np.isnan con objetos mixtos.
    """
    trs = pre.transformers_ if hasattr(pre, "transformers_") else pre.transformers
    X2 = X.copy()
    for item in trs:
        name, transformer, cols = item[0], item[1], item[2]
        est = transformer.steps[-1][1] if isinstance(transformer, SKPipeline) else transformer
        if isinstance(est, OneHotEncoder):
            for c in cols:
                if c in X2.columns:
                    X2[c] = X2[c].astype(str)
    return X2


class XGBBundle:
    """
    Carga el pipeline entrenado (model_xgb.joblib) y expone:
    - pipe_full: Pipeline(pre -> TransformedTargetRegressor(XGBRegressor))  [tal cual entrenaste]
    - pre: ColumnTransformer (aplanado, sin SimpleImputer)
    - reg: XGBRegressor (estimador final)
    - pipe_for_embed: Pipeline(pre -> XGBRegressor)  [mismo objeto fitted, sin TTR]
    - log_target: bool
    """
    def __init__(self, model_path: Path = PATHS.xgb_model_file):
        self.model_path = model_path

        # Pipeline completo tal cual lo entrenaste (FITTED)
        self.pipe_full: SKPipeline = joblib.load(self.model_path)

        # Detectar TTR
        self._ttr: TransformedTargetRegressor | None = None
        last = self.pipe_full.named_steps.get("xgb")
        if isinstance(last, TransformedTargetRegressor):
            self._ttr = last
        self.log_target: bool = self._ttr is not None

        # Componentes
        self.pre: ColumnTransformer = self.pipe_full.named_steps["pre"]
        _flatten_column_transformer_inplace(self.pre)
        _patch_ohe_categories_inplace(self.pre)

        if self._ttr is not None:
            self.reg: XGBRegressor = self._ttr.regressor_
        else:
            self.reg: XGBRegressor = self.pipe_full.named_steps["xgb"]  # type: ignore

        # --- Asegurar Booster en self.reg aunque joblib lo haya perdido ---
        try:
            _ = self.reg.get_booster()  # si ya está bien, no hace nada
        except Exception:
            # 1) Intentar desde bytes picklados (si el retraining los guardó)
            if hasattr(self.reg, "_Booster_raw") and getattr(self.reg, "_Booster_raw", None):
                bst = Booster()
                bst.load_model(self.reg._Booster_raw)  # bytes/bytearray
                self.reg._Booster = bst
            else:
                # 2) Intentar desde archivo booster.json
                bj = PATHS.model_dir / "booster.json"
                if bj.exists():
                    bst = Booster()
                    bst.load_model(str(bj))
                    self.reg._Booster = bst
                else:
                    raise FileNotFoundError(
                        "No se encontró Booster interno ni 'booster.json'. "
                        "Vuelve a entrenar con el script que guarda _Booster_raw/booster.json."
                    )

        # Si aun así sklearn cree que no está 'fitted', inyecta booster desde archivo
        try:
            _ = self.reg.get_booster()
        except NotFittedError:
            bj = self.model_path.parent / "booster.json"
            self.reg.load_model(str(bj))

        # Pipeline NUEVO para embed: (pre -> XGBRegressor), ambos ya FITTED
        self.pipe_for_embed: SKPipeline = SKPipeline(steps=[
            ("pre", self.pre),
            ("xgb", self.reg),
        ])

    def feature_names_in(self) -> List[str]:
        return list(getattr(self.pipe_full, "feature_names_in_", []))

    def is_log_target(self) -> bool:
        return self.log_target

    def pipe_for_gurobi(self) -> SKPipeline:
        # Devolvemos el pipeline SIN TTR, sólo (pre -> XGBRegressor)
        return self.pipe_for_embed

    def predict(self, X: pd.DataFrame) -> pd.Series:
        # Alinear dtypes de entrada para OHE y predecir con el pipeline completo
        X_fixed = _align_ohe_input_dtypes(X, self.pre)
        y = self.pipe_full.predict(X_fixed)
        return pd.Series(y, index=X.index)
