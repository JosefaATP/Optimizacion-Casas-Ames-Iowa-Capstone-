from pathlib import Path
from typing import List

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor

from .config import PATHS


class XGBBundle:
    """
    Carga el pipeline entrenado (model_xgb.joblib) y expone:
    - pipe_full: Pipeline(pre -> TransformedTargetRegressor(XGBRegressor))  [tu pipeline]
    - pre: ColumnTransformer
    - reg: XGBRegressor (estimador final)
    - pipe_for_embed: Pipeline(pre -> XGBRegressor)  [mismo objeto fitted, sin TTR]
    - log_target: bool
    """
    def __init__(self, model_path: Path = PATHS.xgb_model_file):
        self.model_path = model_path

        # pipeline completo tal cual lo entrenaste (FITTED)
        self.pipe_full: Pipeline = joblib.load(self.model_path)

        # detectar TTR
        self._ttr: TransformedTargetRegressor | None = None
        last = self.pipe_full.named_steps.get("xgb")
        if isinstance(last, TransformedTargetRegressor):
            self._ttr = last
        self.log_target: bool = self._ttr is not None

        # componentes
        self.pre: ColumnTransformer = self.pipe_full.named_steps["pre"]
        if self._ttr is not None:
            self.reg: XGBRegressor = self._ttr.regressor  # type: ignore
        else:
            self.reg: XGBRegressor = self.pipe_full.named_steps["xgb"]  # type: ignore

        # ***** Pipeline SIN TTR para embed, conservando el mismo objeto FITTED *****
        self.pipe_for_embed: Pipeline = self.pipe_full

        # ***************************************************************************

    def feature_names_in(self) -> List[str]:
        return list(getattr(self.pipe_full, "feature_names_in_", []))

    def is_log_target(self) -> bool:
        return self.log_target

    def pipe_for_gurobi(self) -> Pipeline:
        return self.pipe_for_embed

    def predict(self, X: pd.DataFrame) -> pd.Series:
        # predice con el pipeline COMPLETO (devuelve precio en escala original)
        y = self.pipe_full.predict(X)
        return pd.Series(y, index=X.index)
