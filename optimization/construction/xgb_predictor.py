# optimization/construction/xgb_predictor.py
from pathlib import Path
from typing import List

import math, json
import joblib
import numpy as np
import pandas as pd
import json
import gurobipy as gp

from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.exceptions import NotFittedError

from xgboost import XGBRegressor, Booster

from .config import PATHS

# columnas “de calidad” presentes en Ames
QUALITY_CANDIDATE_NAMES = [
    "Kitchen Qual", "Exter Qual", "Exter Cond",
    "Bsmt Qual", "Bsmt Cond",
    "Heating QC",
    "Fireplace Qu",
    "Garage Qual", "Garage Cond",
    "Pool QC",
]
MAP_Q_ORD = {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4}

# Utilities: orden (de peor→mejor, el que usamos en entrenamiento)
UTIL_ORDER = ["ELO", "NoSeWa", "NoSewr", "AllPub"]
UTIL_TO_ORD = {u: i for i, u in enumerate(UTIL_ORDER)}

ROOF_STYLE_TO_ORD = {
    "Flat":0, "Gable":1, "Gambrel":2, "Hip":3, "Mansard":4, "Shed":5
}
ROOF_MATL_TO_ORD = {
    "ClyTile":0, "CompShg":1, "Membran":2, "Metal":3, "Roll":4, "Tar&Grv":5, "WdShake":6, "WdShngl":7
}

def _coerce_roof_ordinals_inplace(X: pd.DataFrame) -> pd.DataFrame:
    if "Roof Style" in X.columns:
        s = X["Roof Style"]
        as_num = pd.to_numeric(s, errors="coerce")
        ok = as_num.isin(list(range(6)))
        X["Roof Style"] = as_num.where(ok, s.astype(str).map(ROOF_STYLE_TO_ORD)).fillna(-1).astype(int)
    if "Roof Matl" in X.columns:
        s = X["Roof Matl"]
        as_num = pd.to_numeric(s, errors="coerce")
        ok = as_num.isin(list(range(8)))
        X["Roof Matl"] = as_num.where(ok, s.astype(str).map(ROOF_MATL_TO_ORD)).fillna(-1).astype(int)
    return X

def _coerce_quality_ordinals_inplace(X: pd.DataFrame, quality_cols: list[str]) -> pd.DataFrame:
    # Convierte strings Po/Fa/TA/Gd/Ex a 0..4; si ya viene 0..4, lo deja; raro -> -1.
    for c in quality_cols:
        if c in X.columns:
            s = X[c]
            as_num = pd.to_numeric(s, errors="coerce")
            mask_ok = as_num.isin([0, 1, 2, 3, 4])
            s2 = as_num.where(mask_ok, s.astype(str).map(MAP_Q_ORD))
            X[c] = s2.fillna(-1).astype(int)
    return X

def _coerce_utilities_ordinal_inplace(X: pd.DataFrame) -> pd.DataFrame:
    if "Utilities" in X.columns:
        s = X["Utilities"]
        as_num = pd.to_numeric(s, errors="coerce")
        mask_ok = as_num.isin([0, 1, 2, 3])
        s2 = as_num.where(mask_ok, s.astype(str).map(UTIL_TO_ORD))
        X["Utilities"] = s2.fillna(-1).astype(int)
    return X

def _patch_ohe_categories_inplace(ct: ColumnTransformer) -> None:
    trs = ct.transformers_ if hasattr(ct, "transformers_") else ct.transformers
    for item in trs:
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
    from sklearn.impute import SimpleImputer as _SI
    has_fitted = hasattr(ct, "transformers_")
    trs = ct.transformers_ if has_fitted else ct.transformers
    new_trs = []
    for item in trs:
        name, transformer, cols = item[0], item[1], item[2]
        extras = item[3:] if len(item) > 3 else ()
        final_est = transformer
        if isinstance(transformer, SKPipeline):
            final_est = transformer.steps[-1][1]
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
    def __init__(self, model_path: Path = PATHS.xgb_model_file):
        self.model_path = model_path
        self.pipe_full: SKPipeline = joblib.load(self.model_path)

        self.feats = list(getattr(self.pipe_full, "feature_names_in_", []))
        self.quality_cols = [c for c in QUALITY_CANDIDATE_NAMES if c in self.feats]
        self.utilities_cols = [c for c in ["Utilities"] if c in self.feats]

        self._ttr: TransformedTargetRegressor | None = None
        last = self.pipe_full.named_steps.get("xgb")
        if isinstance(last, TransformedTargetRegressor):
            self._ttr = last
        self.log_target: bool = self._ttr is not None

        self.pre: ColumnTransformer = self.pipe_full.named_steps["pre"]

        if self._ttr is not None:
            self.reg: XGBRegressor = self._ttr.regressor_
        else:
            self.reg: XGBRegressor = self.pipe_full.named_steps["xgb"]  # type: ignore

        # === Booster: usa SOLO el que viene dentro del joblib ===
        try:
            _ = self.reg.get_booster()
        except Exception:
            raw = getattr(self.reg, "_Booster_raw", None)
            if raw is None:
                raise RuntimeError(
                    "El modelo XGB cargado no trae Booster ni _Booster_raw. "
                    "Reentrena con retrain_xgb_same_env.py (ese script guarda _Booster_raw)."
                )
            from xgboost import Booster
            bst = Booster()
            bst.load_model(raw)        # raw = buffer devuelto por save_raw()
            self.reg._Booster = bst

        # Pipeline a embedir: MISMO pre aplanado + MISMO regressor
        self.pipe_for_embed: SKPipeline = SKPipeline(steps=[
            ("pre", self.pre),
            ("xgb", self.reg),
        ])

    def feature_names_in(self) -> List[str]:
        return list(getattr(self.pipe_full, "feature_names_in_", []))

    def is_log_target(self) -> bool:
        return self.log_target

    def pipe_for_gurobi(self) -> SKPipeline:
        return self.pipe_for_embed

    def predict(self, X: pd.DataFrame) -> pd.Series:
        # Asegura calidades/utilities como int si corresponde (tú ya lo haces)
        X_fixed = X.copy()
        _coerce_quality_ordinals_inplace(X_fixed, self.quality_cols)
        _coerce_utilities_ordinal_inplace(X_fixed)
        # No hay OHE en el pre, el pipe espera solo columnas numéricas (incluyendo dummies)
        y = self.pipe_full.predict(X_fixed)
        return pd.Series(y, index=X.index)


    def attach_to_gurobi(self, m: gp.Model, x_list: list, y_log: gp.Var, eps: float = 1e-6) -> None:
        import json, math
        try:
            bst = self.reg.get_booster()
        except Exception:
            bst = getattr(self.reg, "_Booster", None)
        if bst is None:
            raise RuntimeError("XGBBundle: no pude obtener Booster del modelo")

        try:
            base = float(bst.attr("base_score") or 0.0)
        except Exception:
            base = 0.0

        dumps = bst.get_dump(with_stats=False, dump_format="json")
        total_expr = gp.LinExpr(base)

        def fin(x):
            try:
                return math.isfinite(float(x))
            except Exception:
                return False

        for t_idx, js in enumerate(dumps):
            node = json.loads(js)

            leaves = []
            def walk(nd, path):
                if "leaf" in nd:
                    leaves.append((path, float(nd["leaf"])))
                    return
                f_idx = int(str(nd["split"]).replace("f", ""))
                thr = float(nd["split_condition"])
                yes_id = nd["yes"]
                for ch in nd["children"]:
                    is_left = (ch["nodeid"] == yes_id)  # rama <=
                    walk(ch, path + [(f_idx, thr, is_left)])
            walk(node, [])

            z = [m.addVar(vtype=gp.GRB.BINARY, name=f"t{t_idx}_leaf{k}") for k in range(len(leaves))]
            m.addConstr(gp.quicksum(z) == 1, name=f"TREE_{t_idx}_ONEHOT")

            for k, (conds, val) in enumerate(leaves):
                for (f_idx, thr, is_left) in conds:
                    xv = x_list[f_idx]

                    # lb/ub efectivos para calcular M
                    lb = float(xv.LB) if fin(getattr(xv, "LB", None)) else -1e6
                    ub = float(xv.UB) if fin(getattr(xv, "UB", None)) else  1e6

                    if is_left:
                        # xv <= thr cuando z=1  -> xv <= thr + M*(1 - z)
                        M_le = max(0.0, ub - thr)
                        m.addConstr(xv <= thr + M_le * (1 - z[k]), name=f"T{t_idx}_L{k}_f{f_idx}_le")
                    else:
                        # xv >= thr + eps cuando z=1 -> xv >= thr+eps - M*(1 - z)
                        M_ge = max(0.0, thr - lb)
                        m.addConstr(xv >= thr + eps - M_ge * (1 - z[k]), name=f"T{t_idx}_R{k}_f{f_idx}_ge")

            total_expr += gp.quicksum(z[k] * leaves[k][1] for k in range(len(leaves)))

        m.addConstr(y_log == total_expr, name="YLOG_XGB_SUM")
