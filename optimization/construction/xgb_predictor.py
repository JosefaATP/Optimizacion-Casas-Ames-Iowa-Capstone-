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

        # Intentar detectar mejor cantidad de árboles con early stopping
        # Preferimos: best_ntree_limit > best_iteration+1 > meta.json
        self.n_trees_use: int | None = None
        try:
            n_limit = int(getattr(self.reg, "best_ntree_limit", -1))
            if n_limit and n_limit > 0:
                self.n_trees_use = n_limit
        except Exception:
            pass
        if self.n_trees_use is None:
            try:
                best_it = int(getattr(self.reg, "best_iteration", -1))
                if best_it and best_it >= 0:
                    self.n_trees_use = best_it + 1
            except Exception:
                pass
        if self.n_trees_use is None:
            try:
                meta_path = self.model_path.parent / "meta.json"
                if meta_path.exists():
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    bi = int(meta.get("best_iteration", -1))
                    if bi and bi >= 0:
                        self.n_trees_use = bi + 1
            except Exception:
                pass

        # Offset de calibración (b0): por defecto 0.0
        self.b0_offset: float = 0.0

    def feature_names_in(self) -> List[str]:
        return list(getattr(self.pipe_full, "feature_names_in_", []))

    def booster_feature_order(self) -> List[str]:
        """Order of features as seen by the Booster (ColumnTransformer numeric cols order)."""
        cols: List[str] = []
        try:
            ct: ColumnTransformer = self.pre
            trs = ct.transformers_ if hasattr(ct, "transformers_") else ct.transformers
            for item in trs:
                name, transformer, tcols = item[0], item[1], item[2]
                # Our preprocess builds only one numeric passthrough named 'num'
                if name == "num":
                    cols = list(tcols)
                    break
            if not cols and trs:
                # fallback: first transformer cols
                cols = list(trs[0][2])
        except Exception:
            pass
        if not cols:
            cols = self.feature_names_in()
        return cols

    # === Utilidades de evaluación del booster (para autocalibración) ===
    def _get_dumps_limited(self):
        try:
            bst = self.reg.get_booster()
        except Exception:
            bst = getattr(self.reg, "_Booster", None)
        if bst is None:
            raise RuntimeError("XGBBundle: no pude obtener Booster para calbración")
        dumps = bst.get_dump(with_stats=False, dump_format="json")
        n_use = None
        try:
            n_use = int(self.n_trees_use) if self.n_trees_use is not None else None
        except Exception:
            n_use = None
        if n_use is not None and 0 < n_use < len(dumps):
            dumps = dumps[:n_use]
        return dumps

    def _eval_sum_leaves(self, Xp_row: np.ndarray) -> float:
        # Recorre cada árbol con splits estrictos: izquierda si val < thr, derecha en caso contrario
        dumps = self._get_dumps_limited()
        s = 0.0
        for js in dumps:
            node = json.loads(js)
            cur = node
            while True:
                if "leaf" in cur:
                    s += float(cur["leaf"]) if cur["leaf"] is not None else 0.0
                    break
                f_idx = int(str(cur["split"]).replace("f", ""))
                thr = float(cur["split_condition"])
                yes_id = cur["yes"]
                no_id  = cur["no"]
                val = float(Xp_row[f_idx]) if f_idx < len(Xp_row) else 0.0
                go_yes = (val < thr)
                next_id = yes_id if go_yes else no_id
                # avanzar al hijo solicitado
                next_node = None
                for ch in cur.get("children", []):
                    if ch.get("nodeid") == next_id:
                        next_node = ch
                        break
                if next_node is None:
                    # fallback: primer hijo
                    next_node = cur.get("children", [cur])[0]
                cur = next_node
        return float(s)

    def autocalibrate_offset(self, X_ref: pd.DataFrame | None = None) -> float:
        """
        Calcula un offset b0 para alinear el embed (suma de hojas) con el margen del regressor
        usando una fila de referencia. Si no se entrega X_ref, usa un vector cero con las
        columnas feature_names_in(). Devuelve b0 y lo deja en self.b0_offset.
        """
        try:
            feats = self.feature_names_in()
            if X_ref is None:
                Z = pd.DataFrame([[0.0] * len(feats)], columns=feats)
            else:
                # Asegura que Z tenga mismas columnas (faltantes a 0)
                Z = pd.DataFrame([{c: float(X_ref.get(c, 0.0)) for c in feats}])

            # Transformación del pre (pasa las columnas numéricas tal cual)
            try:
                Xp = self.pre.transform(Z)
            except Exception:
                Xp = Z.values

            # y_out: margen del XGB
            n_use = int(self.n_trees_use) if self.n_trees_use is not None else None
            if n_use and n_use > 0:
                y_out = float(self.reg.predict(Xp, output_margin=True, iteration_range=(0, n_use))[0])
            else:
                y_out = float(self.reg.predict(Xp, output_margin=True)[0])

            # y_in: suma de hojas emulando embed estricto
            Xp_row = np.ravel(Xp)
            y_in = self._eval_sum_leaves(Xp_row)

            self.b0_offset = float(y_out - y_in)
        except Exception:
            self.b0_offset = 0.0
        return self.b0_offset

    def is_log_target(self) -> bool:
        return self.log_target

    def pipe_for_gurobi(self) -> SKPipeline:
        return self.pipe_for_embed

    def predict(self, X: pd.DataFrame) -> pd.Series:
        # Asegura calidades/utilities como int si corresponde
        X_fixed = X.copy()
        _coerce_quality_ordinals_inplace(X_fixed, self.quality_cols)
        _coerce_utilities_ordinal_inplace(X_fixed)

        # PREDICCIÓN CONSISTENTE CON EARLY STOPPING
        # Transformar X con el preprocesador del pipeline cargado
        try:
            Xp = self.pre.transform(X_fixed)
        except Exception:
            # si el preprocesador no está fitted en este objeto (debería), caer al pipe_full
            y_fallback = self.pipe_full.predict(X_fixed)
            return pd.Series(y_fallback, index=X.index)

    def predict_log_raw(self, X: pd.DataFrame) -> pd.Series:
        """Predice la salida cruda del XGB (margin = y_log), sin inverse_func.
        Respeta early stopping usando iteration_range si está disponible.
        """
        X_fixed = X.copy()
        _coerce_quality_ordinals_inplace(X_fixed, self.quality_cols)
        _coerce_utilities_ordinal_inplace(X_fixed)
        try:
            Xp = self.pre.transform(X_fixed)
        except Exception:
            # último recurso: que al menos no rompa
            Xp = X_fixed.values
        n_use = None
        try:
            n_use = int(self.n_trees_use) if self.n_trees_use is not None else None
        except Exception:
            n_use = None
        try:
            if n_use is not None and n_use > 0:
                y_log = self.reg.predict(Xp, output_margin=True, iteration_range=(0, n_use))
            else:
                y_log = self.reg.predict(Xp, output_margin=True)
            return pd.Series(y_log, index=X.index)
        except Exception:
            # fallback: usar el pipe completo y revertir (si es log_target)
            y = self.pipe_full.predict(X_fixed)
            try:
                import numpy as _np
                y_log = _np.log1p(y)
            except Exception:
                y_log = y
            return pd.Series(y_log, index=X.index)

        # Usar el mismo número de árboles que en el embed si está disponible
        n_use = None
        try:
            n_use = int(self.n_trees_use) if self.n_trees_use is not None else None
        except Exception:
            n_use = None

        try:
            if n_use is not None and n_use > 0:
                y_pred_log = self.reg.predict(Xp, iteration_range=(0, n_use))
            else:
                y_pred_log = self.reg.predict(Xp)
            if self.log_target:
                y_pred = np.expm1(y_pred_log)
            else:
                y_pred = y_pred_log
            return pd.Series(y_pred, index=X.index)
        except Exception:
            # último recurso: pipe completo
            y_fallback = self.pipe_full.predict(X_fixed)
            return pd.Series(y_fallback, index=X.index)


    def attach_to_gurobi(self, m: gp.Model, x_list: list, y_log: gp.Var, eps: float = 1e-6) -> None:
        import json, math

        try:
            bst = self.reg.get_booster()
        except Exception:
            bst = getattr(self.reg, "_Booster", None)
        if bst is None:
            raise RuntimeError("XGBBundle: no pude obtener Booster del modelo")

        try:
            _ = float(bst.attr("base_score") or 0.0)
        except Exception:
            _ = 0.0

        dumps = bst.get_dump(with_stats=False, dump_format="json")
        # Si hay early stopping, usa solo los primeros n arboles
        n_use = None
        try:
            n_use = int(self.n_trees_use) if self.n_trees_use is not None else None
        except Exception:
            n_use = None
        if n_use is not None and 0 < n_use < len(dumps):
            dumps = dumps[:n_use]
            try:
                m._xgb_used_trees = int(n_use)
            except Exception:
                pass
        # Nota: usamos 0.0 como base y luego, si hay b0_offset, lo aplicamos aparte.
        total_expr = gp.LinExpr(0.0)

        def fin(v):
            try:
                return math.isfinite(float(v))
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
                yes_id = nd["yes"]  # hijo que toma la rama <= thr
                for ch in nd["children"]:
                    is_left = (ch["nodeid"] == yes_id)
                    walk(ch, path + [(f_idx, thr, is_left)])
            walk(node, [])

            z = [m.addVar(vtype=gp.GRB.BINARY, name=f"t{t_idx}_leaf{k}") for k in range(len(leaves))]
            m.addConstr(gp.quicksum(z) == 1, name=f"TREE_{t_idx}_ONEHOT")

            for k, (conds, _) in enumerate(leaves):
                for (f_idx, thr, is_left) in conds:
                    xv = x_list[f_idx]

                    # bounds efectivos
                    lb = float(xv.LB) if fin(getattr(xv, "LB", None)) else -1e6
                    ub = float(xv.UB) if fin(getattr(xv, "UB", None)) else  1e6

                    # Dummies (0/1): usar umbral canónico 0.5 para alinear con OHE
                    # Esto reduce discrepancias embed vs fuera cuando XGB guarda thresholds pegados a 0 o 1.
                    if lb >= 0.0 and ub <= 1.0:
                        thr = 0.5

                    # M dirigidos
                    M_le = max(0.0, ub - thr)
                    M_ge = max(0.0, thr - lb)

                    if is_left:
                        # x <= thr cuando z=1  → x <= thr + M*(1 - z)
                        m.addConstr(xv <= thr + M_le * (1 - z[k]), name=f"T{t_idx}_L{k}_f{f_idx}_le")
                    else:
                        # x >= thr cuando z=1  → x >= thr - M*(1 - z)
                        # NOTA: sin +eps aqui para no dejar residuo cuando z=0
                        m.addConstr(xv >= thr - M_ge * (1 - z[k]), name=f"T{t_idx}_R{k}_f{f_idx}_ge")

            total_expr += gp.quicksum(z[k] * leaves[k][1] for k in range(len(leaves)))

        m.addConstr(y_log == total_expr, name="YLOG_XGB_SUM")

    def attach_to_gurobi_strict(self, m: gp.Model, x_list: list, y_log: gp.Var, eps: float = 1e-6) -> None:
        import json, math

        try:
            bst = self.reg.get_booster()
        except Exception:
            bst = getattr(self.reg, "_Booster", None)
        if bst is None:
            raise RuntimeError("XGBBundle: no pude obtener Booster del modelo")

        try:
            _ = float(bst.attr("base_score") or 0.0)
        except Exception:
            _ = 0.0

        dumps = bst.get_dump(with_stats=False, dump_format="json")
        n_use = None
        try:
            n_use = int(self.n_trees_use) if self.n_trees_use is not None else None
        except Exception:
            n_use = None
        if n_use is not None and 0 < n_use < len(dumps):
            dumps = dumps[:n_use]
            try:
                m._xgb_used_trees = int(n_use)
            except Exception:
                pass
        # Nota: muchos dumps no requieren sumar 'base_score' explícito; usamos 0.0 para evitar sesgos.
        total_expr = gp.LinExpr(0.0)

        def fin(v):
            try:
                return math.isfinite(float(v))
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
                for ch in nd.get("children", []):
                    is_left = (ch.get("nodeid") == yes_id)
                    walk(ch, path + [(f_idx, thr, is_left)])
            walk(node, [])

            z = [m.addVar(vtype=gp.GRB.BINARY, name=f"t{t_idx}_leaf{k}") for k in range(len(leaves))]
            m.addConstr(gp.quicksum(z) == 1, name=f"TREE_{t_idx}_ONEHOT")

            for k, (conds, _) in enumerate(leaves):
                for (f_idx, thr, is_left) in conds:
                    xv = x_list[f_idx]
                    lb = float(getattr(xv, "LB", -1e6)) if fin(getattr(xv, "LB", None)) else -1e6
                    ub = float(getattr(xv, "UB",  1e6)) if fin(getattr(xv, "UB", None)) else  1e6

                    if is_left:
                        thr_left = thr - eps
                        M_le = max(0.0, ub - thr_left)
                        m.addConstr(xv <= thr_left + M_le * (1 - z[k]), name=f"T{t_idx}_L{k}_f{f_idx}_le")
                    else:
                        M_ge = max(0.0, thr - lb)
                        m.addConstr(xv >= thr - M_ge * (1 - z[k]), name=f"T{t_idx}_R{k}_f{f_idx}_ge")

            total_expr += gp.quicksum(z[k] * leaves[k][1] for k in range(len(leaves)))

        m.addConstr(y_log == total_expr, name="YLOG_XGB_SUM_STRICT")
