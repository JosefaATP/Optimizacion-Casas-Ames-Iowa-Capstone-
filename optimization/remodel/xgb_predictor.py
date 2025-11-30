# optimization/remodel/xgb_predictor.py
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
import re
import json
import gurobipy as gp

from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import FunctionTransformer

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

def _fix_base_score_if_needed(reg: XGBRegressor) -> None:
    """Algunos pickles guardan base_score como '[5E-1]'; gurobi_ml espera float."""
    try:
        bst = reg.get_booster()
    except Exception:
        return
    try:
        bs = bst.attr("base_score")
    except Exception:
        return
    def _try_fix(val_str: str) -> bool:
        try:
            val = float(val_str)
        except Exception:
            return False
        try:
            conf = json.loads(bst.save_config())
            conf["learner"]["learner_model_param"]["base_score"] = str(val)
            bst.load_config(json.dumps(conf))
            bst.set_attr(base_score=str(val))
            return True
        except Exception:
            try:
                bst.set_param({"base_score": val})
                bst.set_attr(base_score=str(val))
                return True
            except Exception:
                return False

    # 1) intenta con attr directo
    if bs is not None:
        if _try_fix(bs):
            return
        m = re.match(r"\[\s*([0-9.eE+-]+)\s*\]", str(bs))
        if m and _try_fix(m.group(1)):
            return
    # 2) intenta leyendo desde config
    try:
        conf = json.loads(bst.save_config())
        bs_conf = conf.get("learner", {}).get("learner_model_param", {}).get("base_score", None)
        if bs_conf is not None:
            if _try_fix(bs_conf):
                return
            m = re.match(r"\[\s*([0-9.eE+-]+)\s*\]", str(bs_conf))
            if m:
                _try_fix(m.group(1))
    except Exception:
        return

def _fix_passthrough_transformers(ct: ColumnTransformer) -> None:
    """
    Algunos pickles antiguos traen 'passthrough' como string en transformers_,
    y sklearn 1.7 intenta llamarle .transform. Reemplazo por FunctionTransformer(identity).
    """
    try:
        trs = ct.transformers_ if hasattr(ct, "transformers_") else ct.transformers
    except Exception:
        return

    new_trs = []
    changed = False
    for item in trs:
        name, transformer, cols = item[0], item[1], item[2]
        extras = item[3:] if len(item) > 3 else ()
        if isinstance(transformer, str) and transformer == "passthrough":
            transformer = FunctionTransformer(lambda x: x)
            changed = True
        new_trs.append((name, transformer, cols, *extras))
        try:
            if hasattr(ct, "named_transformers_"):
                ct.named_transformers_[name] = transformer
        except Exception:
            pass

    if changed:
        try:
            if hasattr(ct, "transformers_"):
                ct.transformers_ = new_trs
            else:
                ct.transformers = new_trs
        except Exception:
            pass

class XGBBundle:
    def __init__(self, model_path: Path = PATHS.xgb_model_file):
        self.model_path = model_path
        self.pipe_full: SKPipeline = joblib.load(self.model_path)
        # best_iteration/best_ntree_limit para truncar árboles si aplica
        self.n_trees_use: int | None = None
        meta_path = self.model_path.parent / "meta.json"
        try:
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                ntree = meta.get("best_ntree_limit") or meta.get("best_iteration")
                if ntree is not None:
                    try:
                        nt = int(ntree)
                        # best_ntree_limit suele ser best_iteration + 1
                        self.n_trees_use = nt if nt > 0 else None
                    except Exception:
                        self.n_trees_use = None
        except Exception:
            self.n_trees_use = None

        # Para que el embed (gurobi_ml) y las predicciones externas usen el mismo
        # número de árboles, desactivamos el truncado por early stopping.
        # (gurobi_ml no sabe recortar árboles, así que aquí igualamos ambos mundos.)
        self.n_trees_use = None

        self.feats = list(getattr(self.pipe_full, "feature_names_in_", []))
        self.quality_cols = [c for c in QUALITY_CANDIDATE_NAMES if c in self.feats]
        self.utilities_cols = [c for c in ["Utilities"] if c in self.feats]

        self._ttr: TransformedTargetRegressor | None = None
        last = self.pipe_full.named_steps.get("xgb")
        if isinstance(last, TransformedTargetRegressor):
            self._ttr = last
        self.log_target: bool = self._ttr is not None

        self.pre: ColumnTransformer = self.pipe_full.named_steps["pre"]
        _fix_passthrough_transformers(self.pre)

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
        _fix_base_score_if_needed(self.reg)

        # Parche save_raw: normaliza base_score en el JSON que consume gurobi_ml
        try:
            _orig_save_raw = self.reg.save_raw
        except Exception:
            _orig_save_raw = None
        try:
            if _orig_save_raw is None:
                def _sr(raw_format="binary"):
                    return self.reg.get_booster().save_raw(raw_format=raw_format)
                _orig_save_raw = _sr
                self.reg.save_raw = _sr  # type: ignore
            def _save_raw_safe(raw_format="binary"):
                out = _orig_save_raw(raw_format=raw_format)
                if raw_format == "json":
                    try:
                        data = json.loads(out)
                        bs = data.get("learner", {}).get("learner_model_param", {}).get("base_score")
                        if isinstance(bs, str) and "[" in bs:
                            m = re.match(r"\[\s*([0-9.eE+-]+)\s*\]", bs)
                            if m:
                                data["learner"]["learner_model_param"]["base_score"] = m.group(1)
                                return json.dumps(data)
                    except Exception:
                        return out
                return out
            self.reg.save_raw = _save_raw_safe  # type: ignore
        except Exception:
            pass

        # Pipeline a embedir: MISMO pre aplanado + MISMO regressor (SIN TransformedTargetRegressor)
        # CRÍTICO: si pipe_full tiene TransformedTargetRegressor, el embed predice en escala RAW (log1p)
        # mientras que predict() del bundle predice en escala ORIGINAL.
        # Usamos el XGBRegressor bruto para que el MIP use log1p internamente.
        self.pipe_for_embed: SKPipeline = SKPipeline(steps=[
            ("pre", self.pre),
            ("xgb", self.reg),  # XGBRegressor sin wrapper TransformedTargetRegressor
        ])
        
        # Extraer base_score del booster para usar en predict_log_raw()
        # IMPORTANTE: recalcularlo con la misma lógica que se usa en el embed
        # (predict(output_margin) - suma_de_hojas) para evitar desalineaciones.
        self.b0_offset: float = 0.0
        try:
            # Primero, valor declarado en el booster (método oficial)
            try:
                bst = self.reg.get_booster()
                bs_attr = bst.attr("base_score")
                if bs_attr is not None:
                    self.b0_offset = float(bs_attr)
            except Exception:
                self.b0_offset = 0.5

            # Luego, recalcula con la fórmula robusta para alinear con el embed manual
            b0_recalc = self._compute_base_score_from_zero(len(self.feature_names_in()))
            if np.isfinite(b0_recalc):
                self.b0_offset = float(b0_recalc)
        except Exception:
            # Último fallback: valor por defecto de XGBoost
            self.b0_offset = 0.5
        
        # INFO: pipe_for_gurobi() predice en escala LOG1P (raw XGB margin)
        # mientras que predict() predice en escala ORIGINAL (precio).
        # Esto es INTENCIONAL: el MIP trabaja con log1p internamente.

    def feature_names_in(self) -> List[str]:
        return list(getattr(self.pipe_full, "feature_names_in_", []))

    def is_log_target(self) -> bool:
        return self.log_target

    def pipe_for_gurobi(self) -> SKPipeline:
        return self.pipe_for_embed

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predice en escala original respetando n_trees_use (early stopping)."""
        X_fixed = X.copy()
        _coerce_quality_ordinals_inplace(X_fixed, self.quality_cols)
        _coerce_utilities_ordinal_inplace(X_fixed)

        try:
            Xp = self.pre.transform(X_fixed)
        except Exception:
            # fallback: pipeline completo
            y_fallback = self.pipe_full.predict(X_fixed)
            return pd.Series(y_fallback, index=X.index)

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
            y_pred = np.expm1(y_pred_log) if self.log_target else y_pred_log
            return pd.Series(y_pred, index=X.index)
        except Exception:
            y_fallback = self.pipe_full.predict(X_fixed)
            return pd.Series(y_fallback, index=X.index)

    def predict_log_raw(self, X: pd.DataFrame) -> pd.Series:
        """
        Predice el margen crudo del XGB (y_log_raw = sum_of_tree_leaves).
        X puede estar en dos formas:
        1. Raw (83 features) -> será procesado con self.pre
        2. Preprocessed (299 features) -> se usa directamente
        """
        X_fixed = X.copy()
        
        # Detectar si ya está preprocessado
        is_preprocessed = len(X_fixed.columns) == 299
        
        if not is_preprocessed:
            # Aplicar coerciones de ordinals
            _coerce_quality_ordinals_inplace(X_fixed, self.quality_cols)
            _coerce_utilities_ordinal_inplace(X_fixed)
            # Procesar con el pipeline
            try:
                Xp = self.pre.transform(X_fixed)
            except Exception:
                Xp = X_fixed.values
        else:
            # Ya está procesado, usar directamente
            Xp = X_fixed.values

        n_use = None
        try:
            n_use = int(self.n_trees_use) if self.n_trees_use is not None else None
        except Exception:
            n_use = None

        try:
            # Try using the booster's inplace_predict which should be more accurate
            if n_use is not None and n_use > 0:
                # Use iteration_range for tree limiting
                y_raw = self.reg.get_booster().inplace_predict(Xp, iteration_range=(0, n_use), predict_type='margin')
            else:
                y_raw = self.reg.get_booster().inplace_predict(Xp, predict_type='margin')
            return pd.Series(y_raw, index=X.index)
        except Exception:
            try:
                # Fallback to predict with output_margin
                if n_use is not None and n_use > 0:
                    y_raw = self.reg.predict(Xp, output_margin=True, iteration_range=(0, n_use))
                else:
                    y_raw = self.reg.predict(Xp, output_margin=True)
                return pd.Series(y_raw, index=X.index)
            except Exception as e:
                # Last resort: use full prediction and transform
                try:
                    y_pred = self.predict(X)
                    if self.log_target:
                        y_raw = np.log1p(y_pred)
                    else:
                        y_raw = y_pred
                    return y_raw
                except Exception:
                    return pd.Series([0.0] * len(X), index=X.index)
                y_log = y
            return pd.Series(y_log, index=X.index)

    # === utilidades para autocalibrar (igual que en construcción) ===
    def booster_feature_order(self) -> List[str]:
        """Orden de features que ve el Booster (passthrough numérico)."""
        cols: List[str] = []
        try:
            ct: ColumnTransformer = self.pre
            trs = ct.transformers_ if hasattr(ct, "transformers_") else ct.transformers
            for item in trs:
                name, transformer, tcols = item[0], item[1], item[2]
                if name == "num":
                    cols = list(tcols)
                    break
            if not cols and trs:
                cols = list(trs[0][2])
        except Exception:
            pass
        if not cols:
            cols = self.feature_names_in()
        return cols

    def _get_dumps_limited(self):
        try:
            bst = self.reg.get_booster()
        except Exception:
            bst = getattr(self.reg, "_Booster", None)
        if bst is None:
            raise RuntimeError("XGBBundle: no pude obtener Booster para calibración")
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
        dumps = self._get_dumps_limited()
        s = 0.0
        for js in dumps:
            node = json.loads(js)
            cur = node
            while True:
                if "leaf" in cur:
                    s += float(cur.get("leaf") or 0.0)
                    break
                f_idx = int(str(cur.get("split", "0")).replace("f", ""))
                thr = float(cur.get("split_condition", 0.0))
                yes_id = cur.get("yes")
                no_id = cur.get("no")
                val = float(Xp_row[f_idx]) if f_idx < len(Xp_row) else 0.0
                go_yes = (val < thr)
                next_id = yes_id if go_yes else no_id
                nxt = None
                for ch in cur.get("children", []):
                    if ch.get("nodeid") == next_id:
                        nxt = ch
                        break
                if nxt is None and cur.get("children"):
                    nxt = cur["children"][0]
                if nxt is None:
                    break
                cur = nxt
        return float(s)

    def autocalibrate_offset(self, X_ref: pd.DataFrame | None = None) -> float:
        """
        Para el embed manual (suma de hojas), calculamos b0 alineando con el margen del regressor.
        """
        try:
            feats = self.feature_names_in()
            if X_ref is None:
                Z = pd.DataFrame([[0.0] * len(feats)], columns=feats)
            else:
                if hasattr(X_ref, "iloc"):
                    row = X_ref.iloc[0].to_dict()
                elif isinstance(X_ref, dict):
                    row = X_ref
                else:
                    row = {c: 0.0 for c in feats}
                Z = pd.DataFrame([{c: float(row.get(c, 0.0)) for c in feats}])

            Xp = self.pre.transform(Z)
            try:
                if hasattr(Xp, "toarray"):
                    Xp_arr = Xp.toarray()
                else:
                    Xp_arr = Xp
            except Exception:
                Xp_arr = Xp

            n_use = None
            try:
                n_use = int(self.n_trees_use) if self.n_trees_use is not None else None
            except Exception:
                n_use = None

            if n_use is not None and n_use > 0:
                y_out = float(self.reg.predict(Xp_arr, output_margin=True, iteration_range=(0, n_use))[0])
            else:
                y_out = float(self.reg.predict(Xp_arr, output_margin=True)[0])
            y_in = self._eval_sum_leaves(np.ravel(Xp_arr))
            self.b0_offset = float(y_out - y_in)
        except Exception:
            self.b0_offset = 0.0
        return self.b0_offset

    def _compute_base_score_from_zero(self, n_features: int) -> float:
        """Calcula base_score como predict(margen, 0) - suma_de_hojas(0)."""
        zeros = np.zeros((1, n_features))
        y_out = float(self.reg.predict(zeros, output_margin=True)[0])
        y_in = self._eval_sum_leaves(zeros.ravel())
        return float(y_out - y_in)

    # === Embebidos de árboles (igual que construcción) ===
    def attach_to_gurobi(self, m: gp.Model, x_list: list, y_log: gp.Var, eps: float = 1e-6) -> None:
        import json, math

        # IMPORTANT: Update model to ensure variable bounds are readable
        m.update()

        try:
            bst = self.reg.get_booster()
        except Exception:
            bst = getattr(self.reg, "_Booster", None)
        if bst is None:
            raise RuntimeError("XGBBundle: no pude obtener Booster del modelo")

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

        # Recalcular siempre el base_score con la misma base que usa el embed
        try:
            self.b0_offset = self._compute_base_score_from_zero(len(x_list))
        except Exception:
            if (self.b0_offset is None) or (not np.isfinite(self.b0_offset)):
                self.b0_offset = 0.0

        total_expr = gp.LinExpr(0.0)

        def fin(v):
            try:
                return math.isfinite(float(v))
            except Exception:
                return False

        base_tiny = abs(float(eps)) if eps is not None else 1e-8

        for t_idx, js in enumerate(dumps):
            node = json.loads(js)

            leaves = []
            def walk(nd, path):
                if "leaf" in nd:
                    leaves.append((path, float(nd["leaf"])))
                    return
                f_idx = int(str(nd["split"]).replace("f", ""))
                thr = float(nd["split_condition"])
                yes_id = nd.get("yes")
                no_id = nd.get("no")
                
                # Encontrar explícitamente yes_child y no_child
                yes_child = None
                no_child = None
                for ch in nd.get("children", []):
                    ch_id = ch.get("nodeid")
                    if ch_id == yes_id:
                        yes_child = ch
                    elif ch_id == no_id:
                        no_child = ch
                
                # Procesar yes_child (x < threshold, es_left=True)
                if yes_child is not None:
                    walk(yes_child, path + [(f_idx, thr, True)])
                
                # Procesar no_child (x >= threshold, es_left=False)
                if no_child is not None:
                    walk(no_child, path + [(f_idx, thr, False)])
            walk(node, [])

            z = [m.addVar(vtype=gp.GRB.BINARY, name=f"t{t_idx}_leaf{k}") for k in range(len(leaves))]
            m.addConstr(gp.quicksum(z) == 1, name=f"TREE_{t_idx}_ONEHOT")

            # OPTIMIZATION: When all features are fixed (lb==ub), we can determine the UNIQUE correct leaf
            # by walking the tree with the fixed values
            # First, collect all features used in this tree
            tree_features = set()
            for k, (conds, _) in enumerate(leaves):
                for (f_idx, thr, is_left) in conds:
                    if f_idx < len(x_list):
                        tree_features.add(f_idx)
            
            all_fixed = all(
                fin(getattr(x_list[f_idx], "LB", None)) and 
                fin(getattr(x_list[f_idx], "UB", None)) and
                float(x_list[f_idx].LB) == float(x_list[f_idx].UB)
                for f_idx in tree_features
            )
            
            if all_fixed:
                # Compute the correct leaf index by walking the tree with fixed values
                correct_leaf_idx = None
                for k, (conds, leaf_val) in enumerate(leaves):
                    all_conds_satisfied = True
                    for (f_idx, thr, is_left) in conds:
                        xv = x_list[f_idx]
                        x_val = float(xv.LB)
                        if is_left:
                            if not (x_val < thr):  # x < thr (must be strictly less)
                                all_conds_satisfied = False
                                break
                        else:
                            if not (x_val >= thr):  # x >= thr (must be greater or equal)
                                all_conds_satisfied = False
                                break
                    
                    if all_conds_satisfied:
                        correct_leaf_idx = k
                        break
                
                # DEBUG: Log forced leaf selection for first 3 trees
                if t_idx < 3:
                    import sys
                    print(f"[TREE-EMBED-FIXED] Tree {t_idx}: all_fixed=True, forced leaf={correct_leaf_idx}, value={leaves[correct_leaf_idx][1]:.6f if correct_leaf_idx is not None else 'N/A'}", file=sys.stderr)
                
                # Force the correct leaf to be selected
                if correct_leaf_idx is not None:
                    m.addConstr(z[correct_leaf_idx] == 1, name=f"TREE_{t_idx}_FORCE_LEAF_{correct_leaf_idx}")
                    for k in range(len(leaves)):
                        if k != correct_leaf_idx:
                            m.addConstr(z[k] == 0, name=f"TREE_{t_idx}_EXCLUDE_LEAF_{k}")
            else:
                # Features are not all fixed, use normal Big-M constraints
                for k, (conds, leaf_val) in enumerate(leaves):
                    for (f_idx, thr, is_left) in conds:
                        xv = x_list[f_idx]

                        lb = float(xv.LB) if fin(getattr(xv, "LB", None)) else -1e6
                        ub = float(xv.UB) if fin(getattr(xv, "UB", None)) else  1e6

                        # Si está fija y es dummy/ordinal base, descarta la rama imposible
                        if fin(lb) and fin(ub) and lb == ub:
                            base_val = lb
                            if is_left and base_val >= thr:
                                m.addConstr(z[k] == 0, name=f"T{t_idx}_L{k}_f{f_idx}_fixed_off")
                                continue
                            if (not is_left) and base_val < thr:
                                m.addConstr(z[k] == 0, name=f"T{t_idx}_R{k}_f{f_idx}_fixed_off")
                                continue

                        # Big‑M genérico (aplica a todas las VTypes para mantener factibilidad)
                        tiny = base_tiny
                        # Asegura que cuando z=0 no se apriete por debajo del valor fijo/base
                        M_le = max(tiny, ub - thr + tiny)
                        M_ge = max(tiny, thr - lb + tiny)
                        if is_left:
                            m.addConstr(xv <= thr - tiny + M_le * (1 - z[k]), name=f"T{t_idx}_L{k}_f{f_idx}_lt")
                        else:
                            m.addConstr(xv >= thr - M_ge * (1 - z[k]), name=f"T{t_idx}_R{k}_f{f_idx}_ge")

            total_expr += gp.quicksum(z[k] * leaves[k][1] for k in range(len(leaves)))

        # The tree outputs sum to tree leaves + base_score
        # y_log_raw should be: sum of leaves (tree outputs without offset)
        m.addConstr(y_log == total_expr, name="YLOG_XGB_SUM")

    def attach_to_gurobi_strict(self, m: gp.Model, x_list: list, y_log: gp.Var, eps: float = 1e-6) -> None:
        import json, math

        try:
            bst = self.reg.get_booster()
        except Exception:
            bst = getattr(self.reg, "_Booster", None)
        if bst is None:
            raise RuntimeError("XGBBundle: no pude obtener Booster del modelo")

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
                yes_id = nd.get("yes")
                no_id = nd.get("no")
                
                # Encontrar explícitamente yes_child y no_child
                yes_child = None
                no_child = None
                for ch in nd.get("children", []):
                    ch_id = ch.get("nodeid")
                    if ch_id == yes_id:
                        yes_child = ch
                    elif ch_id == no_id:
                        no_child = ch
                
                # Procesar yes_child (x < threshold, es_left=True)
                if yes_child is not None:
                    walk(yes_child, path + [(f_idx, thr, True)])
                
                # Procesar no_child (x >= threshold, es_left=False)
                if no_child is not None:
                    walk(no_child, path + [(f_idx, thr, False)])
            walk(node, [])

            z = [m.addVar(vtype=gp.GRB.BINARY, name=f"t{t_idx}_leaf{k}") for k in range(len(leaves))]
            m.addConstr(gp.quicksum(z) == 1, name=f"TREE_{t_idx}_ONEHOT")

            for k, (conds, _) in enumerate(leaves):
                for (f_idx, thr, is_left) in conds:
                    xv = x_list[f_idx]

                    lb = float(xv.LB) if fin(getattr(xv, "LB", None)) else -1e6
                    ub = float(xv.UB) if fin(getattr(xv, "UB", None)) else  1e6


                    M_le = max(0.0, ub - thr)
                    M_ge = max(0.0, thr - lb)

                    if is_left:
                        # x <= thr cuando z=1
                        m.addConstr(xv <= thr + M_le * (1 - z[k]), name=f"T{t_idx}_L{k}_f{f_idx}_le_strict")
                    else:
                        # x >= thr cuando z=1
                        m.addConstr(xv >= thr + eps - M_ge * (1 - z[k]), name=f"T{t_idx}_R{k}_f{f_idx}_ge_strict")

            total_expr += gp.quicksum(z[k] * leaves[k][1] for k in range(len(leaves)))

        m.addConstr(y_log == total_expr, name="YLOG_XGB_SUM_STRICT")
