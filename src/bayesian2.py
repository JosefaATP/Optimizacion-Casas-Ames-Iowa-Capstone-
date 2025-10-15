"""
Bayesian Optimization for XGBoost (log-target optional) with deterministic training.

- Usa una función común de entrenamiento (xgb_log_train) que fija semillas, fuerza n_jobs=1,
  evita submuestreos y soporta early stopping.
- Optimiza con bayes_opt (BayesianOptimization). Si no está instalado, puedes
  `pip install bayesian-optimization`.
- Registra cada evaluación con métricas (RMSE/MAE/R2), tiempo y hashes de split/labels.
- Permite guardar resultados en CSV/JSON y exportar el mejor modelo (joblib).

Autor: ChatGPT (Ignacio, esta versión está pensada para integrarse fácilmente en tu pipeline)
"""
from __future__ import annotations

import os
import json
import time
import math
import hashlib
import random
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor
import joblib
from bayes_opt import BayesianOptimization




try:
    from bayes_opt import BayesianOptimization
    _HAS_BAYES_OPT = True
except Exception:
    _HAS_BAYES_OPT = False


# ==========================
# Utilidades de reproducibilidad
# ==========================

def set_seeds(seed: int = 42, single_thread: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = "0"
    np.random.seed(seed)
    random.seed(seed)
    if single_thread:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"


def md5_hash_arr(a: np.ndarray) -> str:
    m = hashlib.md5()
    m.update(np.ascontiguousarray(a).data)
    return m.hexdigest()


# ==========================
# Entrenamiento determinista
# ==========================

def make_xgb(xgb_params: Dict[str, Any], seed: int) -> XGBRegressor:
    # Fuerza parámetros críticos para reproducibilidad
    base = dict(
        random_state=seed,
        seed=seed,
        n_jobs=1,
        objective=xgb_params.get("objective", "reg:squarederror"),
        tree_method=xgb_params.get("tree_method", "hist"),
        predictor=xgb_params.get("predictor", "cpu_predictor"),
        subsample=xgb_params.get("subsample", 1.0),
        colsample_bytree=xgb_params.get("colsample_bytree", 1.0),
        colsample_bylevel=xgb_params.get("colsample_bylevel", 1.0),
        colsample_bynode=xgb_params.get("colsample_bynode", 1.0),
    )
    # Mezcla preservando lo que venga en xgb_params
    merged = {**base, **xgb_params}
    return XGBRegressor(**merged)


def xgb_log_train(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    xgb_params: Dict[str, Any],
    log_target: bool = True,
    seed: int = 42,
    eval_metric: str = "rmse",
    early_stopping_rounds: int = 100,
) -> Tuple[Dict[str, Any], Any]:
    """Entrena XGB de forma determinista (log-transform opcional) y devuelve métricas y el modelo.

    Devuelve (metrics, model), donde metrics contiene rmse, mae, r2, best_ntree_limit y secs.
    """
    set_seeds(seed, single_thread=True)

    reg = make_xgb(xgb_params, seed=seed)

    model: Any
    fit_kwargs: Dict[str, Any]

    if log_target:
        model = TransformedTargetRegressor(
            regressor=reg,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )
        fit_kwargs = {
            "reg__eval_set": [(X_val, y_val)],
            "reg__eval_metric": eval_metric,
            "reg__early_stopping_rounds": early_stopping_rounds,
            "reg__verbose": False,
        }
    else:
        model = reg
        fit_kwargs = {
            "eval_set": [(X_val, y_val)],
            "eval_metric": eval_metric,
            "early_stopping_rounds": early_stopping_rounds,
            "verbose": False,
        }

    t0 = time.time()
    model.fit(X_tr, y_tr, **fit_kwargs)
    secs = time.time() - t0

    y_hat = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_hat, squared=False)
    mae = mean_absolute_error(y_val, y_hat)
    r2 = r2_score(y_val, y_hat)

    if isinstance(model, TransformedTargetRegressor):
        best_ntree = getattr(model.regressor_, "best_ntree_limit", None)
    else:
        best_ntree = getattr(model, "best_ntree_limit", None)

    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "best_ntree_limit": int(best_ntree) if best_ntree is not None else None,
        "secs": round(float(secs), 4),
    }
    return metrics, model


# ==========================
# Espacio de búsqueda y wrapper BO
# ==========================

@dataclass
class SearchConfig:
    # Rango continuo/log-escala típico
    learning_rate: Tuple[float, float] = (0.01, 0.3)
    max_depth: Tuple[int, int] = (3, 10)
    min_child_weight: Tuple[float, float] = (1.0, 10.0)
    gamma: Tuple[float, float] = (0.0, 5.0)
    subsample: Tuple[float, float] = (1.0, 1.0)  # fijo en 1 para determinismo
    colsample_bytree: Tuple[float, float] = (1.0, 1.0)
    reg_alpha: Tuple[float, float] = (0.0, 1.0)
    reg_lambda: Tuple[float, float] = (0.5, 5.0)
    n_estimators: Tuple[int, int] = (200, 2000)


def _cast_params(params: Dict[str, float]) -> Dict[str, Any]:
    """Castea a tipos correctos (enteros vs floats) y fija determinismo por defecto."""
    casted = {
        "learning_rate": float(params["learning_rate"]),
        "max_depth": int(round(params["max_depth"])),
        "min_child_weight": float(params["min_child_weight"]),
        "gamma": float(params["gamma"]),
        "subsample": float(params.get("subsample", 1.0)),
        "colsample_bytree": float(params.get("colsample_bytree", 1.0)),
        "reg_alpha": float(params["reg_alpha"]),
        "reg_lambda": float(params["reg_lambda"]),
        "n_estimators": int(round(params["n_estimators"])),
        # Objetivo & método de árbol por defecto
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "predictor": "cpu_predictor",
    }
    # Seguridad para determinismo
    casted["subsample"] = 1.0
    casted["colsample_bytree"] = 1.0
    return casted


@dataclass
class RunConfig:
    seed: int = 42
    test_size: float = 0.2
    eval_metric: str = "rmse"
    early_stopping_rounds: int = 100
    init_points: int = 10
    n_iter: int = 30
    acq_kind: str = "ei"  # ei, poi, ucb
    kappa: float = 2.5
    xi: float = 0.01
    log_target: bool = True
    out_dir: str = "bayes_out"


class XGBBayesSearch:
    def __init__(self, search_cfg: SearchConfig, run_cfg: RunConfig):
        if not _HAS_BAYES_OPT:
            raise ImportError(
                "No se encontró 'bayesian-optimization'. Instala con: pip install bayesian-optimization"
            )
        self.sc = search_cfg
        self.rc = run_cfg
        os.makedirs(self.rc.out_dir, exist_ok=True)

    def _split(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        set_seeds(self.rc.seed)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=self.rc.test_size, random_state=self.rc.seed
        )
        # hashes
        idx_tr = np.arange(len(X_tr))
        idx_val = np.arange(len(X_val))
        self.hash_train_idx = md5_hash_arr(idx_tr)
        self.hash_test_idx = md5_hash_arr(idx_val)
        self.hash_y_train = md5_hash_arr(y_tr.astype(np.float64))
        self.hash_y_test = md5_hash_arr(y_val.astype(np.float64))
        return X_tr, X_val, y_tr, y_val

    def _bounds(self) -> Dict[str, Tuple[float, float]]:
        sc = self.sc
        return {
            "learning_rate": sc.learning_rate,
            "max_depth": sc.max_depth,
            "min_child_weight": sc.min_child_weight,
            "gamma": sc.gamma,
            "subsample": sc.subsample,
            "colsample_bytree": sc.colsample_bytree,
            "reg_alpha": sc.reg_alpha,
            "reg_lambda": sc.reg_lambda,
            "n_estimators": sc.n_estimators,
        }

    def optimize(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        X_tr, X_val, y_tr, y_val = self._split(X, y)

        logs: list[Dict[str, Any]] = []
        best_score = math.inf
        best_params: Optional[Dict[str, Any]] = None
        best_model: Optional[Any] = None

        def black_box(**params) -> float:
            nonlocal best_score, best_params, best_model
            xgb_params = _cast_params(params)
            metrics, model = xgb_log_train(
                X_tr, y_tr, X_val, y_val,
                xgb_params=xgb_params,
                log_target=self.rc.log_target,
                seed=self.rc.seed,
                eval_metric=self.rc.eval_metric,
                early_stopping_rounds=self.rc.early_stopping_rounds,
            )
            row = {**xgb_params, **metrics}
            logs.append(row)
            # Queremos MINIMIZAR RMSE, pero BO maximiza → usamos negativo
            rmse = metrics["rmse"]
            if rmse < best_score:
                best_score = rmse
                best_params = xgb_params
                best_model = model
            return -rmse

        optimizer = BayesianOptimization(
            f=black_box,
            pbounds=self._bounds(),
            random_state=self.rc.seed,
            allow_duplicate_points=True,
            verbose=2,
        )
        
        optimizer.maximize(init_points=self.rc.init_points, n_iter=self.rc.n_iter, acq=self.rc.acq_kind)

        # Persistencia
        df = pd.DataFrame(logs).sort_values("rmse").reset_index(drop=True)
        csv_path = os.path.join(self.rc.out_dir, "bayes_trials.csv")
        df.to_csv(csv_path, index=False)

        meta = {
            "best_rmse": float(best_score),
            "best_params": best_params,
            "hash_train_idx": self.hash_train_idx,
            "hash_test_idx": self.hash_test_idx,
            "hash_y_train": self.hash_y_train,
            "hash_y_test": self.hash_y_test,
            "config": {"search": asdict(self.sc), "run": asdict(self.rc)},
            "csv_path": csv_path,
        }
        with open(os.path.join(self.rc.out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # Guarda el mejor modelo
        model_path = os.path.join(self.rc.out_dir, "best_model.joblib")
        joblib.dump(best_model, model_path)
        meta["model_path"] = model_path
        return meta


# ==========================
# API de alto nivel
# ==========================

def bayesian_xgb_log_optimize(
    X: np.ndarray,
    y: np.ndarray,
    search_cfg: Optional[SearchConfig] = None,
    run_cfg: Optional[RunConfig] = None,
) -> Dict[str, Any]:
    """Optimiza XGB con BO y devuelve metadatos (mejores params, hashes, paths)."""
    if search_cfg is None:
        search_cfg = SearchConfig()
    if run_cfg is None:
        run_cfg = RunConfig()
    search = XGBBayesSearch(search_cfg, run_cfg)
    return search.optimize(X, y)


# ==========================
# CLI mínimo (ejemplo)
# ==========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bayesian XGB (log target opcional)")
    parser.add_argument("--npz", type=str, default=None,
                        help="Ruta a un .npz con arrays 'X' y 'y' (float/numéricos). Si no se pasa, se genera sintético.")
    parser.add_argument("--out_dir", type=str, default="bayes_out")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--log_target", action="store_true")
    parser.add_argument("--init_points", type=int, default=10)
    parser.add_argument("--n_iter", type=int, default=30)
    args = parser.parse_args()

    # Carga datos
    if args.npz and os.path.exists(args.npz):
        data = np.load(args.npz)
        X = data["X"]
        y = data["y"]
    else:
        # Demo sintético
        rng = np.random.default_rng(args.seed)
        X = rng.normal(size=(4000, 20))
        coef = rng.uniform(-2, 2, size=20)
        noise = rng.normal(scale=0.5, size=4000)
        y = X @ coef + noise
        y = np.maximum(y, 0)  # evitar negativos si luego usamos log1p

    sc = SearchConfig(
        learning_rate=(0.01, 0.2),
        max_depth=(3, 9),
        min_child_weight=(1.0, 10.0),
        gamma=(0.0, 3.0),
        subsample=(1.0, 1.0),
        colsample_bytree=(1.0, 1.0),
        reg_alpha=(0.0, 1.0),
        reg_lambda=(0.5, 5.0),
        n_estimators=(200, 1500),
    )

    rc = RunConfig(
        seed=args.seed,
        test_size=args.test_size,
        log_target=args.log_target,
        out_dir=args.out_dir,
        init_points=args.init_points,
        n_iter=args.n_iter,
        eval_metric="rmse",
        early_stopping_rounds=100,
        acq_kind="ei",
        kappa=2.5,
        xi=0.01,
    )

    meta = bayesian_xgb_log_optimize(X, y, sc, rc)
    print("\n=== RESUMEN ===")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nCSV de trials: {meta['csv_path']}")
    print(f"Mejor modelo: {meta['model_path']}")
