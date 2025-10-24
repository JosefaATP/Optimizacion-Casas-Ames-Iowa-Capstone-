# training/bayes_tune_with_retrain.py
import os, json, random, argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Ejecuta siempre como módulo desde la raíz del repo:
#   python -m training.bayes_tune_with_retrain --csv data/processed/base_completa_sin_nulos.csv
#
# Requisitos:
#   - scikit-optimize instalado (skopt)
#   - training.funcion.retrain_xgb disponible (tu función de entrenamiento)

# Semillas para reproducibilidad total
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

from training.funcion import retrain_xgb   # ✅ usa la función desde funcion.py
from training.config import Config
from training.metrics import regression_report
regression_report  # solo para import estático

# ---- skopt ----
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# -------- Configuración por defecto --------
DEFAULT_BASE_OUTDIR = "models/xgb_bayes_search_2"
DEFAULT_N_CALLS = 80
DEFAULT_R2_THRESHOLD = 0.93

# Espacio de búsqueda (ajústalo como quieras)
# Espacio de búsqueda (ajústalo como quieras)
SEARCH_SPACE = [
    Integer(1200, 4000,  name="n_estimators"),           # ↑ techo de árboles
    Real(0.02, 0.07,     prior="log-uniform", name="learning_rate"),  # LR un toque más rápido
    Integer(3, 7,        name="max_depth"),              # permite árboles algo más profundos
    Integer(4, 14,       name="min_child_weight"),       # hojas un poco menos/más grandes
    Real(0.0, 3.0,       name="gamma"),                  # poda más laxa (0) hasta moderada
    Real(0.65, 1.0,      name="subsample"),              # más datos por árbol (varianza ↑ controlada)
    Real(0.4, 1.0,       name="colsample_bytree"),       # más columnas por árbol (captura más señal)
    Real(0.5, 4.0,       name="reg_lambda"),             # L2 menos timorata (puede suavizar si overfit)
    Real(0.05, 1.2,      prior="log-uniform", name="reg_alpha"),  # L1 un poco más amplio
]


# Globals de ejecución
_iteration = 0
_high_performing = []
_args = None  # se llena en main()

# Progresivos
_best_so_far = -float("inf")
_records_path_jsonl = None
_records_path_csv = None

# >>> ADDED: archivo con TODAS las iteraciones
_all_iter_csv = None
# <<< ADDED


def _cfg_with(params: dict) -> Config:
    """
    Construye un Config y mezcla sus xgb_params con los sugeridos por la búsqueda.
    NO toca tu archivo original; solo ajusta el objeto en memoria.
    """
    cfg = Config()
    merged = dict(cfg.xgb_params)  # base del Config
    merged.update(params)          # override con lo propuesto
    # Garantizar algunos campos deterministas si quieres anclar
    merged.setdefault("objective", "reg:squarederror")
    merged.setdefault("tree_method", "hist")
    merged.setdefault("random_state", SEED)
    merged.setdefault("n_jobs", -1)
    cfg.xgb_params = merged
    return cfg


def _normalize(params: dict) -> dict:
    """Asegura tipos correctos para XGB."""
    out = dict(params)
    if "n_estimators" in out: out["n_estimators"] = int(out["n_estimators"])
    if "max_depth" in out:    out["max_depth"]    = int(out["max_depth"])
    for k in ["learning_rate","subsample","colsample_bytree","gamma","min_child_weight","reg_alpha","reg_lambda"]:
        if k in out: out[k] = float(out[k])
    return out


@use_named_args(SEARCH_SPACE)
def objective(**suggested_params):
    global _iteration, _high_performing, _args
    global _best_so_far, _records_path_jsonl, _records_path_csv
    # >>> ADDED: para escribir todas las iteraciones
    global _all_iter_csv
    # <<< ADDED

    _iteration += 1

    params = _normalize(suggested_params)
    cfg = _cfg_with(params)

    # Carpeta específica del trial (se guardan modelo, meta, metrics, etc. del retrain)
    trial_outdir = os.path.join(_args.base_outdir, f"trial_{_iteration:03d}")
    Path(trial_outdir).mkdir(parents=True, exist_ok=True)

    print(f"\n=== Trial #{_iteration} / { _args.n_calls } ===")
    print("Params:", params)
    # Entrena usando *exactamente* tu pipeline del primer archivo
    artefacts = retrain_xgb(
        csv_path=_args.csv,
        outdir=trial_outdir,
        cfg=cfg,
        verbose=False,  # silenciamos; puedes poner True si quieres ver métricas
    )

    # Leer metrics.json completo del retrain
    trial_metrics = None
    try:
        with open(artefacts["paths"]["metrics"], "r") as f:
            trial_metrics = json.load(f)
    except Exception:
        trial_metrics = artefacts.get("metrics", None)

    # Usa el R2 del retrain para consistencia total
    r2 = float(trial_metrics["test"]["R2"] if trial_metrics else artefacts["metrics"]["test"]["R2"])
    print(f"R2 test = {r2:.6f}  |  modelo: {artefacts['paths']['model']}")

    # >>> ADDED: extraer métricas del test para csvs
    test_m = (trial_metrics or {}).get("test", {}) if trial_metrics else {}
    r2_test = float(test_m.get("R2", np.nan))
    rmse_test = float(test_m.get("RMSE", test_m.get("rmse", np.nan)))
    mae_test  = float(test_m.get("MAE",  test_m.get("mae",  np.nan)))
    mape_test = float(test_m.get("MAPE_pct", test_m.get("MAPE", test_m.get("mape", np.nan))))
    res_skew  = float(test_m.get("residual_skew", np.nan))
    res_kurt  = float(test_m.get("residual_kurtosis", np.nan))
    # <<< ADDED

    # Guardar finalistas con métricas completas
    if r2 >= _args.r2_threshold:
        _high_performing.append({
            "iteration": _iteration,
            "r2_score": r2,
            "params": params,
            "model_path": artefacts["paths"]["model"],
            "meta_path": artefacts["paths"]["meta"],
            "metrics_path": artefacts["paths"]["metrics"],
            "metrics": trial_metrics,  # métricas completas del retrain (incluye test/train, skew, kurtosis, etc.)
        })

    # >>> ADDED: CSV con TODAS las iteraciones (se reinicia en main)
    with open(_all_iter_csv, "a") as f_all:
        f_all.write(f"{_iteration},{r2_test},{rmse_test},{mae_test},{mape_test},{res_skew},{res_kurt}\n")
    # <<< ADDED

    # Récords progresivos (jsonl + csv) con TODAS las métricas de test
    EPS = 1e-12
    if r2 > _best_so_far + EPS:
        _best_so_far = r2

        # Extraer métricas ya calculadas arriba (r2_test, rmse_test, etc.)
        record = {
            "iteration": _iteration,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "r2": float(r2),
            "params": params,
            "model_path": artefacts["paths"]["model"],
            "meta_path": artefacts["paths"]["meta"],
            "metrics_path": artefacts["paths"]["metrics"],
            "metrics": trial_metrics,   # guardamos el bloque completo también
            # Métricas planas de test para consumo rápido:
            "r2_test": r2_test,
            "rmse_test": rmse_test,
            "mae_test": mae_test,
            "mape_test": mape_test,
            "residual_skew": res_skew,
            "residual_kurtosis": res_kurt,
        }

        # JSONL (todas las columnas disponibles)
        with open(_records_path_jsonl, "a") as fj:
            fj.write(json.dumps(record) + "\n")

        # CSV (columnas básicas para graficar rápido)
        with open(_records_path_csv, "a") as fc:
            fc.write(
                f"{_iteration},{record['timestamp']},{r2},{rmse_test},{mae_test},{mape_test},{res_skew},{res_kurt}\n"
            )

        print(f"✨ Nuevo récord de R2: {r2:.6f} (iter #{_iteration}) → progressive_records.* actualizados")

    # gp_minimize minimiza → devolvemos el negativo del R2
    return -r2


def main():
    global _args, _high_performing
    global _records_path_jsonl, _records_path_csv, _best_so_far
    # >>> ADDED: ruta del archivo con todas las iteraciones
    global _all_iter_csv
    # <<< ADDED

    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Ruta al CSV limpio (mismo que usa retrain_xgb).")
    p.add_argument("--base-outdir", default=DEFAULT_BASE_OUTDIR, help="Carpeta raíz donde se guardan los trials.")
    p.add_argument("--n-calls", type=int, default=DEFAULT_N_CALLS, help="Iteraciones de gp_minimize.")
    p.add_argument("--r2-threshold", type=float, default=DEFAULT_R2_THRESHOLD, help="Umbral para guardar finalistas.")
    p.add_argument("--n-initial-points", type=int, default=25, help="Puntos iniciales aleatorios para BO.")
    args = p.parse_args()

    _args = args
    Path(args.base_outdir).mkdir(parents=True, exist_ok=True)

    # Inicializar archivos (REINICIAR)
    _records_path_jsonl = os.path.join(args.base_outdir, "progressive_records.jsonl")
    _records_path_csv   = os.path.join(args.base_outdir, "progressive_records.csv")
    # >>> ADDED: all_iterations.csv
    _all_iter_csv       = os.path.join(args.base_outdir, "all_iterations.csv")
    # <<< ADDED

    # Borrar si existen
    for pth in (_records_path_jsonl, _records_path_csv, _all_iter_csv):
        try:
            if os.path.exists(pth):
                os.remove(pth)
        except Exception:
            pass

    # Recrear con headers
    Path(_records_path_jsonl).touch()
    with open(_records_path_csv, "w") as f:
        f.write("iteration,timestamp,r2,rmse,mae,mape,residual_skew,residual_kurtosis\n")
    # >>> ADDED: header de all_iterations.csv
    with open(_all_iter_csv, "w") as f:
        f.write("iteration,r2,rmse,mae,mape,residual_skew,residual_kurtosis\n")
    # <<< ADDED

    _best_so_far = -float("inf")

    print("="*70)
    print("Búsqueda Bayesiana sobre XGBoost usando el entrenamiento oficial (retrain_xgb)")
    print(f"CSV: {args.csv}")
    print(f"Trials: {args.n_calls}  |  Finalistas si R2 ≥ {args.r2_threshold}")
    print("="*70)

    result = gp_minimize(
        func=objective,
        dimensions=SEARCH_SPACE,
        n_calls=args.n_calls,
        random_state=SEED,
        n_initial_points=args.n_initial_points,
        acq_func="EI",
    )

    best_r2 = -result.fun
    best_params = {dim.name: val for dim, val in zip(SEARCH_SPACE, result.x)}

    # Buscar cuál iteración alcanzó este mejor R2
    best_trial = None
    for item in _high_performing:
        if abs(item["r2_score"] - best_r2) < 1e-6:  # tolerancia por redondeo
            best_trial = item["iteration"]
            break

    if best_trial is not None:
        print(f"\n💎 El mejor resultado se obtuvo en la iteración #{best_trial}")
    else:
        print("\n⚠️ No se encontró la iteración del mejor resultado (posiblemente no superó el umbral de guardado)")

    print("\n" + "="*70)
    print("🏆 Mejor combinación encontrada")
    print(f"R2 test = {best_r2:.6f}")
    for k, v in best_params.items():
        print(f"  - {k}: {v}")

    # Incluir métricas completas del mejor si está en finalistas
    best_metrics = None
    if best_trial is not None:
        try:
            best_metrics = next((it.get("metrics") for it in _high_performing if it["iteration"] == best_trial), None)
        except Exception:
            best_metrics = None

    # Guardar resumen
    summary = {
        "best": {
            "r2": float(best_r2),
            "params": _normalize(best_params),
            "iteration": best_trial,
            "metrics": best_metrics,  # métricas completas del mejor
        },
        "finalists": sorted(_high_performing, key=lambda d: d["r2_score"], reverse=True),  # con métricas
        "seed": SEED,
        "n_calls": args.n_calls,
    }
    out_json = os.path.join(args.base_outdir, "bayes_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResumen guardado en: {out_json}")
    print(f"Progresión guardada en: {_records_path_jsonl} y {_records_path_csv}")
    # >>> ADDED: confirmación de all_iterations
    print(f"Todas las iteraciones guardadas en: {_all_iter_csv}")
    # <<< ADDED


if __name__ == "__main__":
    main()
