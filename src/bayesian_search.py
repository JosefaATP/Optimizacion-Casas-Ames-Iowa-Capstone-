import pandas as pd
import json
import os
import joblib
import hashlib, numpy as np
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor # Se usa el de tu script
from xgboost import XGBRegressor
from preprocess import infer_feature_types, build_preprocessor
from sklearn.compose import TransformedTargetRegressor


from skopt.utils import use_named_args

import numpy as np
import random
import os

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)



# --- Parámetros XGB fijos: DEBEN matchear train_xgb_log.py ---
FIXED_XGB_PARAMS = {
    "objective": "reg:squarederror",  # igual al final
    "eval_metric": "rmse",            # r2 lo calculamos afuera; XGB usa rmse interno por consistencia
    "booster": "gbtree",
    "tree_method": "hist",            # rápido y determinista en CPU
    "verbosity": 0,
    "random_state": 42,               # se re-escribe también dentro de objective
    "seed": 42,                       # idem
    "n_jobs": 1,                      # evita no-determinismo
    # Si en train_xgb_log NO usas early stopping, mantenlo desactivado aquí
    # Si SÍ usas early stopping en el final, entonces debemos replicarlo también (ver nota al final)
}



# Librerías para la búsqueda Bayesiana
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# --- 1. IMPORTACIONES DEL PROYECTO ---
# Se mantiene la importación de las herramientas auxiliares
from config import Config
from preprocess import infer_feature_types, build_preprocessor
from metrics import regression_report

# ==============================================================================
# --- 2. CONFIGURACIÓN DE LA BÚSQUEDA ---
# ==============================================================================

# --- PARÁMETROS DEL SCRIPT ---
CSV_PATH = "data/raw/casas_completas_con_present.csv"
BASE_OUTDIR = "models/bayesian_search_final"
N_CALLS = 10 # Puedes ajustar el número de iteraciones

# --- Umbral para guardar resultados ---
R2_THRESHOLD = 0.933

# --- ESPACIO DE BÚSQUEDA DE HIPERPARÁMETROS ---
SEARCH_SPACE = [
    Integer(600, 3000, name='n_estimators'),
    Real(0.01, 0.1, "log-uniform", name='learning_rate'),
    Integer(1, 6, name='max_depth'),
    Integer(5, 20, name='min_child_weight'),
    Real(0.0, 5.0, name='gamma'),
    Real(0.1, 1.0, name='subsample'),
    Real(0.1, 1.0, name='colsample_bytree'),
    Real(0.0, 5.0, name='reg_lambda'),
    Real(0.0, 5.0, name='reg_alpha'),
]

# Parámetros técnicos fijos


# ==============================================================================
# --- LÓGICA DEL SCRIPT ---
# ==============================================================================

# Variables globales
X_train, X_test, y_train, y_test = None, None, None, None
global X_train_prep, X_test_prep, y_train_log, y_test_log
preprocessor = None
iteration_counter = 0
high_performing_results = []

# === Variables globales (solo inicializar) ===
X_train = X_test = y_train = y_test = None
X_train_prep = X_test_prep = None
preprocessor = None
iteration_counter = 0
high_performing_results = []

def _hash_index(idx):
        # hash simple y estable basado en los índices; útil para comparar entre scripts
        import hashlib
        s = ",".join(map(str, idx.tolist()))
        return hashlib.md5(s.encode("utf-8")).hexdigest()

def _hash_array(arr):
    s = ",".join([f"{x:.6f}" for x in np.asarray(arr).ravel()])
    return hashlib.md5(s.encode("utf-8")).hexdigest()

@use_named_args(SEARCH_SPACE)
def bayes_objective(**params):
    global iteration_counter, high_performing_results
    iteration_counter += 1

    # --- Normalización de tipos (por seguridad) ---
    if "max_depth" in params:
        params["max_depth"] = int(params["max_depth"])
    if "n_estimators" in params:
        params["n_estimators"] = int(params["n_estimators"])
    for k in ["learning_rate", "subsample", "colsample_bytree", "gamma", "min_child_weight", "reg_alpha", "reg_lambda"]:
        if k in params:
            params[k] = float(params[k])

    current_xgb_params = FIXED_XGB_PARAMS.copy()
    current_xgb_params.update(params)
    current_xgb_params["random_state"] = 42
    current_xgb_params["seed"] = 42
    current_xgb_params["n_jobs"] = -1

    print(f"\n--- Iteración {iteration_counter}/{N_CALLS} ---")
    print("[bayesian] XGB params efectivos:", current_xgb_params)
    print("[bayesian] hash_train_idx:", _hash_index(X_train.index))
    print("[bayesian] hash_test_idx :", _hash_index(X_test.index))
    print("[bayesian] hash_y_train :", _hash_array(y_train))
    print("[bayesian] hash_y_test  :", _hash_array(y_test))

    import xgboost, sklearn
    print("[env] xgboost.__version__:", xgboost.__version__)
    print("[env] sklearn.__version__:", sklearn.__version__)


    xgb = XGBRegressor(**current_xgb_params)
    reg = TransformedTargetRegressor(
    regressor=xgb,
    func=np.log1p,
    inverse_func=np.expm1
)

    # Usamos los ya-preprocesados (no re-ajustes)
    def _sorted_items(d):
        return {k: d[k] for k in sorted(d.keys())}

    print("[debug] XGB params efectivos:", _sorted_items(current_xgb_params))  # bayesian
    # o en train_xgb_log:
    print("[debug] XGB params efectivos:", _sorted_items(xgb_params))


    reg.fit(X_train_prep, y_train)
    yhat_te = reg.predict(X_test_prep)

    rep_te = regression_report(y_test, yhat_te)
    r2 = rep_te["R2"]
    print(f"Resultado -> Test R2: {r2:.6f}")

    if r2 > R2_THRESHOLD:
        params_serializable = {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}
        high_performing_results.append({"iteration": iteration_counter, "r2_score": r2, "params": params_serializable})

    return -r2

def main():
    global X_train, X_test, y_train, y_test, preprocessor, X_train_prep, X_test_prep
    
    cfg = Config()
    
    # --- Pasos de limpieza de tu script, replicados aquí ---
    # 1) Leer csv con autodetección y limpiar BOM
    df = pd.read_csv(CSV_PATH, sep=None, engine="python")
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]

    # 2) Forzar categóricas
    for col in ["MS SubClass", "Mo Sold"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # 3) Asegurar target numérico y eliminar filas inválidas
    df[cfg.target] = pd.to_numeric(df[cfg.target], errors="coerce")
    df = df.dropna(subset=[cfg.target])
    
    # 4) Inferir tipos de variables
    numeric_cols, categorical_cols = infer_feature_types(
        df, target=cfg.target, drop_cols=cfg.drop_cols
    )
    
    # 5) Split train/test
    X = df.drop(columns=[cfg.target] + [c for c in cfg.drop_cols if c in df.columns], errors="ignore")
    y = df[cfg.target].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )
    # --- Chequeo opcional para asegurar el mismo split que en train_xgb_log.py ---

    print("[bayesian] hash_train_idx:", _hash_index(X_train.index))
    print("[bayesian] hash_test_idx :", _hash_index(X_test.index))

    # 5.1) Aplicar log-transform del target (si se usa en train_xgb_log)
    log_target = True  # igual que en train_xgb_log.py (activa si allí usas --log_target)




    # 6) Preprocesador
    # 6) Preprocesador (ajustado una sola vez, igual que en train_xgb_log.py)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    preprocessor.fit(X_train)

    X_train_prep = preprocessor.transform(X_train)
    X_test_prep  = preprocessor.transform(X_test) 

    # --- Fin de la lógica replicada ---

    print("="*50)
    print(f"INICIANDO BÚSQUEDA BAYESIANA (LÓGICA 'train_xgb_log.py' COMPLETA)")
    print(f"Se guardarán los resultados con R2 > {R2_THRESHOLD}")
    print("="*50)
    
    result = gp_minimize(
    func=bayes_objective,       # <- OJO: el nuevo nombre
    dimensions=SEARCH_SPACE,
    n_calls=N_CALLS,
    random_state=42,
    n_initial_points=10,
    acq_func="EI")

    print("\n\n" + "="*50)
    print("--- 🏆 MEJOR COMBINACIÓN ÚNICA ENCONTRADA 🏆 ---")
    print("="*50)
    print(f"Mejor Test R2 = {-result.fun:.6f}")
    best_params = {space.name: value for space, value in zip(SEARCH_SPACE, result.x)}
    for key, value in best_params.items():
        print(f"  - {key}: {value}")
        
    print("\n\n" + "="*50)
    print(f"--- 💾 GUARDANDO LISTA DE FINALISTAS ---")
    print("="*50)
    
    os.makedirs(BASE_OUTDIR, exist_ok=True)
    output_filepath = os.path.join(BASE_OUTDIR, "mejores_hiperparametros.json")
    
    if not high_performing_results:
        print("No se encontró ninguna combinación que superara el umbral.")
    else:
        sorted_results = sorted(high_performing_results, key=lambda x: x['r2_score'], reverse=True)
        with open(output_filepath, "w") as f:
            json.dump(sorted_results, f, indent=2)
        print(f"✅ Se guardaron {len(sorted_results)} combinaciones en: {output_filepath}")

if __name__ == "__main__":
    main()


#hola