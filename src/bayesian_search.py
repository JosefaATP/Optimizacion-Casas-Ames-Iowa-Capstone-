import pandas as pd
import json
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor # Se usa el de tu script
from xgboost import XGBRegressor

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
N_CALLS = 100 # Puedes ajustar el número de iteraciones

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
FIXED_XGB_PARAMS = {
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "n_jobs": -1,
}

# ==============================================================================
# --- LÓGICA DEL SCRIPT ---
# ==============================================================================

# Variables globales
X_train, X_test, y_train, y_test = None, None, None, None
preprocessor = None
iteration_counter = 0
high_performing_results = []

@use_named_args(SEARCH_SPACE)
def objective(**params):
    global iteration_counter, high_performing_results
    iteration_counter += 1
    
    current_xgb_params = FIXED_XGB_PARAMS.copy()
    current_xgb_params.update(params)
    current_xgb_params['random_state'] = 42
    
    # --- Lógica de tu script replicada fielmente ---
    xgb = XGBRegressor(**current_xgb_params)
    
    # Se usa TransformedTargetRegressor para manejar la transformación logarítmica
    reg = TransformedTargetRegressor(regressor=xgb, func=np.log1p, inverse_func=np.expm1)
    
    pipe = Pipeline(steps=[("pre", preprocessor), ("xgb", reg)])
    
    print(f"\n--- Iteración {iteration_counter}/{N_CALLS} ---")
    print(f"Probando: {params}")
    
    pipe.fit(X_train, y_train)
    
    yhat_te = pipe.predict(X_test)
    rep_te = regression_report(y_test, yhat_te)
    r2_score = rep_te['R2']
    
    print(f"Resultado -> Test R2: {r2_score:.6f}")
    
    if r2_score > R2_THRESHOLD:
        print(f"🎉 ¡Buen resultado! Guardando en la lista... (R2 > {R2_THRESHOLD})")
        params_serializable = {k: v.item() if hasattr(v, 'item') else v for k, v in params.items()}
        high_performing_results.append({
            "iteration": iteration_counter,
            "r2_score": r2_score,
            "params": params_serializable
        })
    
    return -r2_score

def main():
    global X_train, X_test, y_train, y_test, preprocessor
    
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

    # 6) Preprocesador
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # --- Fin de la lógica replicada ---

    print("="*50)
    print(f"INICIANDO BÚSQUEDA BAYESIANA (LÓGICA 'train_xgb_log.py' COMPLETA)")
    print(f"Se guardarán los resultados con R2 > {R2_THRESHOLD}")
    print("="*50)
    
    result = gp_minimize(
        func=objective,
        dimensions=SEARCH_SPACE,
        n_calls=N_CALLS,
        random_state=cfg.random_state,
        n_initial_points=10
    )

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