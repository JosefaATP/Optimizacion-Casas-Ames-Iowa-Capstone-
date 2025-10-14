import pandas as pd
import joblib
import gurobipy as gp
from gurobipy import GRB
import argparse # ¡Importante! Se añade la librería para argumentos de terminal

# ==============================================================================
# --- 1. CONFIGURACIÓN DEL ANÁLISIS ---
# ==============================================================================

# --- Parámetros del escenario de remodelación ---
# (Estos valores se mantienen dentro del script para este ejemplo básico)
ID_CASA_A_EVALUAR = 1460 

REMODELACION = {
    "nombre": "Añadir 1 baño completo",
    "costo": 15000,
    "cambio_en_features": {
        "FullBath": 1
    }
}

# ==============================================================================
# --- LÓGICA DE OPTIMIZACIÓN (Gurobi + XGBoost) ---
# ==============================================================================

def main():
    # --- Configuración de los argumentos de la terminal ---
    parser = argparse.ArgumentParser(description="Corre un análisis de optimización de remodelación para una casa.")
    parser.add_argument("--csv", required=True, help="Ruta al archivo CSV con los datos de las casas. ej: data/raw/casas_completas_con_present.csv")
    parser.add_argument("--model-path", required=True, help="Ruta al modelo de XGBoost entrenado (.joblib). ej: models/xgb/completa_present/model_xgb.joblib")
    args = parser.parse_args()

    try:
        # --- 2. Cargar Modelo y Preparar Datos usando los argumentos ---
        print(f"Cargando modelo predictivo desde: {args.model_path}")
        pipe_xgb = joblib.load(args.model_path)

        print(f"Cargando datos de las casas desde: {args.csv}")
        df_casas = pd.read_csv(args.csv)
        
        casa_actual = df_casas.iloc[[ID_CASA_A_EVALUAR]].copy()
        
        if casa_actual.empty:
            raise ValueError(f"No se encontró la casa con ID {ID_CASA_A_EVALUAR}")

        valor_actual = casa_actual['SalePrice_Present'].iloc[0]

        # --- 3. Simular la Remodelación y Predecir el Valor Futuro ---
        casa_remodelada = casa_actual.copy()
        for feature, cambio in REMODELACION["cambio_en_features"].items():
            print(f"Aplicando remodelación: '{feature}' aumenta en {cambio}")
            casa_remodelada[feature] += cambio

        valor_futuro_predicho = pipe_xgb.predict(casa_remodelada)[0]
        ganancia_bruta = valor_futuro_predicho - valor_actual
        costo_remodelacion = REMODELACION["costo"]

        print("\n--- Análisis Predictivo ---")
        print(f"Valor Actual de la Casa {ID_CASA_A_EVALUAR}: ${valor_actual:,.2f}")
        print(f"Valor Futuro Predicho (con remodelación): ${valor_futuro_predicho:,.2f}")
        print(f"Aumento de valor estimado (Plusvalía): ${ganancia_bruta:,.2f}")
        print(f"Costo de la Remodelación: ${costo_remodelacion:,.2f}")

        # --- 4. Modelo de Optimización con Gurobi ---
        m = gp.Model("DecisionRemodelacion")
        hacer_remodelacion = m.addVar(vtype=GRB.BINARY, name="hacer_remodelacion")
        ganancia_neta = ganancia_bruta - costo_remodelacion
        m.setObjective(ganancia_neta * hacer_remodelacion, GRB.MAXIMIZE)
        
        print("\n--- Optimización con Gurobi ---")
        print("Tomando la decisión óptima...")
        m.optimize()

        # --- 5. Resultados de la Optimización ---
        print("\n--- ✅ Decisión Óptima ---")
        if hacer_remodelacion.X > 0.5:
            print(f"SÍ, se recomienda realizar la remodelación: '{REMODELACION['nombre']}'.")
            print(f"La ganancia neta esperada es de: ${m.objVal:,.2f}")
        else:
            print(f"NO, no se recomienda realizar la remodelación.")
            print("El costo de la remodelación es mayor que la plusvalía generada.")

    except FileNotFoundError as e:
        print(f"Error: No se encontró un archivo. Revisa las rutas. Detalles: {e}")
    except Exception as e:
        print(f"Ha ocurrido un error inesperado: {e}")

if __name__ == "__main__":
    main()