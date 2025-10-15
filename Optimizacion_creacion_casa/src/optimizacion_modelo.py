import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import gurobi_ml
from gurobi_ml import add_predictor_constr
import joblib
import csv
from sklearn.impute import SimpleImputer

# Cargar los nombres de las columnas desde el CSV
with open("data/raw/casas_completas_con_present.csv", newline='') as f:
    reader = csv.reader(f)
    csv_feature_names = next(reader)

def construir_modelo_optimizacion(presupuesto, lot_area, neighborhood):
    
    m = gp.Model("Optimizacion_Vivienda")

    # --- 2. Definición automática de Variables de Decisión según el pipeline ---
    xgb_pipeline = joblib.load("Optimizacion_creacion_casa/Modelos/Caso_bayesiano_top/xgb/model_xgb.joblib")

    # Lee el CSV solo para obtener el orden de las columnas
    df = pd.read_csv("data/raw/casas_completas_con_present.csv", nrows=1)
    drop_cols = ["Order", "PID", "SalePrice", "\ufeffOrder", "SalePrice_Present"]
    feature_names = [col for col in df.columns if col not in drop_cols]

    # Crear variables de decisión en Gurobi para cada feature en el orden correcto
    decision_vars = {}
    for name in feature_names:
        decision_vars[name] = m.addVar(vtype=GRB.CONTINUOUS, name=name)  # Ajusta el tipo si lo necesitas

    # Crear el DataFrame de entrada para el modelo, usando el orden exacto
    input_df = pd.DataFrame([[decision_vars[name] for name in feature_names]], columns=feature_names)

    # Conectar el pipeline completo con Gurobi ML
    valor_predicho = m.addVar(vtype=GRB.CONTINUOUS, name="valor_predicho")
    add_predictor_constr(m, xgb_pipeline, input_df, valor_predicho)

    # --- 3. Definición de la Función Objetivo ---
    costo_por_pie_cuadrado = 110 # Ejemplo: $110/sqft
    costo_bano_completo = 5000 # Ejemplo
    costo_cocina = 15000 # Ejemplo
    
    inversion_total = m.addVar(vtype=GRB.CONTINUOUS, name="inversion_total")
    # Ejemplo de inversión total (ajusta según tus variables)
    m.addConstr(inversion_total == decision_vars.get("Gr Liv Area", 0) * costo_por_pie_cuadrado + \
                                    decision_vars.get("Full Bath", 0) * costo_bano_completo + \
                                    decision_vars.get("Kitchen AbvGr", 0) * costo_cocina)

    m.setObjective(valor_predicho - inversion_total, GRB.MAXIMIZE)

    # --- 4. Definición de Restricciones ---
    # Aquí puedes agregar tus restricciones usando decision_vars, por ejemplo:
    # m.addConstr(decision_vars["Kitchen AbvGr"] >= 1, "R11_Minimo_1_cocina")
    # m.addConstr(decision_vars["Bedroom AbvGr"] >= 1, "R12_Minimo_1_dormitorio")
    # m.addConstr(decision_vars["Full Bath"] >= 1, "R13_Minimo_1_bano_completo")
    # ...etc.

    print("Optimizando el modelo...")
    m.optimize()

    if m.status == GRB.OPTIMAL:
        print("\n¡Solución óptima encontrada!")
        print(f"Valor objetivo (Beneficio Neto): ${m.objVal:,.2f}")
        
        solucion = {}
        for v in m.getVars():
            if v.x > 0.9: 
                print(f"{v.varName}: {v.x:,.2f}")
                solucion[v.varName] = v.x
        return solucion
    else:
        print("No se encontró una solución óptima.")
        return None

# --- Ejecución del Modelo ---
if __name__ == "__main__":
    # Parámetros de entrada para la construcción
    PRESUPUESTO_CLIENTE = 500000  # $500,000
    AREA_TERRENO = 10000      # 10,000 sqft
    VECINDARIO = "CollgCr"    # Ejemplo

    resultado_casa = construir_modelo_optimizacion(
        presupuesto=PRESUPUESTO_CLIENTE,
        lot_area=AREA_TERRENO,
        neighborhood=VECINDARIO
    )

    if resultado_casa:
        print("\n--- Resumen de la Casa a Construir ---")
        # Aquí podrías formatear el resultado en un reporte o DataFrame
        # y usar el diccionario `resultado_casa` para hacer una predicción final
        # con tu modelo XGBoost real para validar el valor.