"""
2_modelo_remodelacion.py
========================
Modelo de optimización de REMODELACIÓN con TODAS las restricciones del PDF.

Implementa:
- Sección 3: Función objetivo Max(ΔP - C_total)
- Sección 5: Restricciones básicas (1-5)
- Sección 6: Restricciones de renovación (Utilities, Roof, Exterior, MasVnr)

Grupo 5 - Capstone 2025
"""

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from datetime import datetime
import os
from utils import guardar_resultado

# Importar módulos propios
from config import (
    MODEL_PATH,
    DATA_PATH,
    ATRIBUTOS_ACCIONABLES,
    ATRIBUTOS_FIJOS,
    GUROBI_PARAMS
    #COSTO_DEMOLICION_POR_SQFT
)
from utils import (
    cargar_modelo_xgboost,
    calcular_gradientes_xgboost,
    validar_restricciones_basicas,
    formatear_resultados_remodelacion,
    guardar_resultados
)
from restricciones_renovacion import agregar_todas_restricciones_renovacion

# ============================================================================
# FUNCIÓN PRINCIPAL: OPTIMIZAR REMODELACIÓN
# ============================================================================

def optimizar_remodelacion(casa_actual, presupuesto, modelo_xgb, verbose=True):
    """
    Optimiza qué remodelaciones hacer a una casa existente.
    
    Implementa la función objetivo del PDF Sección 3:
        Max(ΔP - C_total)
    
    Donde:
        ΔP = ValorFinal - ValorInicial
        C_total = Costos construcción + Costos demolición + Costos categóricos
    
    Args:
        casa_actual (pd.Series): Características actuales de la casa
        presupuesto (float): Presupuesto disponible para remodelar
        modelo_xgb: Modelo XGBoost entrenado
        verbose (bool): Mostrar información detallada
        
    Returns:
        dict: Resultados de la optimización o None si no hay solución
    """
    
    if verbose:
        print("\n" + "="*70)
        print("MODELO DE OPTIMIZACIÓN DE REMODELACIÓN")
        print("="*70)
        print(f"\n💰 Presupuesto disponible: ${presupuesto:,.0f}")
    
    # ========================================================================
    # PASO 1: CREAR MODELO DE GUROBI
    # ========================================================================
    
    modelo = gp.Model("Remodelacion_Optima")
    
    # Configurar parámetros de Gurobi
    for param, valor in GUROBI_PARAMS.items():
        modelo.setParam(param, valor)
    
    if verbose:
        modelo.Params.LogToConsole = 1  # Mostrar log
        print("\n⚙️  Modelo de Gurobi creado")
    
    # ========================================================================
    # PASO 2: CREAR VARIABLES DE DECISIÓN (NUMÉRICAS)
    # ========================================================================
    
    if verbose:
        print("\n📊 Creando variables de decisión numéricas...")
    
    x_nuevo = {}  # Variables para valores nuevos
    delta = {}    # Variables para cambios
    delta_abs = {}  # Variables para valor absoluto de cambios
    
    for attr in casa_actual.index:
        if attr in ATRIBUTOS_ACCIONABLES:
            costo, es_discreta, min_val, max_val, desc = ATRIBUTOS_ACCIONABLES[attr]
            
            # Variable para valor NUEVO
            if es_discreta:
                x_nuevo[attr] = modelo.addVar(
                    lb=min_val, ub=max_val, 
                    vtype=GRB.INTEGER, 
                    name=f"x_new_{attr}"
                )
            else:
                x_nuevo[attr] = modelo.addVar(
                    lb=min_val, ub=max_val, 
                    vtype=GRB.CONTINUOUS, 
                    name=f"x_new_{attr}"
                )
            
            # Variable para CAMBIO (delta = nuevo - actual)
            delta[attr] = modelo.addVar(
                lb=-GRB.INFINITY, 
                name=f"delta_{attr}"
            )
            
            # Variable para |CAMBIO|
            delta_abs[attr] = modelo.addVar(
                lb=0, 
                name=f"delta_abs_{attr}"
            )
            
            # Restricción: delta = nuevo - actual
            modelo.addConstr(
                delta[attr] == x_nuevo[attr] - casa_actual[attr],
                name=f"def_delta_{attr}"
            )
            
            # Restricciones para valor absoluto
            modelo.addConstr(delta_abs[attr] >= delta[attr], name=f"abs1_{attr}")
            modelo.addConstr(delta_abs[attr] >= -delta[attr], name=f"abs2_{attr}")
            
        elif attr in ATRIBUTOS_FIJOS:
            # Atributos fijos: mantener valor actual
            x_nuevo[attr] = casa_actual[attr]
    
    if verbose:
        print(f"  ✓ {len([k for k in x_nuevo if k in ATRIBUTOS_ACCIONABLES])} variables numéricas creadas")
    
    # ========================================================================
    # PASO 3: CALCULAR PRECIO ACTUAL
    # ========================================================================
    
    if verbose:
        print("\n💵 Calculando precio actual con XGBoost...")
    
    X_actual = pd.DataFrame([casa_actual]).reindex(columns=casa_actual.index)
    precio_actual = modelo_xgb.predict(X_actual)[0]
    
    if verbose:
        print(f"  Precio actual: ${precio_actual:,.0f}")
    
    # ========================================================================
    # PASO 4: APROXIMAR PRECIO NUEVO (LINEARIZACIÓN)
    # ========================================================================
    
    if verbose:
        print("\n🔢 Calculando gradientes para linearización...")
    
    gradientes, _ = calcular_gradientes_xgboost(
        casa_actual, 
        modelo_xgb, 
        atributos_accionables=list(ATRIBUTOS_ACCIONABLES.keys())
    )
    
    # Variable para precio nuevo (aproximado)
    precio_nuevo_aprox = modelo.addVar(lb=0, name="precio_nuevo_aprox")

    modelo.addConstr(
        precio_nuevo_aprox == precio_actual + gp.quicksum(
        gradientes[attr] * delta[attr]
        for attr in delta.keys() if attr in gradientes
        ),
        name="precio_nuevo_def"
    )
    
    if verbose:
        print(f"  ✓ Gradientes calculados para {len(gradientes)} atributos")
    
    # ========================================================================
    # PASO 5: CALCULAR COSTO TOTAL DE REMODELACIÓN
    # ========================================================================
    
    if verbose:
        print("\n💸 Definiendo costos de remodelación...")
    
    # Costo de construcción/modificación de atributos numéricos
    costo_construccion_numerico = gp.quicksum(
        delta_abs[attr] * ATRIBUTOS_ACCIONABLES[attr][0]
        for attr in ATRIBUTOS_ACCIONABLES.keys()
        if attr in delta_abs
    )
    
    # Costo de demolición (cuando reduces área)
    # costo_demolicion = gp.quicksum(
    #     -delta[attr] * COSTO_DEMOLICION_POR_SQFT
    #     for attr in ['GrLivArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
    #     if attr in delta
    # )
    
    # Variable para costo total numérico
    costo_numerico = modelo.addVar(lb=0, name="costo_numerico")
    modelo.addConstr(
        costo_numerico == costo_construccion_numerico,  # + costo_demolicion,
        name="costo_numerico_def"
    )
    
    # if verbose:
    #     print(f"  ✓ Costo construcción numérico definido")
    #     print(f"  ✓ Costo demolición: ${COSTO_DEMOLICION_POR_SQFT}/sqft")
    
    # ========================================================================
    # PASO 6: AGREGAR RESTRICCIONES DE RENOVACIÓN (SECCIÓN 6 DEL PDF)
    # ========================================================================
    
    if verbose:
        print("\n🔧 Agregando restricciones de renovación categóricas...")
    
    try:
        restricciones_categoricas = agregar_todas_restricciones_renovacion(
            modelo, 
            casa_actual, 
            i=0
        )
        costo_categorico = restricciones_categoricas['costo_total_categorico']
        
        if verbose:
            print("  ✓ Restricciones de Utilities, Roof, Exterior, MasVnr agregadas")
    except Exception as e:
        if verbose:
            print(f"  ⚠️  Error al agregar restricciones categóricas: {e}")
            print("  ⚠️  Continuando sin restricciones categóricas...")
        costo_categorico = 0
    
    # ========================================================================
    # PASO 7: COSTO TOTAL Y RESTRICCIÓN DE PRESUPUESTO
    # ========================================================================
    
    costo_total = modelo.addVar(lb=0, name="costo_total")
    modelo.addConstr(
        costo_total == costo_numerico + costo_categorico,
        name="costo_total_def"
    )
    
    # RESTRICCIÓN: Presupuesto (del PDF)
    modelo.addConstr(
        costo_total <= presupuesto,
        name="restriccion_presupuesto"
    )
    
    if verbose:
        print(f"\n💰 Restricción de presupuesto: ≤ ${presupuesto:,.0f}")
    
    # ========================================================================
    # PASO 8: RESTRICCIONES TÉCNICAS BÁSICAS (SECCIÓN 5 DEL PDF)
    # ========================================================================
    
    if verbose:
        print("\n🏗️  Agregando restricciones técnicas básicas...")
    
    # Restricción 1: 2do piso ≤ 1er piso
    if '1stFlrSF' in x_nuevo and '2ndFlrSF' in x_nuevo:
        modelo.addConstr(
            x_nuevo['2ndFlrSF'] <= x_nuevo['1stFlrSF'],
            name="r1_2ndo_piso_max"
        )
        if verbose:
            print("  ✓ R1: 2ndFlrSF ≤ 1stFlrSF")
    
    # Restricción 2: GrLivArea ≤ LotArea
    if 'GrLivArea' in x_nuevo and 'LotArea' in casa_actual.index:
        modelo.addConstr(
            x_nuevo['GrLivArea'] <= casa_actual['LotArea'],
            name="r2_area_max_terreno"
        )
        if verbose:
            print("  ✓ R2: GrLivArea ≤ LotArea")
    
    # Restricción 3: Sótano ≤ 1er piso
    if 'TotalBsmtSF' in x_nuevo and '1stFlrSF' in x_nuevo:
        modelo.addConstr(
            x_nuevo['TotalBsmtSF'] <= x_nuevo['1stFlrSF'],
            name="r3_sotano_max"
        )
        if verbose:
            print("  ✓ R3: TotalBsmtSF ≤ 1stFlrSF")
    
    # Restricción 4: Baños ≤ Dormitorios + 2
    if 'FullBath' in x_nuevo and 'BedroomAbvGr' in x_nuevo:
        total_banos = x_nuevo['FullBath']
        if 'HalfBath' in x_nuevo:
            total_banos = total_banos + 0.5 * x_nuevo['HalfBath']
        
        modelo.addConstr(
            total_banos <= x_nuevo['BedroomAbvGr'] + 2,
            name="r4_banos_vs_dormitorios"
        )
        if verbose:
            print("  ✓ R4: FullBath + 0.5×HalfBath ≤ Bedrooms + 2")
    
    # Restricción 5: Mínimos básicos
    if 'FullBath' in x_nuevo:
        modelo.addConstr(x_nuevo['FullBath'] >= 1, name="r5_min_1_bano")
    if 'BedroomAbvGr' in x_nuevo:
        modelo.addConstr(x_nuevo['BedroomAbvGr'] >= 1, name="r5_min_1_dormitorio")
    if 'KitchenAbvGr' in x_nuevo:
        modelo.addConstr(x_nuevo['KitchenAbvGr'] >= 1, name="r5_min_1_cocina")
    
    if verbose:
        print("  ✓ R5: Mínimos (1 baño, 1 dormitorio, 1 cocina)")
    
    # Restricción 7: Consistencia GrLivArea
    if all(k in x_nuevo for k in ['GrLivArea', '1stFlrSF', '2ndFlrSF']):
        low_qual = x_nuevo.get('LowQualFinSF', 0)
        modelo.addConstr(
            x_nuevo['GrLivArea'] == x_nuevo['1stFlrSF'] + x_nuevo['2ndFlrSF'] + low_qual,
            name="r7_grlivarea_consistencia"
        )
        if verbose:
            print("  ✓ R7: GrLivArea = 1stFlr + 2ndFlr + LowQual")
    
    # ========================================================================
    # PASO 9: DEFINIR GANANCIA NETA Y FUNCIÓN OBJETIVO
    # ========================================================================
    
    if verbose:
        print("\n🎯 Definiendo función objetivo...")
    
    # ΔP = ValorFinal - ValorInicial
    aumento_valor = modelo.addVar(lb=-GRB.INFINITY, name="aumento_valor")
    modelo.addConstr(
        aumento_valor == precio_nuevo_aprox - precio_actual,
        name="aumento_valor_def"
    )
    
    # Ganancia Neta = ΔP - C_total
    ganancia_neta = modelo.addVar(lb=-GRB.INFINITY, name="ganancia_neta")
    modelo.addConstr(
        ganancia_neta == aumento_valor - costo_total,
        name="ganancia_neta_def"
    )
    
    # RESTRICCIÓN CRÍTICA: La remodelación DEBE ser rentable
    modelo.addConstr(
        ganancia_neta >= 0,
        name="rentabilidad_minima"
    )
    
    # FUNCIÓN OBJETIVO (Sección 3 del PDF): Max(ΔP - C_total)
    modelo.setObjective(ganancia_neta, GRB.MAXIMIZE)
    
    if verbose:
        print("  ✓ Objetivo: Max(Ganancia Neta) = Max(ΔP - C_total)")
        print("  ✓ Restricción: Ganancia Neta ≥ 0 (rentabilidad mínima)")
    
    # ========================================================================
    # PASO 10: RESOLVER
    # ========================================================================
    
    if verbose:
        print("\n" + "="*70)
        print("⚙️  RESOLVIENDO MODELO...")
        print("="*70)
    
    inicio = datetime.now()
    modelo.optimize()
    tiempo_resolucion = (datetime.now() - inicio).total_seconds()
    
    if verbose:
        print(f"\n⏱️  Tiempo de resolución: {tiempo_resolucion:.2f} segundos")
    
    # ========================================================================
    # PASO 11: PROCESAR RESULTADOS
    # ========================================================================
    
    if modelo.status == GRB.OPTIMAL or (modelo.status == GRB.TIME_LIMIT and modelo.SolCount > 0):
        if verbose:
            print("\n✅ SOLUCIÓN ENCONTRADA")
        
        # Validar solución con XGBoost real
        X_optimo = pd.DataFrame([{
            attr: x_nuevo[attr].X if attr in x_nuevo and hasattr(x_nuevo[attr], 'X') else casa_actual[attr]
            for attr in casa_actual.index
        }])
        
        precio_real_xgb = modelo_xgb.predict(X_optimo)[0]
        aumento_valor_real = precio_real_xgb - precio_actual
        ganancia_neta_real = aumento_valor_real - costo_total.X
        
        # Construir resultado
        resultado = {
            'status': 'optimal',
            'precio_actual': precio_actual,
            'precio_nuevo_aprox': precio_nuevo_aprox.X,
            'precio_nuevo_real': precio_real_xgb,
            'aumento_valor': aumento_valor_real,
            'costo_inversion': costo_total.X,
            'ganancia_neta': ganancia_neta_real,
            'roi_porcentaje': (ganancia_neta_real / costo_total.X) * 100 if costo_total.X > 0 else 0,
            'tiempo_resolucion': tiempo_resolucion,
            'modificaciones': {}
        }
        
        # Extraer modificaciones significativas
        for attr in ATRIBUTOS_ACCIONABLES.keys():
            if attr in delta and hasattr(delta[attr], 'X'):
                if abs(delta[attr].X) > 0.01:
                    resultado['modificaciones'][attr] = {
                        'actual': casa_actual[attr],
                        'nuevo': x_nuevo[attr].X,
                        'cambio': delta[attr].X,
                        'costo': delta_abs[attr].X * ATRIBUTOS_ACCIONABLES[attr][0]
                    }
        
        return resultado
        
    elif modelo.status == GRB.INFEASIBLE:
        if verbose:
            print("\n❌ MODELO INFACTIBLE")
            print("   No existe solución que cumpla todas las restricciones")
            print("   Posibles causas:")
            print("   - Presupuesto insuficiente para remodelación rentable")
            print("   - Restricciones técnicas incompatibles")
        return None
        
    else:
        if verbose:
            print(f"\n❌ No se encontró solución óptima.")
            if modelo.status == GRB.TIME_LIMIT and modelo.SolCount > 0:
                print("⏰ Se acabó el tiempo, pero tengo una buena solución")
                # Procesar la mejor solución encontrada hasta ahora
            elif modelo.status == GRB.INF_OR_UNBD:
                print("❌ Infactible o no acotado")
            elif modelo.status == GRB.INFEASIBLE:
                print("❌ No hay NINGUNA remodelación que cumpla todas las reglas (infactible)")
             # Revisar presupuesto y restricciones
            elif modelo.status == GRB.UNBOUNDED:
                print("❌ No acotado")
            else:  # Todos los otros casos
                print(f"⚠️  Estado inesperado: {modelo.status}")

# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("MODELO DE REMODELACIÓN - GRUPO 5 CAPSTONE")
    print("="*70)
    
    # Cargar modelo XGBoost (si no existe, entrenar y guardar)
    print("\n📦 Cargando modelo XGBoost...")
    try:
        modelo_xgb = cargar_modelo_xgboost(MODEL_PATH)
    except Exception as e:
        print("⚠️  Modelo no encontrado o error al cargar:", e)
        print("🔁 Entrenando un nuevo modelo XGBoost y guardándolo en disco...")
        try:
            from utils import entrenar_y_guardar_modelo
            modelo_xgb = entrenar_y_guardar_modelo(DATA_PATH, MODEL_PATH)
            print("  ✓ Modelo entrenado y guardado en:", MODEL_PATH)
        except Exception as e2:
            print("❌ Error al entrenar el modelo:", e2)
            print("   Ejecuta primero 1_entrenamiento_xgboost.py o revisa los datos.")
            exit(1)
    
    # Cargar datos
    print("\n📊 Cargando dataset...")
    df = pd.read_csv(DATA_PATH)
    #print(df.columns)
    base = df.iloc[0].copy()

    # Si existe SalePrice, eliminarlo de la serie de características
    if 'SalePrice_Present' in base.index:
        base = base.drop(labels=['SalePrice_Present'])
    if 'SalePrice' in base.index:
        base = base.drop(labels=['SalePrice'])

    # Modificar valores para "inventar" la casa actual (ajusta según prefieras)
    casa_actual = {
    'Gr Liv Area': 1656,
    '1st Flr SF': 1200,
    '2nd Flr SF': 600,
    'Total Bsmt SF': 800,
    'Full Bath': 2,
    'Half Bath': 1,
    'Bedroom AbvGr': 2,
    'Kitchen AbvGr': 1,
    'Overall Qual': 5,
    'Lot Area': 9000,
    }

    # Completar atributos faltantes
    for attr in ATRIBUTOS_ACCIONABLES.keys():
        if attr not in casa_actual:
            casa_actual[attr] = ATRIBUTOS_ACCIONABLES[attr][2]  # Si este atributo no está definido en la casa base, asígnale su valor mínimo permitido según el diccionario de atributos accionables.

    for attr in base.index:
        if attr not in casa_actual:
            casa_actual[attr] = base[attr]

    # Presupuesto disponible para remodelación
    presupuesto = 5000000

    casa_actual = pd.Series(casa_actual)
    
    # Ejecutar optimización
    resultado = optimizar_remodelacion(
        casa_actual=casa_actual,
        presupuesto=presupuesto,
        modelo_xgb=modelo_xgb,
        verbose=True
    )
    
    # Mostrar resultados
    if resultado:
        print("\n" + formatear_resultados_remodelacion(resultado))
        
        # Guardar resultados
        guardar_resultado(resultado, carpeta='results', prefijo='resultado_casa')

    else:
        print("\n❌ No se pudo generar una solución de remodelación rentable")
    
    print("\n" + "="*70)
    print("FIN DEL PROCESO")
    print("="*70)