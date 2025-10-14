"""
utils.py
========
Funciones auxiliares compartidas entre los modelos de optimización.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
import joblib
import os

# ============================================================================
# FUNCIONES DE CARGA Y GUARDADO
# ============================================================================

def cargar_modelo_xgboost(path):
    """Carga modelo XGBoost desde disco (joblib)."""
    try:
        modelo = joblib.load(path)
        return modelo
    except Exception as e:
        raise RuntimeError(f"No se pudo cargar el modelo XGBoost desde {path}: {e}")


def cargar_datos(path, columnas=None):
    """
    Carga el dataset de viviendas.
    
    Args:
        path (str): Ruta al CSV
        columnas (list, optional): Lista de columnas a cargar
        
    Returns:
        pd.DataFrame: Dataset cargado
    """
    if columnas:
        df = pd.read_csv(path, usecols=columnas)
    else:
        df = pd.read_csv(path)
    
    print(f"✅ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def guardar_resultados(resultado, filename, carpeta='../results/'):
    """
    Guarda los resultados de una optimización en un archivo de texto.
    
    Args:
        resultado (dict): Diccionario con resultados
        filename (str): Nombre del archivo (sin extensión)
        carpeta (str): Carpeta donde guardar
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"{carpeta}{filename}_{timestamp}.txt"
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(f"RESULTADOS DE OPTIMIZACIÓN\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        for key, value in resultado.items():
            f.write(f"{key}:\n{value}\n\n")
    
    print(f"✅ Resultados guardados en: {filepath}")
    return filepath


# ============================================================================
# FUNCIONES DE VALIDACIÓN
# ============================================================================

def validar_restricciones_basicas(caracteristicas):
    """
    Valida que las características cumplan restricciones básicas de coherencia.
    
    Args:
        caracteristicas (dict): Diccionario con características de la casa
        
    Returns:
        tuple: (es_valido, lista_errores)
    """
    errores = []
    
    # 1. Segundo piso no puede ser más grande que el primero
    if '2ndFlrSF' in caracteristicas and '1stFlrSF' in caracteristicas:
        if caracteristicas['2ndFlrSF'] > caracteristicas['1stFlrSF']:
            errores.append("❌ 2ndFlrSF > 1stFlrSF (segundo piso más grande que primero)")
    
    # 2. Sótano no puede ser más grande que primer piso
    if 'TotalBsmtSF' in caracteristicas and '1stFlrSF' in caracteristicas:
        if caracteristicas['TotalBsmtSF'] > caracteristicas['1stFlrSF']:
            errores.append("❌ TotalBsmtSF > 1stFlrSF (sótano más grande que primer piso)")
    
    # 3. Área habitable debe ser coherente con pisos
    if all(k in caracteristicas for k in ['GrLivArea', '1stFlrSF', '2ndFlrSF']):
        suma_pisos = caracteristicas['1stFlrSF'] + caracteristicas['2ndFlrSF']
        if abs(caracteristicas['GrLivArea'] - suma_pisos) > 10:  # Tolerancia de 10 sqft
            errores.append(f"❌ GrLivArea ({caracteristicas['GrLivArea']}) != 1stFlr + 2ndFlr ({suma_pisos})")
    
    # 4. No más baños que dormitorios
    if 'FullBath' in caracteristicas and 'BedroomAbvGr' in caracteristicas:
        total_banos = caracteristicas.get('FullBath', 0) + caracteristicas.get('HalfBath', 0) * 0.5
        if total_banos > caracteristicas['BedroomAbvGr'] + 2:  # +2 de tolerancia
            errores.append(f"❌ Demasiados baños ({total_banos}) para {caracteristicas['BedroomAbvGr']} dormitorios")
    
    # 5. Al menos 1 baño, 1 dormitorio, 1 cocina
    if caracteristicas.get('FullBath', 0) < 1:
        errores.append("❌ Debe haber al menos 1 baño completo")
    
    if caracteristicas.get('BedroomAbvGr', 0) < 1:
        errores.append("❌ Debe haber al menos 1 dormitorio")
    
    if caracteristicas.get('KitchenAbvGr', 0) < 1:
        errores.append("❌ Debe haber al menos 1 cocina")
    
    # 6. Área mínima razonable
    if caracteristicas.get('GrLivArea', 0) < 600:
        errores.append("❌ GrLivArea < 600 sqft (área muy pequeña)")
    
    es_valido = len(errores) == 0
    return es_valido, errores


# ============================================================================
# CÁLCULO DE GRADIENTES NUMÉRICOS
# ============================================================================

def calcular_gradientes_xgboost(casa_referencia, modelo_xgb, epsilon=1e-4, atributos_accionables=None):
    """
    Calcula gradientes numéricos de XGBoost para linearización local.
    
    Args:
        casa_referencia (pd.Series or dict): Casa de referencia
        modelo_xgb: Modelo XGBoost entrenado
        epsilon (float): Tamaño de perturbación
        atributos_accionables (list): Lista de atributos a calcular gradientes
        
    Returns:
        dict: Diccionario {atributo: gradiente}
    """
    if isinstance(casa_referencia, dict):
        casa_referencia = pd.Series(casa_referencia)
    
    # Crear DataFrame con estructura correcta
    X_ref = pd.DataFrame([casa_referencia])
    precio_ref = modelo_xgb.predict(X_ref)[0]
    
    gradientes = {}
    
    # Si no se especifican, calcular para todos los atributos numéricos
    if atributos_accionables is None:
        atributos_accionables = [col for col in X_ref.columns if X_ref[col].dtype in ['int64', 'float64']]
    
    for attr in atributos_accionables:
        if attr in X_ref.columns:
            X_perturb = X_ref.copy()
            X_perturb[attr] = X_ref[attr] + epsilon
            precio_perturb = modelo_xgb.predict(X_perturb)[0]
            gradientes[attr] = (precio_perturb - precio_ref) / epsilon
        else:
            gradientes[attr] = 0
    
    return gradientes, precio_ref


# ============================================================================
# FORMATEO DE RESULTADOS
# ============================================================================

def formatear_resultados_remodelacion(resultado):
    """
    Formatea los resultados de una optimización de remodelación para impresión bonita.
    
    Args:
        resultado (dict): Diccionario con resultados
        
    Returns:
        str: Texto formateado
    """
    texto = []
    texto.append("="*70)
    texto.append("RESULTADOS DE OPTIMIZACIÓN DE REMODELACIÓN")
    texto.append("="*70)
    texto.append("")
    
    # Análisis económico
    texto.append("📊 ANÁLISIS ECONÓMICO:")
    texto.append(f"  {'Precio ANTES (casa actual):':<45} ${resultado['precio_actual']:>15,.0f}")
    texto.append(f"  {'Precio DESPUÉS (remodelada):':<45} ${resultado['precio_nuevo_real']:>15,.0f}")
    texto.append(f"  {'-'*70}")
    texto.append(f"  {'Aumento de valor:':<45} ${resultado['aumento_valor']:>15,.0f}")
    texto.append(f"  {'Costo de inversión:':<45} ${resultado['costo_inversion']:>15,.0f}")
    texto.append(f"  {'-'*70}")
    texto.append(f"  {'GANANCIA NETA:':<45} ${resultado['ganancia_neta']:>15,.0f}")
    texto.append(f"  {'ROI:':<45} {resultado['roi_porcentaje']:>15.2f}%")
    texto.append("")
    
    if resultado['ganancia_neta'] > 0:
        texto.append("  ✅ LA REMODELACIÓN ES RENTABLE")
        texto.append(f"     Por cada $1 invertido, recuperas ${1 + resultado['ganancia_neta']/resultado['costo_inversion']:.2f}")
    else:
        texto.append("  ❌ LA REMODELACIÓN NO ES RENTABLE")
    
    texto.append("")
    
    # Modificaciones
    if resultado.get('modificaciones'):
        texto.append("💡 MODIFICACIONES RECOMENDADAS:")
        texto.append(f"  {'Característica':<25} {'Actual':<12} {'→':<3} {'Nuevo':<12} {'Cambio':<12} {'Costo':<15}")
        texto.append(f"  {'-'*85}")
        
        for attr, cambio in resultado['modificaciones'].items():
            texto.append(f"  {attr:<25} {cambio['actual']:>12.1f} {'→':<3} {cambio['nuevo']:>12.1f} "
                        f"{cambio['cambio']:>+12.1f} ${cambio['costo']:>13,.0f}")
        
        texto.append(f"  {'-'*85}")
        texto.append(f"  {'TOTAL INVERSIÓN:':<65} ${resultado['costo_inversion']:>15,.0f}")
    
    return "\n".join(texto)


def formatear_resultados_construccion(resultado):
    """
    Formatea los resultados de una optimización de construcción desde cero.
    
    Args:
        resultado (dict): Diccionario con resultados
        
    Returns:
        str: Texto formateado
    """
    texto = []
    texto.append("="*70)
    texto.append("RESULTADOS DE OPTIMIZACIÓN - CONSTRUCCIÓN DESDE CERO")
    texto.append("="*70)
    texto.append("")
    
    # Análisis económico
    texto.append("📊 ANÁLISIS ECONÓMICO:")
    texto.append(f"  {'Costo de construcción:':<45} ${resultado['costo_construccion']:>15,.0f}")
    texto.append(f"  {'Precio de venta estimado:':<45} ${resultado['precio_venta_real']:>15,.0f}")
    texto.append(f"  {'-'*70}")
    texto.append(f"  {'Rentabilidad (Precio/Costo):':<45} {resultado['rentabilidad']:>15.2f}x")
    texto.append(f"  {'ROI:':<45} {resultado['roi_porcentaje']:>15.2f}%")
    texto.append("")
    
    # Características
    if resultado.get('caracteristicas'):
        texto.append("🏠 CARACTERÍSTICAS DE LA CASA ÓPTIMA:")
        texto.append(f"  {'Característica':<30} {'Valor':<15} {'Costo Unitario':<15} {'Costo Total':<15}")
        texto.append(f"  {'-'*80}")
        
        for attr, info in resultado['caracteristicas'].items():
            texto.append(f"  {attr:<30} {info['valor']:>15.1f} ${info['costo_unitario']:>13,.0f} ${info['costo_total']:>13,.0f}")
        
        texto.append(f"  {'-'*80}")
        texto.append(f"  {'TOTAL CONSTRUCCIÓN:':<62} ${resultado['costo_construccion']:>15,.0f}")
    
    return "\n".join(texto)


# ============================================================================
# FUNCIÓN DE ANÁLISIS DE SENSIBILIDAD
# ============================================================================

def analisis_sensibilidad_presupuesto(funcion_optimizacion, presupuestos, **kwargs):
    """
    Realiza análisis de sensibilidad variando el presupuesto.
    
    Args:
        funcion_optimizacion: Función de optimización a ejecutar
        presupuestos (list): Lista de presupuestos a probar
        **kwargs: Argumentos adicionales para la función
        
    Returns:
        pd.DataFrame: Resultados del análisis
    """
    resultados = []
    
    for presupuesto in presupuestos:
        print(f"\n🔍 Probando presupuesto: ${presupuesto:,.0f}")
        resultado = funcion_optimizacion(presupuesto=presupuesto, **kwargs)
        
        if resultado:
            resultados.append({
                'Presupuesto': presupuesto,
                'Ganancia_Neta': resultado.get('ganancia_neta', resultado.get('precio_venta_real', 0) - presupuesto),
                'ROI': resultado.get('roi_porcentaje', 0),
                'Precio_Final': resultado.get('precio_nuevo_real', resultado.get('precio_venta_real', 0)),
            })
        else:
            resultados.append({
                'Presupuesto': presupuesto,
                'Ganancia_Neta': None,
                'ROI': None,
                'Precio_Final': None,
            })
    
    df_sensibilidad = pd.DataFrame(resultados)
    print("\n" + "="*70)
    print("ANÁLISIS DE SENSIBILIDAD - PRESUPUESTO")
    print("="*70)
    print(df_sensibilidad.to_string(index))


# ============================================================================
# GUARDADO DE RESULTADOS SECUENCIALES
# ============================================================================


def guardar_resultado(resultado, carpeta='results', prefijo='resultado_casa'):
    # Crear carpeta si no existe
    os.makedirs(carpeta, exist_ok=True)
    
    # Buscar el siguiente número disponible
    existentes = [f for f in os.listdir(carpeta) if f.startswith(prefijo)]
    numeros = [int(f[len(prefijo):].split('.')[0]) for f in existentes if f[len(prefijo):].split('.')[0].isdigit()]
    siguiente = max(numeros)+1 if numeros else 1
    
    # Crear path final
    filepath = os.path.join(carpeta, f"{prefijo}{siguiente}.txt")
    
    # Guardar resultado
    with open(filepath, 'w', encoding='utf-8') as f:
        for k, v in resultado.items():
            f.write(f"{k}: {v}\n")
    
    print(f"✅ Resultado guardado en {filepath}")
