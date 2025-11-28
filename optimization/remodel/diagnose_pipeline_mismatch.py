"""
Script para diagnosticar las diferencias entre pipe_for_gurobi() y pipe_full.

La sospecha: pipe_for_gurobi() y pipe_full están devolviendo predicciones en escalas diferentes.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from .xgb_predictor import XGBBundle

def diagnose():
    bundle = XGBBundle()
    
    # Lee una fila de la base de datos
    df = pd.read_csv(Path("data/processed/base_completa_sin_nulos.csv"))
    
    # Filtra una fila con todas las categorías presentes
    X_base = df.iloc[[0]].copy()
    # Asegúrate de que todas las columnas esperadas estén presentes
    expected_cols = bundle.feature_names_in()
    for col in expected_cols:
        if col not in X_base.columns:
            X_base[col] = 0
    
    print("=" * 80)
    print("DIAGNÓSTICO: pipe_for_gurobi() vs pipe_full")
    print("=" * 80)
    
    # 1. Pipeline FULL
    print("\n[1] pipe_full.predict(X_base):")
    try:
        y_full = bundle.pipe_full.predict(X_base[expected_cols])
        print(f"    Valor: {y_full[0]:,.2f}")
        print(f"    Escala: asumida como ORIGINAL (precio)")
    except Exception as e:
        print(f"    ERROR: {e}")
        y_full = None
    
    # 2. Pipeline PARA EMBED
    print("\n[2] pipe_for_gurobi().predict(X_base):")
    try:
        y_embed = bundle.pipe_for_gurobi().predict(X_base[expected_cols])
        print(f"    Valor: {y_embed[0]:.6f}")
        print(f"    Escala: asumida como LOG (raw margin del XGB)")
    except Exception as e:
        print(f"    ERROR: {e}")
        y_embed = None
    
    if y_full is None or y_embed is None:
        print("\nNo se pudieron ejecutar ambas pipelines. Saliendo.")
        return
    
    # 3. Comparar con expm1
    print("\n[3] Intentar revertir y_embed:")
    y_embed_reverted = np.expm1(y_embed[0])
    print(f"    expm1(y_embed) = {y_embed_reverted:,.2f}")
    print(f"    Diferencia vs y_full: {abs(y_embed_reverted - y_full[0]):,.2f}")
    
    # 4. Intentar convertir y_full a log
    print("\n[4] Convertir y_full a log:")
    y_full_as_log = np.log1p(y_full[0])
    print(f"    log1p(y_full) = {y_full_as_log:.6f}")
    print(f"    Diferencia vs y_embed: {abs(y_full_as_log - y_embed[0]):.6f}")
    
    # 5. Inspeccionar tipo de regressor
    print("\n[5] Inspeccionar estructura de pipe_full:")
    print(f"    Tipo de last step: {type(bundle.pipe_full.named_steps['xgb'])}")
    print(f"    ¿Es TransformedTargetRegressor? {bundle._ttr is not None}")
    if bundle._ttr is not None:
        print(f"    Transformer func: {bundle._ttr.transformer}")
    
    # 6. Inspeccionar estructura de pipe_for_gurobi
    print("\n[6] Inspeccionar estructura de pipe_for_gurobi:")
    print(f"    Tipo de xgb step: {type(bundle.pipe_for_gurobi().named_steps['xgb'])}")
    
    # 7. Resumo
    print("\n" + "=" * 80)
    print("RESUMEN DEL PROBLEMA:")
    print("=" * 80)
    print(f"pipe_full.predict devuelve escala ORIGINAL (precio)")
    print(f"  -> y_full = {y_full[0]:,.2f}")
    print(f"pipe_for_gurobi().predict devuelve escala LOG (raw margin)")
    print(f"  -> y_embed = {y_embed[0]:.6f}")
    print(f"Cuando conviertes y_embed con expm1, recuperas el precio: {y_embed_reverted:,.2f}")
    print(f"Diferencia numérica: {abs(y_embed_reverted - y_full[0]):,.2f} ({100*abs(y_embed_reverted - y_full[0])/y_full[0]:.3f}%)")
    
    if abs(y_embed_reverted - y_full[0]) < 1e-2:
        print("\n✓ Los pipelines son CONSISTENTES (match numérico)")
    else:
        print("\n✗ Los pipelines son INCONSISTENTES (divergencia importante)")

if __name__ == "__main__":
    diagnose()
