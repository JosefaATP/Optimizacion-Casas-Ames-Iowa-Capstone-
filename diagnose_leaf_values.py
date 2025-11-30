#!/usr/bin/env python3
"""
Diagnóstico profundo del mismatch de predicciones del árbol XGBoost en MIP.

Investigamos:
1. ¿Se extraen correctamente los valores de las hojas?
2. ¿Se aplica correctamente el base_score?
3. ¿Hay errores de redondeo acumulados?
4. ¿Hay problemas en la selección de hojas?
5. ¿Hay problemas de alineación de características?
"""

import sys
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup paths
WORKSPACE = Path(__file__).parent
sys.path.insert(0, str(WORKSPACE))

from optimization.remodel.config import PATHS
from optimization.remodel.xgb_predictor import XGBBundle

def load_data():
    """Carga los datos de Ames."""
    try:
        train_df = pd.read_parquet(PATHS.data_clean / "train_clean.parquet")
        return train_df
    except Exception as e:
        print(f"[ERROR] No se pudo cargar train_clean.parquet: {e}")
        return None

def analyze_xgb_model():
    """Analiza la estructura interna del modelo XGBoost."""
    print("\n" + "="*80)
    print("ANÁLISIS DE ESTRUCTURA INTERNA DEL MODELO XGBOOST")
    print("="*80)
    
    try:
        bundle = joblib.load(PATHS.xgb_model_file)
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el modelo: {e}")
        return None
    
    print(f"\n[INFO] Modelo cargado: {PATHS.xgb_model_file}")
    print(f"[INFO] n_trees_use = {bundle.n_trees_use}")
    print(f"[INFO] b0_offset = {bundle.b0_offset}")
    
    # Obtener el booster
    try:
        bst = bundle.reg.get_booster()
    except:
        bst = getattr(bundle.reg, "_Booster", None)
    
    if bst is None:
        print("[ERROR] No se pudo obtener el booster")
        return None
    
    # Obtener dumps de los árboles
    dumps = bst.get_dump(with_stats=False, dump_format="json")
    n_trees = len(dumps)
    print(f"\n[INFO] Número total de árboles en el modelo: {n_trees}")
    
    # Analizar primeros 5 árboles
    leaf_values_by_tree = []
    total_leaves = 0
    
    for t_idx in range(min(5, n_trees)):
        js = dumps[t_idx]
        node = json.loads(js)
        
        # Extraer todas las hojas
        leaves = []
        def walk(nd, path):
            if "leaf" in nd:
                leaves.append((path, float(nd["leaf"])))
                return
            f_idx = int(str(nd["split"]).replace("f", ""))
            thr = float(nd["split_condition"])
            yes_id = nd.get("yes")
            no_id = nd.get("no")
            if yes_id is not None:
                walk(node[yes_id], path + [(f_idx, thr, True)])  # yes = left
            if no_id is not None:
                walk(node[no_id], path + [(f_idx, thr, False)])  # no = right
        
        walk(node, [])
        
        # Estadísticas del árbol
        leaf_vals = [v for _, v in leaves]
        print(f"\n  Árbol {t_idx}:")
        print(f"    - Número de hojas: {len(leaves)}")
        print(f"    - Valores de hojas: min={min(leaf_vals):.8f}, max={max(leaf_vals):.8f}, mean={np.mean(leaf_vals):.8f}")
        print(f"    - Suma de valores: {sum(leaf_vals):.8f}")
        print(f"    - Primeras 3 hojas: {leaf_vals[:3]}")
        
        leaf_values_by_tree.append(leaf_vals)
        total_leaves += len(leaves)
    
    print(f"\n[SUMMARY] Total de hojas analizadas (primeros 5 árboles): {total_leaves}")
    
    # Verificar base_score
    print(f"\n[BASE_SCORE] b0_offset = {bundle.b0_offset}")
    try:
        bs_attr = bst.attr("base_score")
        print(f"[BASE_SCORE] bst.attr('base_score') = {bs_attr}")
    except:
        print("[BASE_SCORE] No se pudo obtener base_score del booster")
    
    return bundle, dumps, leaf_values_by_tree

def test_prediction_on_property(bundle, pid, budget=50000):
    """Prueba la predicción en una propiedad específica."""
    print("\n" + "="*80)
    print(f"PRUEBA DE PREDICCIÓN EN PROPIEDAD {pid} CON BUDGET {budget}")
    print("="*80)
    
    try:
        # Cargar datos
        train_df = load_data()
        if train_df is None:
            return
        
        # Obtener la propiedad
        if pid not in train_df.index:
            print(f"[ERROR] Propiedad {pid} no encontrada en datos")
            return
        
        X_property = train_df.loc[[pid]]
        print(f"\n[INFO] Propiedad encontrada: {pid}")
        print(f"[INFO] Característica shape: {X_property.shape}")
        
        # Predicción del modelo sin MIP
        y_pred_raw = bundle.reg.predict(X_property, output_margin=True)[0]
        print(f"\n[PREDICTION] y_raw (con base_score): {y_pred_raw:.8f}")
        
        # Obtener margin sin base_score
        y_leaves = bundle._eval_sum_leaves(X_property.values.ravel())
        print(f"[PREDICTION] y_leaves (suma de hojas): {y_leaves:.8f}")
        print(f"[PREDICTION] base_score calculado: {y_pred_raw - y_leaves:.8f}")
        
        # Análisis árbol por árbol
        print(f"\n[DETAILED] Análisis de cada árbol:")
        
        bst = bundle.reg.get_booster()
        dumps = bst.get_dump(with_stats=False, dump_format="json")
        
        total_sum = 0.0
        for t_idx in range(min(10, len(dumps))):
            js = dumps[t_idx]
            node = json.loads(js)
            
            # Extraer hojas y valores
            leaves = []
            def walk(nd, path):
                if "leaf" in nd:
                    leaves.append((path, float(nd["leaf"])))
                    return
                f_idx = int(str(nd["split"]).replace("f", ""))
                thr = float(nd["split_condition"])
                yes_id = nd.get("yes")
                no_id = nd.get("no")
                if yes_id is not None:
                    walk(node[yes_id], path + [(f_idx, thr, True)])
                if no_id is not None:
                    walk(node[no_id], path + [(f_idx, thr, False)])
            
            walk(node, [])
            
            # Simular la selección de hoja en la propiedad
            X_row = X_property.values.ravel()
            selected_leaf_idx = None
            
            for k, (conds, leaf_val) in enumerate(leaves):
                all_satisfied = True
                for f_idx, thr, is_left in conds:
                    x_val = X_row[f_idx]
                    if is_left:
                        if not (x_val < thr):
                            all_satisfied = False
                            break
                    else:
                        if not (x_val >= thr):
                            all_satisfied = False
                            break
                
                if all_satisfied:
                    selected_leaf_idx = k
                    break
            
            if selected_leaf_idx is not None:
                selected_val = leaves[selected_leaf_idx][1]
                total_sum += selected_val
                print(f"  Árbol {t_idx}: hoja {selected_leaf_idx} seleccionada, valor = {selected_val:.8f}, suma acum = {total_sum:.8f}")
            else:
                print(f"  Árbol {t_idx}: [WARNING] No se seleccionó ninguna hoja")
        
        print(f"\n[DETAILED] Suma de primeros 10 árboles: {total_sum:.8f}")
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()

def main():
    print("[START] Iniciando diagnóstico de valores de hojas...")
    
    # Análisis del modelo
    result = analyze_xgb_model()
    if result is None:
        return
    
    bundle, dumps, _ = result
    
    # Prueba en propiedades específicas
    test_prediction_on_property(bundle, pid=526351010, budget=50000)
    test_prediction_on_property(bundle, pid=528328100, budget=250000)
    
    print("\n" + "="*80)
    print("DIAGNÓSTICO COMPLETADO")
    print("="*80)

if __name__ == "__main__":
    main()
