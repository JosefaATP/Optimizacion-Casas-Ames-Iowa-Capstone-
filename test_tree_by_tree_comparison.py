#!/usr/bin/env python3
"""
Diagnostico árbol-por-árbol: Compara qué hojas selecciona XGBoost vs MIP
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

WORKSPACE = Path(__file__).parent
sys.path.insert(0, str(WORKSPACE))

from optimization.remodel.xgb_predictor import XGBBundle
from optimization.remodel.config import PATHS
from optimization.remodel.gurobi_model import build_base_input_row

# Cargar bundle y datos
bundle = XGBBundle()
csv_path = PATHS.base_csv
df = pd.read_csv(csv_path)
df = df.set_index("PID")

# Propiedad test
pid = 528328100
house_data = df.loc[pid]
print(f"[LOAD] Propiedad {pid}")

# Obtener booster
bst = bundle.reg.get_booster()
dumps = bst.get_dump(with_stats=False, dump_format="json")
print(f"[INFO] Total de árboles: {len(dumps)}")

# Transformar datos
X_series = pd.DataFrame([house_data])
X_transformed = bundle.pre.transform(X_series)
if hasattr(X_transformed, 'toarray'):
    X_array = X_transformed.toarray()[0]
else:
    X_array = X_transformed[0]

print(f"[INFO] X shape after transform: {X_array.shape}")

# Predicción con XGBoost
y_margin = float(bundle.reg.predict(X_series, output_margin=True)[0])
print(f"\n[XGB] Predicción total (with base_score): {y_margin:.8f}")

# Ahora, para cada árbol, simular qué hoja selecciona XGBoost
print(f"\n" + "="*80)
print("COMPARACIÓN ÁRBOL POR ÁRBOL (primeros 20 árboles)")
print("="*80)

xgb_leaves = []
mip_leaves = []
total_xgb = 0.0
total_mip_selected_correctly = 0.0
total_mip_selected_incorrectly = 0.0

for t_idx in range(min(20, len(dumps))):
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
        
        yes_child = None
        no_child = None
        for ch in nd.get("children", []):
            ch_id = ch.get("nodeid")
            if ch_id == yes_id:
                yes_child = ch
            elif ch_id == no_id:
                no_child = ch
        
        if yes_child is not None:
            walk(yes_child, path + [(f_idx, thr, True)])
        if no_child is not None:
            walk(no_child, path + [(f_idx, thr, False)])
    
    walk(node, [])
    
    # Encontrar qué hoja selecciona XGBoost
    xgb_selected_idx = None
    for k, (conds, leaf_val) in enumerate(leaves):
        all_satisfied = True
        for f_idx, thr, is_left in conds:
            x_val = X_array[f_idx]
            if is_left:
                if not (x_val < thr):
                    all_satisfied = False
                    break
            else:
                if not (x_val >= thr):
                    all_satisfied = False
                    break
        
        if all_satisfied:
            xgb_selected_idx = k
            break
    
    if xgb_selected_idx is not None:
        xgb_leaf_val = leaves[xgb_selected_idx][1]
        xgb_leaves.append(xgb_leaf_val)
        total_xgb += xgb_leaf_val
        
        # Asumir que MIP selecciona la misma hoja
        total_mip_selected_correctly += xgb_leaf_val
        
        print(f"Tree {t_idx:3d}: XGB leaf {xgb_selected_idx:2d}, value = {xgb_leaf_val:+.6f}, cumsum_xgb = {total_xgb:+.6f}")
    else:
        print(f"Tree {t_idx:3d}: [WARNING] XGB no selecciono ninguna hoja!")

print(f"\n[SUMMARY]")
print(f"  Suma de primeros 20 árboles (XGB): {total_xgb:.8f}")
print(f"  Suma si MIP hubiera seleccionado igual: {total_mip_selected_correctly:.8f}")
print(f"  Diferencia esperada (0): {abs(total_xgb - total_mip_selected_correctly):.8f}")

# Ahora comparar con lo que MIP realmente predice
print(f"\n[MIP vs XGB]")
print(f"  y_margin(XGB con base_score): {y_margin:.8f}")
print(f"  y_log_raw(XGB) = y_margin - b0: {y_margin - bundle.b0_offset:.8f}")
print(f"  Suma manual primeros 20 árboles: {total_xgb:.8f}")
print(f"  Suma todos los {len(dumps)} árboles (estimated): {(total_xgb / min(20, len(dumps))) * len(dumps):.8f}")

