#!/usr/bin/env python3
"""
Script de prueba para validar el cálculo del base_score.
Compara diferentes métodos de extracción del base_score.
"""

import sys
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path

WORKSPACE = Path(__file__).parent
sys.path.insert(0, str(WORKSPACE))

from optimization.remodel.xgb_predictor import XGBBundle
from optimization.remodel.config import PATHS

print("\n" + "="*80)
print("VALIDACIÓN DE BASE_SCORE")
print("="*80)

# Cargar bundle
bundle = XGBBundle()
print(f"\n[1] Bundle cargado")
print(f"    - b0_offset en bundle: {bundle.b0_offset}")

# Obtener booster
bst = bundle.reg.get_booster()
print(f"\n[2] Booster obtenido")

# Método 1: bst.attr('base_score')
try:
    bs_attr = bst.attr("base_score")
    print(f"\n[3] bst.attr('base_score'): {bs_attr}")
except Exception as e:
    print(f"\n[3] Error en bst.attr: {e}")
    bs_attr = None

# Método 2: Extraer de JSON config
try:
    json_model = bst.save_raw("json")
    data = json.loads(json_model)
    bs_str = data.get("learner", {}).get("learner_model_param", {}).get("base_score", None)
    print(f"\n[4] base_score en JSON: {bs_str}")
    
    # Parsear si tiene formato especial
    if isinstance(bs_str, str) and "[" in bs_str:
        m = re.match(r"\[\s*([0-9.eE+-]+)\s*\]", bs_str)
        if m:
            bs_parsed = float(m.group(1))
            print(f"    - Parseado: {bs_parsed}")
except Exception as e:
    print(f"\n[4] Error extrayendo de JSON: {e}")

# Método 3: Calcular a partir de predict en punto cero
print(f"\n[5] Calculando base_score por predict en punto cero:")
try:
    zeros = np.zeros((1, 299))  # 299 features después del preprocesamiento
    y_margin_zero = float(bundle.reg.predict(zeros, output_margin=True)[0])
    print(f"    - y_margin en punto cero: {y_margin_zero:.8f}")
    
    # Calcular suma de hojas en punto cero
    dumps = bst.get_dump(with_stats=False, dump_format="json")
    leaves_sum_zero = 0.0
    for js in dumps:
        node = json.loads(js)
        cur = node
        while "leaf" not in cur:
            # En punto cero, x=0, así que go_yes si 0 < thr (siempre true)
            yes_id = cur.get("yes")
            for ch in cur.get("children", []):
                if ch.get("nodeid") == yes_id:
                    cur = ch
                    break
        if "leaf" in cur:
            leaves_sum_zero += float(cur["leaf"])
    
    calculated_b0 = y_margin_zero - leaves_sum_zero
    print(f"    - Suma de hojas en punto cero: {leaves_sum_zero:.8f}")
    print(f"    - Calculated base_score: {calculated_b0:.8f}")
except Exception as e:
    print(f"    - Error: {e}")
    import traceback
    traceback.print_exc()

# Comparar métodos
print(f"\n[RESUMEN]")
print(f"  Método 1 (bst.attr):        {bs_attr if bs_attr else 'N/A'}")
print(f"  Método 2 (JSON):            {bs_str if 'bs_str' in locals() else 'N/A'}")
print(f"  Método 3 (calculado):       {calculated_b0 if 'calculated_b0' in locals() else 'N/A'}")
print(f"  Bundle.b0_offset:           {bundle.b0_offset}")

# Ahora probar en una propiedad real
print(f"\n" + "="*80)
print("PRUEBA EN PROPIEDAD REAL")
print("="*80)

try:
    # Intentar cargar CSV
    csv_path = PATHS.base_csv
    train_df = pd.read_csv(csv_path)
    train_df = train_df.set_index("PID")
    
    pid = 528328100
    if pid in train_df.index:
        X_prop = train_df.loc[[pid]].values
        print(f"\n[TEST] Propiedad {pid}")
        
        # Predicción completa
        y_pred = bundle.reg.predict(X_prop, output_margin=True)[0]
        print(f"  - y_margin completo: {y_pred:.8f}")
        
        # Calcular suma de hojas
        leaves_sum = 0.0
        for js in dumps:
            node = json.loads(js)
            cur = node
            while "leaf" not in cur:
                f_idx = int(str(cur.get("split", "0")).replace("f", ""))
                thr = float(cur.get("split_condition", 0.0))
                x_val = X_prop[0, f_idx] if f_idx < len(X_prop[0]) else 0.0
                
                yes_id = cur.get("yes")
                go_yes = (x_val < thr)
                next_id = yes_id if go_yes else cur.get("no")
                
                found = False
                for ch in cur.get("children", []):
                    if ch.get("nodeid") == next_id:
                        cur = ch
                        found = True
                        break
                if not found:
                    break
            
            if "leaf" in cur:
                leaves_sum += float(cur["leaf"])
        
        print(f"  - Suma de hojas: {leaves_sum:.8f}")
        print(f"  - Diferencia (y_margin - hojas): {y_pred - leaves_sum:.8f}")
        print(f"  - Bundle.b0_offset: {bundle.b0_offset:.8f}")
        print(f"  - ¿Coincide? {abs((y_pred - leaves_sum) - bundle.b0_offset) < 1e-6}")
    else:
        print(f"[ERROR] Propiedad {pid} no encontrada")
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
