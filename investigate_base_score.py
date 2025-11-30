#!/usr/bin/env python3
"""
Verificar el cálculo del base_score y su relación con el mismatch.

Investigamos:
1. ¿Cuál es el base_score real según XGBoost?
2. ¿Se calcula correctamente en XGBBundle.__init__?
3. ¿Es este el culpable del 13.4% mismatch?
"""

import sys
import json
import numpy as np
import re
from pathlib import Path

WORKSPACE = Path(__file__).parent
sys.path.insert(0, str(WORKSPACE))

# Cargar datos y bundle usando el mismo camino que run_opt.py
from optimization.remodel.xgb_predictor import XGBBundle

print("="*80)
print("INVESTIGACIÓN DE BASE_SCORE")
print("="*80)

# Cargar bundle
bundle = XGBBundle()
print(f"\n[BUNDLE] b0_offset almacenado: {bundle.b0_offset}")

# Obtener booster
bst = bundle.reg.get_booster()

# Método 1: Usando attr
print("\n[METHOD 1] bst.attr('base_score'):")
try:
    bs_attr = bst.attr("base_score")
    print(f"  Resultado crudo: {repr(bs_attr)}")
    print(f"  Tipo: {type(bs_attr)}")
    if bs_attr:
        print(f"  Interpretación: {float(bs_attr)}")
except Exception as e:
    print(f"  Error: {e}")

# Método 2: Usando save_raw JSON
print("\n[METHOD 2] bst.save_raw('json') -> learner_model_param.base_score:")
try:
    json_model = bst.save_raw("json")
    data = json.loads(json_model)
    bs_json = data.get("learner", {}).get("learner_model_param", {}).get("base_score", "NOT FOUND")
    print(f"  Valor en JSON: {repr(bs_json)}")
    print(f"  Tipo: {type(bs_json)}")
    
    # Intentar interpretarlo con regex
    if isinstance(bs_json, str) and "[" in bs_json:
        m = re.match(r"\[\s*([0-9.eE+-]+)\s*\]", bs_json)
        if m:
            value = float(m.group(1))
            print(f"  Regex match: {value}")
        else:
            print(f"  Regex NO coincide")
    else:
        try:
            value = float(bs_json)
            print(f"  Direct float: {value}")
        except:
            print(f"  Cannot convert to float")
except Exception as e:
    print(f"  Error: {e}")

# Método 3: Calcular manualmente
print("\n[METHOD 3] Manual calculation (predict at origin - sum_of_leaves at origin):")
try:
    # Predict at zero vector
    X_zero = np.zeros((1, len(bundle.feature_names_in())))
    y_margin_zero = bundle.reg.predict(X_zero, output_margin=True)[0]
    print(f"  predict(zeros, output_margin=True): {y_margin_zero:.8f}")
    
    # Sum of leaves at zero vector
    sum_leaves_zero = bundle._eval_sum_leaves(X_zero.ravel())
    print(f"  _eval_sum_leaves(zeros): {sum_leaves_zero:.8f}")
    
    # Implied base score
    implied_b0 = y_margin_zero - sum_leaves_zero
    print(f"  Implied b0 = {implied_b0:.8f}")
except Exception as e:
    print(f"  Error: {e}")

# Comparación
print("\n" + "="*80)
print("RESUMEN")
print("="*80)
print(f"Bundle.b0_offset: {bundle.b0_offset}")
print(f"Stored vs Method 3: Δ = {bundle.b0_offset - implied_b0:.8f}")

if abs(bundle.b0_offset - implied_b0) > 1e-6:
    print("\n⚠️  ALERTA: Los valores no coinciden!")
    print("   Posible causa del mismatch de predicciones.")
else:
    print("\n✓ Los valores coinciden. Base score es correcto.")

