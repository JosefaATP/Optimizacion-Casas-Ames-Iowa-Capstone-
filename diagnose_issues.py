#!/usr/bin/env python3
"""
Diagnóstico: verificar número de árboles y restricciones
"""
import pandas as pd
import pickle
import json

# Load bundle and meta
bundle = pickle.load(open("models/xgb/completa_present_log_p2_1800_ELEGIDO/model_xgb.joblib", "rb"))
meta = json.load(open("models/xgb/completa_present_log_p2_1800_ELEGIDO/meta.json"))

print("=" * 60)
print("INFORMACIÓN DE ÁRBOLES")
print("=" * 60)
print(f"Meta best_iteration: {meta.get('best_iteration')}")
print(f"Meta best_ntree_limit: {meta.get('best_ntree_limit')}")

try:
    bst = bundle.reg.get_booster()
except:
    bst = getattr(bundle.reg, "_Booster", None)

if bst:
    dumps = bst.get_dump(with_stats=False, dump_format="json")
    print(f"Total árboles en booster: {len(dumps)}")
    print(f"bundle.n_trees_use: {bundle.n_trees_use}")
    
    if bundle.n_trees_use:
        print(f"✓ Usando PRIMEROS {bundle.n_trees_use} árboles en embed")
    else:
        print(f"✗ Usando TODOS {len(dumps)} árboles en embed")

print("\n" + "=" * 60)
print("VERIFICAR RESTRICCIONES")
print("=" * 60)

# Load test data
df = pd.read_csv("data/processed/base_completa_sin_nulos.csv")
test_row = df[df['PID'] == 526351010].iloc[0]

print("\nFeatures que CAMBIARON sin costo en solución:")
print("  Open Porch SF: 36 → 39.6 (costo=0)")
print("  Heating_GasA: 1 → 0 (costo=0)")

# Check constraints - Apéndice 3
print("\nVerificando restricciones (Apéndice 3):")

# Heating features are interdependent
heating_type = "GasA"  # base
new_heating_gasA = 0  # optimized value
if new_heating_gasA == 0:
    print(f"  ✗ VIOLACIÓN: Heating_GasA cambió de 1 a 0")
    print(f"    Esto cambiaría el tipo de calefacción")
    print(f"    Debería tener costo asociado")

porch_sf_base = test_row['Open Porch SF']
porch_sf_opt = 39.6
print(f"\n  Open Porch SF: {porch_sf_base} → {porch_sf_opt}")
if porch_sf_opt != porch_sf_base:
    print(f"    Diferencia: +{porch_sf_opt - porch_sf_base} SF")
    print(f"    Costo reportado: $0 (DEBERÍA TENER COSTO)")
