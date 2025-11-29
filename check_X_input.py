"""
Verificar que la solución del MIP tiene todas las 299 columnas
"""
from optimization.remodel.xgb_predictor import XGBBundle
from optimization.remodel.gurobi_model import build_mip_embed
from optimization.remodel.run_opt import get_base_house
from optimization.remodel import costs
import pandas as pd
import numpy as np

bundle = XGBBundle()
ct = costs.CostTables()

# Get a base house
pid = 526351010
base_house = get_base_house(pid, base_csv='data/processed/base_completa_sin_nulos.csv')

print(f"Base house type: {type(base_house)}")
print(f"Base house dir: {[x for x in dir(base_house) if not x.startswith('_')]}")

# Check what X_input is in the MIP
# Build a simple MIP
m = build_mip_embed(
    base_row=base_house,  # Pass the BaseHouse object directly
    budget=100000,
    ct=ct,
    bundle=bundle,
    verbose=0
)

# Check m._X_input
X_input = getattr(m, '_X_input', None)
print(f"\nm._X_input type: {type(X_input)}")
if X_input is not None:
    if hasattr(X_input, 'shape'):
        print(f"m._X_input shape: {X_input.shape}")
    elif hasattr(X_input, 'columns'):
        print(f"m._X_input columns: {len(X_input.columns)}")
        print(f"First 10 columns: {list(X_input.columns[:10])}")
    print(f"m._X_input:\n{X_input}")
    
    # Try to preprocess it
    try:
        Xp = bundle.pre.transform(X_input)
        print(f"\nPreprocessing X_input: SUCCESS")
        print(f"  Shape: {Xp.shape}")
    except Exception as e:
        print(f"\nPreprocessing X_input: ERROR - {e}")
