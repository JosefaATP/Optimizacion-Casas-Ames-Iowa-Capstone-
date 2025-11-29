"""
Verificar si las 299 variables del MIP coinciden exactamente con las del preprocessor
"""
from optimization.remodel.xgb_predictor import XGBBundle
from optimization.remodel.run_opt import get_base_house, build_mip_embed
from optimization.remodel import costs
import pandas as pd

bundle = XGBBundle()
ct = costs.CostTables()

# Get base house (80 raw features)
base_house = get_base_house(526351010, 'data/processed/base_completa_sin_nulos.csv')
print(f"[1] Base house features: {len(base_house.row)} variables")
print(f"    First 10: {list(base_house.row.index)[:10]}")

# Build preprocessing manually
df_all = pd.read_csv('data/processed/base_completa_sin_nulos.csv')
X_raw = df_all[df_all['PID'] == 526351010].drop(columns=['SalePrice', 'PID', 'SalePrice_Present'], errors='ignore')

print(f"\n[2] Raw DataFrame: {X_raw.shape[1]} columns")
print(f"    First 10: {list(X_raw.columns)[:10]}")

# Apply preprocessing
Xp = bundle.pre.transform(X_raw)
print(f"\n[3] After preprocessing: {Xp.shape[1]} features")
print(f"    Type: {type(Xp)}")

# Get booster feature order
booster_order = bundle.booster_feature_order()
print(f"\n[4] Booster expects: {len(booster_order)} features")
print(f"    First 10: {booster_order[:10]}")
print(f"    Last 10:  {booster_order[-10:]}")

# Build MIP
print(f"\n[5] Building MIP...")
m = build_mip_embed(
    base_row=base_house,
    budget=100000,
    ct=ct,
    bundle=bundle,
    verbose=0,
)

# Get X_input from MIP
X_input = getattr(m, '_X_input', None)
if X_input is not None:
    print(f"\n[6] MIP X_input: {X_input.shape[1]} columns")
    print(f"    First 10: {list(X_input.columns)[:10]}")
    print(f"    Last 10:  {list(X_input.columns)[-10:]}")
    
    # Compare
    print(f"\n[7] COMPARISON:")
    if Xp.shape[1] != X_input.shape[1]:
        print(f"    ✗ DIFFERENT NUMBER OF FEATURES!")
        print(f"      Preprocessed: {Xp.shape[1]}")
        print(f"      MIP X_input:  {X_input.shape[1]}")
    else:
        print(f"    ✓ Same number of features: {Xp.shape[1]}")
    
    # Check if columns match
    if isinstance(Xp, pd.DataFrame):
        xp_cols = set(Xp.columns)
    else:
        xp_cols = set(range(Xp.shape[1]))
    
    mip_cols = set(X_input.columns)
    
    print(f"\n[8] COLUMN MATCHING:")
    if xp_cols == mip_cols:
        print(f"    ✓ Columns match exactly")
    else:
        print(f"    ✗ Columns don't match")
        missing_in_xp = mip_cols - xp_cols
        missing_in_mip = xp_cols - mip_cols
        if missing_in_xp:
            print(f"    Missing in Xp: {list(missing_in_xp)[:5]}...")
        if missing_in_mip:
            print(f"    Missing in MIP: {list(missing_in_mip)[:5]}...")
    
    # Check column order
    if isinstance(Xp, pd.DataFrame) and list(Xp.columns) != list(X_input.columns):
        print(f"    ⚠ Column ORDER is different!")
        # Find first mismatch
        for i, (c1, c2) in enumerate(zip(Xp.columns, X_input.columns)):
            if c1 != c2:
                print(f"    First mismatch at position {i}: '{c1}' vs '{c2}'")
                break
else:
    print(f"\n[6] No X_input in MIP")
