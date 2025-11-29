"""
Investigar si el preprocesamiento es diferente entre training y prediction
"""
from optimization.remodel.xgb_predictor import XGBBundle
import pandas as pd
import numpy as np

bundle = XGBBundle()

# Load base data
df_all = pd.read_csv('data/processed/base_completa_sin_nulos.csv')
X_test = df_all[df_all['PID'] == 526351010].drop(columns=['SalePrice', 'PID', 'SalePrice_Present'], errors='ignore')

print("="*80)
print("PREPROCESSING INVESTIGATION")
print("="*80)

# Check 1: Feature names
print("\n[1] Feature names in pipeline:")
print(f"    Num features from bundle.feature_names_in(): {len(bundle.feature_names_in())}")
print(f"    Num columns in X_test: {len(X_test.columns)}")

# Check 2: Booster feature order
booster_order = bundle.booster_feature_order()
print(f"\n[2] Booster feature order:")
print(f"    Num features expected by booster: {len(booster_order)}")
print(f"    First 10: {booster_order[:10]}")

# Check 3: Try preprocessing
try:
    Xp = bundle.pre.transform(X_test)
    print(f"\n[3] Preprocessing result:")
    print(f"    Shape after preprocessing: {Xp.shape}")
    print(f"    Dtype: {Xp.dtype}")
    print(f"    Expected num features: {len(booster_order)}")
    print(f"    Actual num features: {Xp.shape[1]}")
    
    if Xp.shape[1] != len(booster_order):
        print(f"    ✗ MISMATCH! Preprocessed has {Xp.shape[1]} features but booster expects {len(booster_order)}")
    else:
        print(f"    ✓ Feature count matches")
        
    # Check 4: Test prediction with and without preprocessing
    print(f"\n[4] Predictions:")
    
    # Method 1: Use bundle.predict (includes preprocessing)
    y1 = bundle.predict(X_test)
    print(f"    bundle.predict() = {y1.values[0]:.2f}")
    
    # Method 2: Direct regressor prediction on preprocessed data
    try:
        y2_pred = bundle.reg.predict(Xp)
        print(f"    bundle.reg.predict(Xp) = {y2_pred[0]:.2f}")
    except Exception as e:
        print(f"    bundle.reg.predict(Xp) - Error: {e}")
    
    # Method 3: output_margin
    try:
        y2_margin = bundle.reg.predict(Xp, output_margin=True)
        print(f"    bundle.reg.predict(Xp, output_margin=True) = {y2_margin[0]:.6f}")
    except Exception as e:
        print(f"    bundle.reg.predict(..., output_margin=True) - Error: {e}")
    
except Exception as e:
    print(f"\n[3] Preprocessing ERROR: {e}")

# Check 5: Booster internals
print(f"\n[5] Booster internals:")
try:
    bst = bundle.reg.get_booster()
    print(f"    Base score: {bst.attr('base_score')}")
except:
    print(f"    Could not get base_score")

print(f"    bundle.b0_offset: {bundle.b0_offset}")
print(f"    bundle.log_target: {bundle.log_target}")

# Check 6: Compare base_score values
print(f"\n[6] Base score comparison:")
try:
    import json
    bst = bundle.reg.get_booster()
    json_model = bst.save_raw("json")
    data = json.loads(json_model)
    bs_str = data.get("learner", {}).get("learner_model_param", {}).get("base_score", "NOT FOUND")
    print(f"    From JSON: {bs_str}")
    print(f"    bundle.b0_offset: {bundle.b0_offset}")
    
    if "E" in str(bs_str):
        import re
        m = re.match(r"\[\s*([0-9.eE+-]+)\s*\]", str(bs_str))
        if m:
            parsed = float(m.group(1))
            print(f"    Parsed: {parsed}")
except Exception as e:
    print(f"    Error: {e}")
