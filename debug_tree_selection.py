#!/usr/bin/env python3
"""
Detailed diagnostic: which specific trees differ between Gurobi and XGBoost?
"""
import sys
import pandas as pd
sys.path.insert(0, 'optimization/remodel')

from config import PATHS
import pickle
import json
import numpy as np
import gurobipy as gp

# Load model and bundle
model_pkl = "models/xgb/completa_present_log_p2_1800_ELEGIDO/xgb_model.pkl"
with open(model_pkl, "rb") as f:
    bundle = pickle.load(f)

# Load test data
test_pid = 526351010
df = pd.read_csv("data/processed/base_completa_sin_nulos.csv")
test_row = df[df['PID'] == test_pid].iloc[0]

# Prepare X with modified values
X_test = test_row.to_frame().T.copy()
# Make the same modifications as in the optimization
X_test['Open Porch SF'] = 39.6  # increased from 36
X_test['Heating_GasA'] = 0      # changed from 1

print(f"Test X shape: {X_test.shape}")
print(f"Open Porch SF: {X_test['Open Porch SF'].values[0]}")
print(f"Heating_GasA: {X_test['Heating_GasA'].values[0]}")

# Get XGBoost's decision
bst = bundle.reg.get_booster()
dumps = bst.get_dump(with_stats=False, dump_format="json")

# Transform X
try:
    Xp = bundle.pre.transform(X_test)
except:
    Xp = X_test.values

# Get XGBoost predictions
pred_raw = bundle.reg.predict(Xp, output_margin=True)[0]
print(f"\nXGBoost output_margin (with base_score): {pred_raw:.6f}")

# Now, for EACH tree, get which leaf XGBoost selected
xgb_leaves = []
for t_idx, js in enumerate(dumps[:min(100, len(dumps))]):  # Check first 100 trees
    node = json.loads(js)
    
    # Traverse tree following XGBoost's logic
    def walk_xgb(nd):
        if "leaf" in nd:
            return float(nd["leaf"])
        # Get split info
        f_idx = int(str(nd["split"]).replace("f", ""))
        thr = float(nd["split_condition"])
        
        # Get feature value (use transformed X which is numeric)
        try:
            feat_val = float(Xp[0, f_idx])
        except:
            feat_val = float(Xp[0][f_idx])
        
        # XGBoost goes LEFT if x < threshold (yes), RIGHT if x >= threshold (no)
        if feat_val < thr:
            # Go left (yes)
            for ch in nd.get("children", []):
                if ch.get("nodeid") == nd.get("yes"):
                    return walk_xgb(ch)
        else:
            # Go right (no)
            for ch in nd.get("children", []):
                if ch.get("nodeid") == nd.get("no"):
                    return walk_xgb(ch)
    
    leaf_val = walk_xgb(node)
    xgb_leaves.append((t_idx, leaf_val))
    if t_idx < 10:
        print(f"Tree {t_idx}: XGBoost leaf value = {leaf_val:.6f}")

xgb_sum = sum(v for _, v in xgb_leaves)
print(f"\nXGBoost sum of leaves (first 100 trees): {xgb_sum:.6f}")

# Compare with what MIP selected
# Unfortunately we don't have the MIP solution here, but we can at least verify
# that we can get the XGBoost leaves by traversing correctly
base_score = float(bst.attr("base_score") or 0.5)
print(f"Base score: {base_score}")
print(f"Expected y_log_raw: {xgb_sum:.6f}")
print(f"Expected y_log: {xgb_sum + base_score:.6f}")
