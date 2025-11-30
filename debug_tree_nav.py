#!/usr/bin/env python3
"""
Deep debug to understand where the y_log mismatch is coming from.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from optimization.remodel.xgb_predictor import XGBBundle
from optimization.remodel.gurobi_model import build_mip_embed
from optimization.remodel.io import get_base_house
from optimization.remodel.config import PARAMS

bundle = XGBBundle()
print(f"Bundle base_score: {bundle.b0_offset}")

# Get a tree and check leaf contributions
bst = bundle.reg.get_booster()
dumps = bst.get_dump(with_stats=False, dump_format="json")

# Test on a simple fixed X: all zeros
# Need to figure out the number of features
n_features = len(dumps[0].split('f')) - 1  # rough estimate
bst = bundle.reg.get_booster()
feature_names = bst.feature_names
test_X = pd.DataFrame([[0.0] * len(feature_names)], columns=feature_names)
print(f"\nTest X shape: {test_X.shape}")

pred = bundle.predict_log_raw(test_X)
print(f"Prediction on zeros: {float(pred.iloc[0]):.6f}")

# Now let's manually walk a few trees to see their contributions
total_manual = bundle.b0_offset
for t_idx in range(min(3, len(dumps))):
    node = json.loads(dumps[t_idx])
    
    # Find which leaf we reach with all features = 0
    def find_leaf(nd):
        if "leaf" in nd:
            return float(nd["leaf"])
        f_idx = int(str(nd["split"]).replace("f", ""))
        thr = float(nd["split_condition"])
        
        x_val = 0.0  # Test with x = 0
        
        if x_val < thr:
            # Go to yes (left) child
            yes_id = nd.get("yes")
            for ch in nd.get("children", []):
                if ch.get("nodeid") == yes_id:
                    return find_leaf(ch)
        else:
            # Go to no (right) child
            no_id = nd.get("no")
            for ch in nd.get("children", []):
                if ch.get("nodeid") == no_id:
                    return find_leaf(ch)
        
        return None  # shouldn't reach here
    
    leaf_val = find_leaf(node)
    print(f"Tree {t_idx}: leaf_value = {leaf_val:.6f}")
    total_manual += leaf_val

print(f"\nManual total (b0 + tree leaves): {total_manual:.6f}")
print(f"Actual predict_log_raw: {float(pred.iloc[0]):.6f}")
print(f"Difference: {abs(total_manual - float(pred.iloc[0])):.9f}")
