#!/usr/bin/env python3
"""Check if bundle calculates y_log_raw same as embed should."""

import sys
from pathlib import Path
import json
import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from optimization.remodel.xgb_predictor import XGBBundle
import pandas as pd

bundle = XGBBundle()

# Create a simple test: all zeros
test_X = pd.DataFrame(np.zeros((1, 105)))
test_X.columns = [f'f{i}' for i in range(105)]

# Method 1: Use bundle.predict_log_raw
y_log_raw_bundle = float(bundle.predict_log_raw(test_X).iloc[0])

# Method 2: Manually walk trees
bst = bundle.reg.get_booster()
dumps = bst.get_dump(with_stats=False, dump_format="json")

y_log_raw_manual = bundle.b0_offset

for t_idx, js in enumerate(dumps):
    node = json.loads(js)
    
    # Walk tree with x=0 for all features
    def find_leaf(nd):
        if "leaf" in nd:
            return float(nd["leaf"])
        
        f_idx = int(str(nd["split"]).replace("f", ""))
        thr = float(nd["split_condition"])
        
        x_val = 0.0  # All features are 0
        
        yes_id = nd.get("yes")
        no_id = nd.get("no")
        
        if x_val < thr:
            # Go left (yes)
            for ch in nd.get("children", []):
                if ch.get("nodeid") == yes_id:
                    return find_leaf(ch)
        else:
            # Go right (no)
            for ch in nd.get("children", []):
                if ch.get("nodeid") == no_id:
                    return find_leaf(ch)
        
        return 0.0  # Should not reach
    
    leaf_val = find_leaf(node)
    y_log_raw_manual += leaf_val

print(f"bundle.predict_log_raw(zeros): {y_log_raw_bundle:.6f}")
print(f"Manual walk of trees:           {y_log_raw_manual:.6f}")
print(f"Difference:                     {abs(y_log_raw_bundle - y_log_raw_manual):.9f}")

if abs(y_log_raw_bundle - y_log_raw_manual) < 1e-6:
    print("\n[OK] Bundle and manual walk agree!")
else:
    print("\n[PROBLEM] Bundle and manual walk DISAGREE!")
