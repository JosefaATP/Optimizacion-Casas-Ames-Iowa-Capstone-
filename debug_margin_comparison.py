#!/usr/bin/env python3
"""
Compare XGBoost's margin prediction with manual tree leaf summation
"""
import json
import numpy as np
import pandas as pd
from optimization.remodel.io import get_base_house
from optimization.remodel.xgb_predictor import XGBBundle
from optimization.remodel.gurobi_model import build_base_input_row

pid = 526351010
base_house = get_base_house(pid)
bundle = XGBBundle()

# Build base input
X_input = build_base_input_row(bundle, base_house.row)
X_array = X_input.values[0]  # Shape (299,)

print("=" * 70)
print("COMPARING XGBOOST MARGIN VS MANUAL TREE SUM")
print("=" * 70)

# Method 1: XGBoost's built-in margin prediction
xgb_margin = float(bundle.reg.predict(X_input, output_margin=True)[0])
print(f"\n1. XGBoost margin prediction: {xgb_margin:.6f}")

# Method 2: Manual tree leaf summation
bst = bundle.reg.get_booster()
dumps = bst.get_dump(with_stats=False, dump_format="json")

manual_sum = 0.0
for t_idx in range(len(dumps)):
    node_json = json.loads(dumps[t_idx])
    
    def find_leaf_value(node):
        if "leaf" in node:
            return float(node["leaf"])
        
        f_idx_str = node["split"]
        f_idx = int(f_idx_str.replace("f", ""))
        thr = float(node["split_condition"])
        
        feat_val = X_array[f_idx]
        
        yes_id = node.get("yes")
        no_id = node.get("no")
        
        yes_child = None
        no_child = None
        for ch in node.get("children", []):
            if ch.get("nodeid") == yes_id:
                yes_child = ch
            elif ch.get("nodeid") == no_id:
                no_child = ch
        
        if feat_val < thr:
            if yes_child is not None:
                return find_leaf_value(yes_child)
        else:
            if no_child is not None:
                return find_leaf_value(no_child)
        
        return 0.0
    
    leaf_val = find_leaf_value(node_json)
    manual_sum += leaf_val

print(f"2. Manual tree leaf sum: {manual_sum:.6f}")

# Method 3: What about base_score?
try:
    base_score_str = bst.attr("base_score")
    base_score = float(base_score_str) if base_score_str else 0.0
    print(f"3. Base score from booster: {base_score:.6f}")
    print(f"   Manual sum + base_score = {manual_sum + base_score:.6f}")
except Exception as e:
    print(f"3. Could not get base_score: {e}")

# Method 4: What does bundle.predict_log_raw return?
pred_log_raw = float(bundle.predict_log_raw(X_input).iloc[0])
print(f"\n4. bundle.predict_log_raw(): {pred_log_raw:.6f}")

print(f"\n" + "=" * 70)
print("ANALYSIS:")
print("=" * 70)
print(f"Does manual_sum match xgb_margin? {abs(manual_sum - xgb_margin) < 0.001}")
print(f"Does manual_sum + base_score match xgb_margin? {abs(manual_sum + base_score - xgb_margin) < 0.001}")
print(f"Does xgb_margin match predict_log_raw? {abs(xgb_margin - pred_log_raw) < 0.001}")
print(f"\nDelta (xgb_margin - manual_sum): {xgb_margin - manual_sum:.6f}")
print(f"Delta (pred_log_raw - xgb_margin): {pred_log_raw - xgb_margin:.6f}")
