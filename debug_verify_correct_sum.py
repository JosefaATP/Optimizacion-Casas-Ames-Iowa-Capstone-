#!/usr/bin/env python3
"""
Check: is the sum of CORRECT leaves exactly the negative of what MIP is getting?
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

bst = bundle.reg.get_booster()
dumps = bst.get_dump(with_stats=False, dump_format="json")

print("=" * 70)
print("CHECKING ALL TREES FOR CORRECT LEAF VALUES")
print("=" * 70)

correct_leaves_sum = 0.0

for t_idx in range(len(dumps)):
    node_json = json.loads(dumps[t_idx])
    
    # Manually walk the tree following split conditions
    def find_leaf_value(node):
        if "leaf" in node:
            return float(node["leaf"])
        
        f_idx_str = node["split"]
        f_idx = int(f_idx_str.replace("f", ""))
        thr = float(node["split_condition"])
        
        feat_val = X_array[f_idx]
        
        yes_id = node.get("yes")
        no_id = node.get("no")
        
        # Find yes and no children
        yes_child = None
        no_child = None
        for ch in node.get("children", []):
            if ch.get("nodeid") == yes_id:
                yes_child = ch
            elif ch.get("nodeid") == no_id:
                no_child = ch
        
        # In XGBoost: yes is left (x < thr), no is right (x >= thr)
        if feat_val < thr:
            if yes_child is not None:
                return find_leaf_value(yes_child)
        else:
            if no_child is not None:
                return find_leaf_value(no_child)
        
        return 0.0  # fallback
    
    correct_leaf_val = find_leaf_value(node_json)
    correct_leaves_sum += correct_leaf_val

print(f"\nSum of CORRECT leaves (all 914 trees): {correct_leaves_sum:.6f}")
print(f"External predict_log_raw: 12.388893")
print(f"MIP is getting: 0.035883")
print(f"Negative of MIP: {-0.035883:.6f}")
print(f"\nIs correct_sum == external value? {abs(correct_leaves_sum - 12.388893) < 0.001}")
print(f"Is correct_sum == -MIP value? {abs(correct_leaves_sum - (-0.035883)) < 0.001}")
