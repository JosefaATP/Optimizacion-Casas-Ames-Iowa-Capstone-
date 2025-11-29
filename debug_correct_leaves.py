#!/usr/bin/env python3
"""
Manually evaluate each tree to see what the CORRECT leaf indices should be
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

booster_order = bundle.booster_feature_order()

print("=" * 70)
print("CORRECT TREE LEAF EVALUATION")
print("=" * 70)

correct_leaves_sum = 0.0
wrong_leaves_sum = 0.035883  # from previous run

discrepancies = []

for t_idx in range(min(5, len(dumps))):  # Check first 5 trees in detail
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
    
    # Get all leaves for this tree
    all_leaves = []
    def get_all_leaves(node):
        if "leaf" in node:
            all_leaves.append(float(node["leaf"]))
            return
        for ch in node.get("children", []):
            get_all_leaves(ch)
    
    get_all_leaves(node_json)
    
    # Which leaf index is the correct one?
    try:
        correct_leaf_idx = all_leaves.index(correct_leaf_val)
    except ValueError:
        correct_leaf_idx = -1
    
    # Try to find which leaf was selected in MIP (from the previous output)
    # This is hacky but we know tree 0 selected leaf 1 value=-0.022715
    # Let's just look at what XGBoost itself predicts
    
    print(f"\nTree {t_idx}:")
    print(f"  Correct leaf value: {correct_leaf_val:.6f}")
    print(f"  All leaves: {[f'{v:.6f}' for v in all_leaves]}")
    if correct_leaf_idx >= 0:
        print(f"  Correct leaf index: {correct_leaf_idx}")
    
    correct_leaves_sum += correct_leaf_val

print(f"\n" + "=" * 70)
print(f"Sum of first 5 correct leaves: {correct_leaves_sum:.6f}")
print(f"\nExpected total from all 914 trees: ~12.388893")
print(f"MIP is getting: 0.035883")
print(f"Difference: {12.388893 - 0.035883:.6f}")
