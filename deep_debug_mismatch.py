#!/usr/bin/env python3
"""
Deep analysis of XGBoost mismatch: identify exactly which trees are selecting wrong leaves
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import json
import gurobipy as gp
from optimization.remodel.xgb_predictor import XGBBundle
from optimization.remodel.io import get_base_house
from optimization.remodel.gurobi_model import build_mip_embed

# Load data and bundle
print("=" * 80)
print("DEEP DEBUG: XGBoost Mismatch Analysis")
print("=" * 80)

PID = 528328100
budget = 250000

# Get base house
base_row = get_base_house(PID)
print(f"\n[INPUT] Property PID={PID}, Budget=${budget:,.0f}")

# Load XGB bundle
try:
    bundle = XGBBundle()
    print(f"[BUNDLE] Loaded: {bundle.reg.n_estimators} trees")
except Exception as e:
    print(f"[ERROR] Could not load bundle: {e}")
    sys.exit(1)

# Get feature names
feats = bundle.feature_names_in()
print(f"[FEATURES] Total features: {len(feats)}")

# Create test data with base house values
test_data = pd.DataFrame([base_row])
test_data = test_data[feats]

# Get XGBoost prediction
xgb_pred_log = bundle.predict_log_raw(test_data).iloc[0]
print(f"\n[XGB-BASELINE]")
print(f"  XGBoost predict_log_raw = {xgb_pred_log:.8f}")

# Now get tree-by-tree predictions to identify which trees are selecting wrong leaves
print(f"\n[TREE-BY-TREE ANALYSIS]")
print(f"  Checking each of {bundle.reg.n_estimators} trees individually...")

bst = bundle.reg.get_booster()
dumps = bst.get_dump(with_stats=False, dump_format="json")

Xp = bundle.pre.transform(test_data)
if hasattr(Xp, "toarray"):
    Xp_arr = Xp.toarray()
else:
    Xp_arr = Xp

# Get booster feature order
booster_features = bundle.booster_feature_order()
x_values = {}
for i, feat in enumerate(booster_features):
    if i < Xp_arr.shape[1]:
        x_values[i] = float(Xp_arr[0, i])

# Walk each tree and collect predictions
tree_predictions = []
mismatches = []

for t_idx, js in enumerate(dumps):
    tree_json = json.loads(js)
    
    # Walk tree to find selected leaf
    def walk_tree(node):
        if "leaf" in node:
            return float(node["leaf"])
        
        f_idx = int(str(node["split"]).replace("f", ""))
        threshold = float(node["split_condition"])
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
        
        x_val = x_values.get(f_idx, 0.0)
        
        # XGBoost logic: if x < threshold, go left (yes)
        if x_val < threshold:
            if yes_child:
                return walk_tree(yes_child)
            elif no_child:
                return walk_tree(no_child)
        else:
            # x >= threshold, go right (no)
            if no_child:
                return walk_tree(no_child)
            elif yes_child:
                return walk_tree(yes_child)
        
        return 0.0
    
    leaf_val = walk_tree(tree_json)
    tree_predictions.append(leaf_val)

sum_leaves_xgb = np.sum(tree_predictions)
print(f"  Sum of all tree leaves (XGB direct): {sum_leaves_xgb:.8f}")
print(f"  Base score: {bundle.b0_offset:.8f}")
print(f"  Total (with base score): {sum_leaves_xgb + bundle.b0_offset:.8f}")
print(f"  Matches XGB predict_log_raw? {np.isclose(sum_leaves_xgb, xgb_pred_log, atol=1e-6)}")

# Now build MIP and solve it to see what the MIP selects
print(f"\n[MIP ANALYSIS]")
print(f"  Building MIP model...")

m = gp.Model("mip_debug")
m.Params.LogToConsole = 0
m.Params.TimeLimit = 10  # Short time limit for quick analysis

# Create variables for each feature
x_vars = []
for feat in feats:
    lb, ub = -1e6, 1e6
    v = m.addVar(lb=lb, ub=ub, name=f"x_{feat}")
    x_vars.append(v)
    # Fix base values
    if feat in base_row:
        v_val = float(base_row[feat])
        v.LB = v_val
        v.UB = v_val

m.update()

# Add tree embedding
y_log_raw = m.addVar(lb=-gp.GRB.INFINITY, name="y_log_raw")
bundle.attach_to_gurobi(m, x_vars, y_log_raw, eps=-1e-6)

# Dummy objective
m.setObjective(0, gp.GRB.MINIMIZE)

# Optimize
m.optimize()

# Read solution
X_sol = np.array([v.X for v in x_vars])
y_log_mip = y_log_raw.X

print(f"  MIP y_log_raw = {y_log_mip:.8f}")
print(f"  XGB y_log_raw = {xgb_pred_log:.8f}")
print(f"  Mismatch (MIP - XGB) = {y_log_mip - xgb_pred_log:+.8f}")

# Now check which trees have mismatches
print(f"\n[LEAF SELECTION MISMATCH]")
print(f"  Checking which tree leaves the MIP selected...")

tree_leaves_mip = []
mismatch_count = 0
mismatch_details = []

for t_idx, js in enumerate(dumps):
    tree_json = json.loads(js)
    
    # Get leaf index from MIP solution
    leaf_vars = [m.getVarByName(f"t{t_idx}_leaf{k}") for k in range(100)]
    leaf_vars = [v for v in leaf_vars if v is not None]
    
    mip_selected_leaf = None
    for k, v in enumerate(leaf_vars):
        if v.X > 0.5:
            mip_selected_leaf = k
            break
    
    if mip_selected_leaf is None:
        continue
    
    # Get the leaves and their values for this tree
    leaves = []
    def walk_leaves(node, path=[]):
        if "leaf" in node:
            leaves.append((path, float(node["leaf"])))
            return
        f_idx = int(str(node["split"]).replace("f", ""))
        thr = float(node["split_condition"])
        yes_id = node.get("yes")
        no_id = node.get("no")
        
        yes_child = None
        no_child = None
        for ch in node.get("children", []):
            if ch.get("nodeid") == yes_id:
                yes_child = ch
            elif ch.get("nodeid") == no_id:
                no_child = ch
        
        if yes_child:
            walk_leaves(yes_child, path + [(f_idx, thr, True)])
        if no_child:
            walk_leaves(no_child, path + [(f_idx, thr, False)])
    
    walk_leaves(tree_json)
    
    if mip_selected_leaf < len(leaves):
        mip_leaf_val = leaves[mip_selected_leaf][1]
        xgb_leaf_val = tree_predictions[t_idx]
        
        if not np.isclose(mip_leaf_val, xgb_leaf_val, atol=1e-8):
            mismatch_count += 1
            mismatch_details.append({
                'tree': t_idx,
                'xgb_leaf': xgb_leaf_val,
                'mip_leaf': mip_leaf_val,
                'diff': mip_leaf_val - xgb_leaf_val,
                'mip_leaf_idx': mip_selected_leaf,
                'xgb_conditions': leaves[mip_selected_leaf][0]
            })

print(f"\n  Mismatches found: {mismatch_count} trees out of {len(dumps)}")
print(f"  Mismatch percentage: {100*mismatch_count/len(dumps):.2f}%")

if mismatch_details:
    print(f"\n  Top 20 mismatches by contribution:")
    sorted_mismatches = sorted(mismatch_details, key=lambda x: abs(x['diff']), reverse=True)
    for i, m_detail in enumerate(sorted_mismatches[:20]):
        print(f"    {i+1}. Tree {m_detail['tree']}: {m_detail['diff']:+.6f} "
              f"(XGB={m_detail['xgb_leaf']:.6f}, MIP={m_detail['mip_leaf']:.6f})")

total_mismatch_contribution = np.sum([m['diff'] for m in mismatch_details])
print(f"\n  Total contribution from mismatches: {total_mismatch_contribution:+.8f}")
print(f"  Expected total mismatch: {y_log_mip - xgb_pred_log:+.8f}")

print(f"\n" + "=" * 80)
