#!/usr/bin/env python3
"""
Minimal test: create a simple 2-tree XGBoost model and verify tree embedding constraints work correctly.
"""
import sys
sys.path.insert(0, 'optimization/remodel')

import xgboost as xgb
import numpy as np
import pandas as pd
import gurobipy as gp

# Create simple data
X_train = np.array([
    [1.0, 0.0],
    [2.0, 0.0],
    [3.0, 1.0],
    [4.0, 1.0],
], dtype=np.float32)
y_train = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32)

# Train simple XGB model
params = {
    'objective': 'reg:squarederror',
    'max_depth': 2,
    'learning_rate': 1.0,
    'num_rounds': 1,
    'tree_method': 'hist',
}
dmatrix = xgb.DMatrix(X_train, label=y_train)
model = xgb.train(params, dmatrix, num_boost_round=2)

# Get booster and dump
bst = model
dumps = bst.get_dump(with_stats=False, dump_format="json")

print(f"Number of trees: {len(dumps)}\n")

for t_idx, js in enumerate(dumps):
    print(f"Tree {t_idx}:\n{js}\n")

# Now test: for a specific input, which leaf does XGBoost select?
test_X = np.array([[2.5, 0.5]], dtype=np.float32)
xgb_pred = model.predict(xgb.DMatrix(test_X))[0]
print(f"XGBoost prediction for {test_X[0]}: {xgb_pred}")

# Now manually traverse and check which leaves are selected
import json

def walk_xgb(nd, path=""):
    if "leaf" in nd:
        print(f"  -> Leaf: value={nd['leaf']}, path={path}")
        return float(nd["leaf"])
    
    f_idx = int(str(nd["split"]).replace("f", ""))
    thr = float(nd["split_condition"])
    feat_val = test_X[0, f_idx]
    
    print(f"  Split on f{f_idx} (val={feat_val:.2f}) < {thr:.2f}?", end=" ")
    
    if feat_val < thr:
        print("YES -> go left")
        for ch in nd.get("children", []):
            if ch.get("nodeid") == nd.get("yes"):
                return walk_xgb(ch, path + f" f{f_idx}<{thr}")
    else:
        print("NO -> go right")
        for ch in nd.get("children", []):
            if ch.get("nodeid") == nd.get("no"):
                return walk_xgb(ch, path + f" f{f_idx}>={thr}")

print("\nManual traversal:")
for t_idx, js in enumerate(dumps):
    node = json.loads(js)
    print(f"Tree {t_idx}:")
    leaf_val = walk_xgb(node)
    print(f"  Leaf value: {leaf_val}\n")
