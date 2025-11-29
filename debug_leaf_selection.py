#!/usr/bin/env python3
"""
Build the MIP with the actual YLOG constraints and see which leaves are being selected
"""
import json
import gurobipy as gp
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
print(f"X_input shape: {X_input.shape}, columns: {len(X_input.columns)}")

# Predict externally first
pred_raw = bundle.predict_log_raw(X_input)
print(f"External predict_log_raw: {pred_raw}")
print(f"b0_offset: {bundle.b0_offset}")

# Create simple MIP with just YLOG embedding
m = gp.Model("test_ylog")
m.setParam("OutputFlag", 0)

# Create x_vars with base values
x_vars = []
for feat_name, base_val in zip(X_input.columns, X_input.values[0]):
    xv = m.addVar(lb=base_val, ub=base_val, name=feat_name)
    x_vars.append(xv)

# Create y_log_raw var
y_log_raw = m.addVar(lb=-100, ub=100, name="y_log_raw")

# Attach tree embedding
print("\n" + "=" * 70)
print("Attaching tree embedding...")
print("=" * 70)
bundle.attach_to_gurobi(m, x_vars, y_log_raw, eps=-1e-6)

# Add offset constraint
y_log = m.addVar(lb=-100, ub=100, name="y_log")
m.addConstr(y_log == y_log_raw + bundle.b0_offset, name="Y_LOG_OFFSET")

# Dummy objective
m.setObjective(y_log, gp.GRB.MAXIMIZE)

print("Optimizing...")
m.optimize()

print(f"\nMIP Status: {m.status}")
if m.status != 2:  # 2 = optimal
    print(f"ERROR: MIP did not solve optimally. Status code: {m.status}")
    if m.status == 3:
        print("Model is INFEASIBLE!")
    exit(1)

print("\n" + "=" * 70)
print("MIP RESULTS:")
print("=" * 70)
print(f"y_log_raw.X = {y_log_raw.X:.6f}")
print(f"y_log.X = {y_log.X:.6f}")
pred_raw_val = float(pred_raw.iloc[0])
print(f"Expected y_log_raw: {pred_raw_val:.6f}")
print(f"Expected y_log: {pred_raw_val + bundle.b0_offset:.6f}")
print(f"Delta: {abs(y_log_raw.X - pred_raw_val):.6f}")

# Now check which leaves were selected
print("\n" + "=" * 70)
print("Tree selection analysis:")
print("=" * 70)

bst = bundle.reg.get_booster()
dumps = bst.get_dump(with_stats=False, dump_format="json")

# Extract which tree leaves were selected
selected_leaves_sum = 0.0
selected_leaves_count = 0

for t_idx, js in enumerate(dumps[:914]):  # all trees
    node = json.loads(js)
    
    # Find all leaves
    leaves = []
    def walk(nd):
        if "leaf" in nd:
            leaves.append(float(nd["leaf"]))
            return
        for ch in nd.get("children", []):
            walk(ch)
    walk(node)
    
    # Find which z variable is selected
    for k in range(len(leaves)):
        z_var_name = f"t{t_idx}_leaf{k}"
        try:
            z_var = m.getVarByName(z_var_name)
            if z_var is not None and z_var.X > 0.5:  # binary, so > 0.5 means selected
                print(f"Tree {t_idx}: Selected leaf {k}, value={leaves[k]:.6f}")
                selected_leaves_sum += leaves[k]
                selected_leaves_count += 1
                break
        except:
            pass

print(f"\nTotal selected leaves: {selected_leaves_count}")
print(f"Sum of selected leaves: {selected_leaves_sum:.6f}")
pred_raw_val = float(pred_raw.iloc[0])
print(f"Expected sum: {pred_raw_val:.6f}")
print(f"Delta: {abs(selected_leaves_sum - pred_raw_val):.6f}")
