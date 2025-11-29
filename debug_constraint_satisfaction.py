#!/usr/bin/env python3
"""
Check if the tree constraints are satisfying the split conditions correctly
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
X_array = X_input.values[0]

# Build MIP
m = gp.Model("test")
m.setParam("OutputFlag", 0)

x_vars = []
for feat_name, base_val in zip(X_input.columns, X_input.values[0]):
    xv = m.addVar(lb=base_val, ub=base_val, name=feat_name)
    x_vars.append(xv)

y_log_raw = m.addVar(lb=-100, ub=100, name="y_log_raw")

# Attach embedding
bundle.attach_to_gurobi(m, x_vars, y_log_raw, eps=-1e-6)

# Add objective
m.setObjective(y_log_raw, gp.GRB.MAXIMIZE)

# Optimize
m.optimize()

print("=" * 70)
print("CHECKING TREE CONSTRAINT SATISFACTION")
print("=" * 70)

bst = bundle.reg.get_booster()
dumps = bst.get_dump(with_stats=False, dump_format="json")

# For first tree, check if selected leaf's conditions are satisfied
tree0_json = json.loads(dumps[0])

# Find which leaf was selected
for k in range(10):  # up to 10 leaves
    z_var_name = f"t0_leaf{k}"
    try:
        z_var = m.getVarByName(z_var_name)
        if z_var is not None and z_var.X > 0.5:
            print(f"\nTree 0: Selected leaf {k}")
            
            # Manually traverse to get the path to this leaf
            leaves = []
            def walk(nd, path):
                if "leaf" in nd:
                    leaves.append((path, float(nd["leaf"])))
                    return
                f_idx = int(str(nd["split"]).replace("f", ""))
                thr = float(nd["split_condition"])
                yes_id = nd.get("yes")
                no_id = nd.get("no")
                
                yes_child = None
                no_child = None
                for ch in nd.get("children", []):
                    ch_id = ch.get("nodeid")
                    if ch_id == yes_id:
                        yes_child = ch
                    elif ch_id == no_id:
                        no_child = ch
                
                if yes_child is not None:
                    walk(yes_child, path + [(f_idx, thr, True)])
                if no_child is not None:
                    walk(no_child, path + [(f_idx, thr, False)])
            
            walk(tree0_json, [])
            
            if k < len(leaves):
                conds, leaf_val = leaves[k]
                print(f"  Leaf value: {leaf_val:.6f}")
                print(f"  Path conditions:")
                for f_idx, thr, is_left in conds:
                    x_name = X_input.columns[f_idx]
                    x_val = X_array[f_idx]
                    x_sol = x_vars[f_idx].X
                    condition_type = "<" if is_left else ">="
                    
                    # Check if condition is satisfied
                    if is_left:
                        satisfied = x_sol <= thr
                    else:
                        satisfied = x_sol >= thr
                    
                    status = "✓" if satisfied else "✗"
                    print(f"    {status} f{f_idx} ({x_name}): {x_sol:.2f} {condition_type} {thr:.2f}")
            break
    except:
        pass

print(f"\ny_log_raw.X = {y_log_raw.X:.6f}")
print(f"Expected: -0.048855 (sum of tree leaves)")
