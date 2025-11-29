#!/usr/bin/env python3
"""
Debug whether the hard-coded leaf selection is being used
"""
import json
import gurobipy as gp
import pandas as pd
from optimization.remodel.io import get_base_house
from optimization.remodel.xgb_predictor import XGBBundle
from optimization.remodel.gurobi_model import build_base_input_row

pid = 526351010
base_house = get_base_house(pid)
bundle = XGBBundle()

X_input = build_base_input_row(bundle, base_house.row)

m = gp.Model("test")
m.setParam("OutputFlag", 1)  # Enable output to see what's happening

x_vars = []
for feat_name, base_val in zip(X_input.columns, X_input.values[0]):
    xv = m.addVar(lb=base_val, ub=base_val, name=feat_name)
    x_vars.append(xv)

y_log_raw = m.addVar(lb=-100, ub=100, name="y_log_raw")

# Patch attach_to_gurobi to add debug output
original_attach = bundle.attach_to_gurobi

def debug_attach(m_arg, x_list, y_log, eps=-1e-6):
    print("\n" + "=" * 70)
    print("DEBUG: Inside attach_to_gurobi")
    print("=" * 70)
    
    import json, math
    
    bst = bundle.reg.get_booster()
    dumps = bst.get_dump(with_stats=False, dump_format="json")
    
    total_expr = gp.LinExpr(0.0)
    
    def fin(v):
        try:
            return math.isfinite(float(v))
        except Exception:
            return False
    
    for t_idx, js in enumerate(dumps[:3]):  # First 3 trees only
        node = json.loads(js)

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
        walk(node, [])

        z = [m_arg.addVar(vtype=gp.GRB.BINARY, name=f"t{t_idx}_leaf{k}") for k in range(len(leaves))]
        m_arg.addConstr(gp.quicksum(z) == 1, name=f"TREE_{t_idx}_ONEHOT")

        # Check if all_fixed
        tree_features = set()
        for k, (conds, _) in enumerate(leaves):
            for (f_idx, thr, is_left) in conds:
                if f_idx < len(x_list):
                    tree_features.add(f_idx)
        
        all_fixed = all(
            fin(getattr(x_list[f_idx], "LB", None)) and 
            fin(getattr(x_list[f_idx], "UB", None)) and
            float(x_list[f_idx].LB) == float(x_list[f_idx].UB)
            for f_idx in tree_features
        )
        
        print(f"\nTree {t_idx}: all_fixed = {all_fixed}, tree_features = {tree_features}")
        
        if all_fixed:
            print(f"  Using HARD-CODED leaf selection")
            # Find correct leaf
            for k, (conds, _) in enumerate(leaves):
                all_conds_satisfied = True
                for (f_idx, thr, is_left) in conds:
                    xv = x_list[f_idx]
                    x_val = float(xv.LB)
                    if is_left:
                        sat = (x_val < thr - 1e-9)
                    else:
                        sat = (x_val >= thr - 1e-9)
                    all_conds_satisfied = all_conds_satisfied and sat
                
                if all_conds_satisfied:
                    print(f"  Correct leaf: {k}")
                    m_arg.addConstr(z[k] == 1, name=f"TREE_{t_idx}_FORCE")
                    break
        else:
            print(f"  Using BIG-M constraints")

bundle.attach_to_gurobi = debug_attach

bundle.attach_to_gurobi(m, x_vars, y_log_raw, eps=-1e-6)
