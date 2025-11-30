#!/usr/bin/env python3
"""
Debug script to understand the y_log mismatch in path-based embedding.
"""

import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from optimization.remodel.xgb_predictor import XGBBundle

# Get the XGBoost model
bundle = XGBBundle()

print("="*70)
print("Checking path-based embedding implementation")
print("="*70)

# Get the tree dumps
bst = bundle.reg.get_booster()
dumps = bst.get_dump(with_stats=False, dump_format="json")

print(f"Number of trees: {len(dumps)}")
print(f"Base score offset: {bundle.b0_offset}")

# Check first tree
node = json.loads(dumps[0])

# Extract paths
paths = []
def walk(nd, path):
    if "leaf" in nd:
        paths.append((path, float(nd["leaf"])))
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

print(f"\nTree 0: {len(paths)} paths")
for path_idx, (conditions, leaf_value) in enumerate(paths):
    print(f"  Path {path_idx}: leaf_value = {leaf_value:.6f}")

print("\n" + "="*70)
print("Checking implementation in xgb_predictor.py...")
print("="*70)

# Check if our changes are being used
with open('optimization/remodel/xgb_predictor.py', 'r') as f:
    content = f.read()
    if 'p[k] * paths[k][1]' in content:
        print("[OK] Path-based formulation FOUND in xgb_predictor.py")
    else:
        print("[FAIL] Path-based formulation NOT FOUND")
    
    if 'total_expr += gp.quicksum(p[k] * paths[k][1]' in content:
        print("[OK] Correct leaf summation formula found")
    else:
        print("[FAIL] Leaf summation formula issue")
