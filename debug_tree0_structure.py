#!/usr/bin/env python3
"""
Directly extract all leaves and their paths from tree0
"""
import json
from optimization.remodel.io import get_base_house
from optimization.remodel.xgb_predictor import XGBBundle

pid = 526351010
base_house = get_base_house(pid)
bundle = XGBBundle()

bst = bundle.reg.get_booster()
dumps = bst.get_dump(with_stats=False, dump_format="json")

tree0_json = json.loads(dumps[0])

print("=" * 70)
print("TREE 0 STRUCTURE AND LEAVES")
print("=" * 70)

leaves = []

def walk(nd, path=""):
    if "leaf" in nd:
        leaves.append((path, float(nd["leaf"])))
        return
    
    node_id = nd.get("nodeid")
    f_idx_str = nd["split"]
    f_idx = int(f_idx_str.replace("f", ""))
    thr = float(nd["split_condition"])
    yes_id = nd.get("yes")
    no_id = nd.get("no")
    
    # Find children
    yes_child = None
    no_child = None
    for ch in nd.get("children", []):
        if ch.get("nodeid") == yes_id:
            yes_child = ch
        elif ch.get("nodeid") == no_id:
            no_child = ch
    
    if yes_child is not None:
        walk(yes_child, path + f"[f{f_idx}<{thr:.1f}]")
    if no_child is not None:
        walk(no_child, path + f"[f{f_idx}>={thr:.1f}]")

walk(tree0_json)

print(f"\nFound {len(leaves)} leaves:")
for k, (path, val) in enumerate(leaves):
    print(f"  Leaf {k}: {val:.6f}, path={path}")
