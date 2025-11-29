#!/usr/bin/env python3
"""
Extract and print all leaf values from all trees to see their magnitudes
"""
import json
from optimization.remodel.io import get_base_house
from optimization.remodel.xgb_predictor import XGBBundle

pid = 526351010
base_house = get_base_house(pid)
bundle = XGBBundle()

bst = bundle.reg.get_booster()
dumps = bst.get_dump(with_stats=False, dump_format="json")

print(f"Total trees: {len(dumps)}")
print("=" * 70)

all_leaves = []
for t_idx, js in enumerate(dumps[:3]):  # First 3 trees
    node = json.loads(js)
    
    leaves = []
    def walk(nd):
        if "leaf" in nd:
            leaves.append(float(nd["leaf"]))
            return
        for ch in nd.get("children", []):
            walk(ch)
    
    walk(node)
    all_leaves.extend(leaves)
    
    print(f"Tree {t_idx}: {len(leaves)} leaves, values: {[f'{v:.6f}' for v in leaves[:5]]}")
    if len(leaves) > 5:
        print(f"             ... and {len(leaves) - 5} more")

print("=" * 70)
print(f"Total leaves in first 3 trees: {len(all_leaves)}")
print(f"Sum of all leaves (should be ~0 because raw model output): {sum(all_leaves):.6f}")
print(f"Mean leaf value: {sum(all_leaves) / len(all_leaves):.6f}")

# Ahora vamos a simular lo que hace attach_to_gurobi
print("\n" + "=" * 70)
print("Manual tree walk to check if leaves are found correctly")
print("=" * 70)

dumps_subset = dumps[:1]
total_expr_manual = 0.0

for t_idx, js in enumerate(dumps_subset):
    node = json.loads(js)
    
    leaves = []
    def walk(nd, path=""):
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
            walk(yes_child, path + f"[f{f_idx}<{thr}]")
        
        if no_child is not None:
            walk(no_child, path + f"[f{f_idx}>={thr}]")
    
    walk(node)
    
    print(f"\nTree {t_idx}:")
    for k, (path, leaf_val) in enumerate(leaves):
        print(f"  Leaf {k}: value={leaf_val:.6f}, path={path}")
        total_expr_manual += leaf_val
    
    # This is WRONG - we need to select only ONE leaf based on z[k]
    # But for debugging, what's the sum?
    print(f"  Sum if all leaves selected: {sum(lv for _, lv in leaves):.6f}")

print("\n" + "=" * 70)
print(f"Sum of all leaves in tree 0 (if all selected): {total_expr_manual:.6f}")
