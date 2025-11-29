#!/usr/bin/env python3
"""
Analyze the distribution of leaf values across all trees
"""
import json
from optimization.remodel.io import get_base_house
from optimization.remodel.xgb_predictor import XGBBundle

pid = 526351010
base_house = get_base_house(pid)
bundle = XGBBundle()

bst = bundle.reg.get_booster()
dumps = bst.get_dump(with_stats=False, dump_format="json")

positive_count = 0
negative_count = 0
positive_sum = 0.0
negative_sum = 0.0
all_leaves = []

for t_idx, js in enumerate(dumps):
    node = json.loads(js)
    
    def extract_leaves(nd):
        if "leaf" in nd:
            val = float(nd["leaf"])
            all_leaves.append(val)
            return
        for ch in nd.get("children", []):
            extract_leaves(ch)
    
    extract_leaves(node)

# Sort and display
all_leaves.sort()

print("=" * 70)
print("LEAF VALUE DISTRIBUTION")
print("=" * 70)
print(f"\nTotal leaves: {len(all_leaves)}")
print(f"Min: {all_leaves[0]:.6f}")
print(f"Max: {all_leaves[-1]:.6f}")
print(f"Mean: {sum(all_leaves) / len(all_leaves):.6f}")
print(f"Sum of all leaves (if all selected): {sum(all_leaves):.6f}")

positive = [x for x in all_leaves if x > 0]
negative = [x for x in all_leaves if x < 0]
zero = [x for x in all_leaves if x == 0]

print(f"\nPositive leaves: {len(positive)}")
print(f"Negative leaves: {len(negative)}")
print(f"Zero leaves: {len(zero)}")

print(f"\nSum of positive leaves: {sum(positive):.6f}")
print(f"Sum of negative leaves: {sum(negative):.6f}")

# Helper function
def extract_all_leaves(nd):
    if "leaf" in nd:
        return [float(nd["leaf"])]
    leaves = []
    for ch in nd.get("children", []):
        leaves.extend(extract_all_leaves(ch))
    return leaves

max_leaf_per_tree = []
for t_idx, js in enumerate(dumps):
    node = json.loads(js)
    leaves = extract_all_leaves(node)
    max_leaf_per_tree.append(max(leaves))

print(f"\nMax leaf value per tree (if always picking max): {sum(max_leaf_per_tree):.6f}")

# What about if we're trying to maximize y_log in the MIP?
# The MIP objective is maximize y_log_raw
# y_log_raw = sum of tree leaves
# So it should select leaves that make the sum as large as possible
# Which would be selecting the positive leaves where they exist

print("\n" + "=" * 70)
print("MIP OPTIMIZATION INSIGHT")
print("=" * 70)
print(f"\nIf MIP is trying to MAXIMIZE y_log_raw = sum of leaves,")
print(f"it should select the maximum leaf from each tree.")
print(f"That would give: {sum(max_leaf_per_tree):.6f}")
print(f"\nBut MIP is getting: 0.035883")
print(f"Which is close to: {sum(max_leaf_per_tree) / len(dumps):.6f} (average of max leaves)")
print(f"\nExpected (correct sum): -0.048855")
