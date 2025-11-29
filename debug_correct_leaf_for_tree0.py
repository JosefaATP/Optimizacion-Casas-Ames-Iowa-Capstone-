#!/usr/bin/env python3
"""
Check which leaf SHOULD be selected for tree 0 based on base house features
"""
import pandas as pd
from optimization.remodel.io import get_base_house
from optimization.remodel.xgb_predictor import XGBBundle
from optimization.remodel.gurobi_model import build_base_input_row

pid = 526351010
base_house = get_base_house(pid)
bundle = XGBBundle()

X_input = build_base_input_row(bundle, base_house.row)
X_array = X_input.values[0]

booster_order = bundle.booster_feature_order()

# The feature indices from tree 0
feat_to_check = {
    5: "Year Built",      # From f5<1986, f5>=1986
    29: "Fireplaces",     # From f29<1, f29>=1
    6: "Year Remod/Add",  # From f6<1954, f6>=1954
    20: "Gr Liv Area",    # From f20<1740, f20>=1740, f20<1661, f20>=1661
    33: "Garage Area",    # From f33<650, f33>=650
}

print("=" * 70)
print("BASE HOUSE FEATURE VALUES FOR TREE 0 DECISIONS")
print("=" * 70)

for f_idx, expected_name in feat_to_check.items():
    actual_name = booster_order[f_idx]
    x_val = X_array[f_idx]
    print(f"\nf{f_idx}: {actual_name} (expected: {expected_name})")
    print(f"  Value: {x_val:.2f}")

# Now manually determine the correct leaf
print("\n" + "=" * 70)
print("TREE 0 PATH EVALUATION")
print("=" * 70)

f5_val = X_array[5]  # Year Built
f29_val = X_array[29]  # Fireplaces
f6_val = X_array[6]  # Year Remod/Add
f20_val = X_array[20]  # Gr Liv Area
f33_val = X_array[33]  # Garage Area

print(f"\nf5 ({f5_val:.2f}) < 1986? {f5_val < 1986} -> ", end="")
if f5_val < 1986:
    print("YES (go left to f29 branch)")
    print(f"  f29 ({f29_val:.2f}) < 1.0? {f29_val < 1.0} -> ", end="")
    if f29_val < 1.0:
        print("YES (go left to f6 branch)")
        print(f"    f6 ({f6_val:.2f}) < 1954? {f6_val < 1954} -> ", end="")
        if f6_val < 1954:
            print("YES -> LEAF 0 (-0.022715)")
        else:
            print("NO -> LEAF 1 (-0.010932)")
    else:
        print("NO (go right to f20 branch)")
        print(f"    f20 ({f20_val:.2f}) < 1740? {f20_val < 1740} -> ", end="")
        if f20_val < 1740:
            print("YES -> LEAF 2 (-0.004585)")
        else:
            print("NO -> LEAF 3 (0.010367)")
else:
    print("NO (go right to f33 branch)")
    print(f"  f33 ({f33_val:.2f}) < 650? {f33_val < 650} -> ", end="")
    if f33_val < 650:
        print("YES (go left to f20 branch)")
        print(f"    f20 ({f20_val:.2f}) < 1484? {f20_val < 1484} -> ", end="")
        if f20_val < 1484:
            print("YES -> LEAF 4 (0.002694)")
        else:
            print("NO -> LEAF 5 (0.012319)")
    else:
        print("NO (go right to f20 branch)")
        print(f"    f20 ({f20_val:.2f}) < 1661? {f20_val < 1661} -> ", end="")
        if f20_val < 1661:
            print("YES -> LEAF 6 (0.013785)")
        else:
            print("NO -> LEAF 7 (0.028053)")
