#!/usr/bin/env python3
"""
CRITICAL: Check if tree feature indices f0,f1,f2... match booster_feature_order()
"""
import json
from optimization.remodel.io import get_base_house
from optimization.remodel.xgb_predictor import XGBBundle

pid = 526351010
base_house = get_base_house(pid)
bundle = XGBBundle()

bst = bundle.reg.get_booster()
dumps = bst.get_dump(with_stats=False, dump_format="json")

# Inspeccionar primer árbol
tree0 = json.loads(dumps[0])

booster_order = bundle.booster_feature_order()

print("=" * 70)
print("TREE 0 - Feature indices used")
print("=" * 70)

def find_feature_indices(nd, indices=set()):
    if "split" in nd:
        f_idx_str = nd["split"]  # e.g., "f42"
        f_idx = int(f_idx_str.replace("f", ""))
        indices.add(f_idx)
        if "children" in nd:
            for ch in nd["children"]:
                find_feature_indices(ch, indices)
    return indices

tree0_features = find_feature_indices(tree0)
print(f"Feature indices used in tree0: {sorted(tree0_features)[:10]}...")

for f_idx in sorted(tree0_features)[:5]:
    feature_name = booster_order[f_idx] if f_idx < len(booster_order) else f"UNKNOWN[{f_idx}]"
    print(f"  f{f_idx} -> {feature_name}")

print(f"\nTotal features in booster_order: {len(booster_order)}")
print(f"Total feature indices in tree0: {len(tree0_features)}")
print(f"Max index in tree0: {max(tree0_features)}")

# Ver si el máximo índice es válido
if max(tree0_features) >= len(booster_order):
    print(f"\n⚠️  WARNING: Max index {max(tree0_features)} >= len(booster_order) {len(booster_order)}")
