#!/usr/bin/env python3
"""
Investiga dónde viene la diferencia en y_log entre MIP y predictor externo.
"""
import sys
sys.path.insert(0, r"c:\Users\valen\OneDrive - uc.cl\UNIVERSIDAD\2024-1\Bases de datos\Optimizacion-Casas-Ames-Iowa-Capstone-")

from optimization.remodel import gurobi_model, config, costs
from optimization.remodel.xgb_predictor import XGBBundle
from optimization.remodel.io import get_base_house
import pandas as pd
import numpy as np

pid = 526351010
base_house = get_base_house(pid)
base_row = base_house.row if hasattr(base_house, 'row') else base_house

# Load bundle
bundle = XGBBundle()

# Get base price
X_base = gurobi_model.build_base_input_row(bundle, base_row)

print("=" * 80)
print("COMPARING y_log CALCULATIONS")
print("=" * 80)
print()

# 1. Get the external predictions
y_pred = float(bundle.predict(X_base).iloc[0])
y_log_raw_external = float(bundle.predict_log_raw(X_base).iloc[0])
y_log_pipeline = float(bundle.pipe_for_gurobi().predict(X_base)[0])

print(f"External predictions for base X:")
print(f"  predict() [final price]        = {y_pred:,.2f}")
print(f"  predict_log_raw() [margin]     = {y_log_raw_external:.6f}")
print(f"  pipe_for_gurobi().predict()    = {y_log_pipeline:.6f}")
print()

# 2. Understand what's in the booster
try:
    bst = bundle.reg.get_booster()
except Exception:
    bst = getattr(bundle.reg, "_Booster", None)

if bst is not None:
    print("Booster information:")
    print(f"  Number of trees: {len(bst.get_dump())}")
    
    # Check base_score
    try:
        bs_attr = bst.attr("base_score")
        print(f"  base_score attribute: {bs_attr}")
    except Exception:
        print(f"  base_score attribute: NOT FOUND")
    
    # Try raw evaluation
    print()
    print("Raw booster evaluation on X_base:")
    Xp = bundle.pre.transform(X_base)
    
    # Call predict with output_margin=True (raw sum of leaves + base_score)
    y_raw_margin = float(bundle.reg.predict(Xp, output_margin=True)[0])
    print(f"  reg.predict(output_margin=True) = {y_raw_margin:.6f}")
    
    # Try to evaluate just the tree sum (without base_score)
    try:
        from xgboost import DMatrix
        dmat = DMatrix(Xp)
        y_tree_sum = float(bst.predict(dmat)[0])  # This includes base_score
        print(f"  bst.predict(DMatrix) = {y_tree_sum:.6f}")
    except Exception as e:
        print(f"  bst.predict(DMatrix) failed: {e}")
    
    # Check what the predictor returns directly
    print(f"  reg.predict() [final price]    = {float(bundle.reg.predict(Xp)[0]):,.2f}")
    
print()
print("=" * 80)
print("DIFFERENCE ANALYSIS")
print("=" * 80)
print(f"Difference between:")
print(f"  predict_log_raw() - predict_log_raw() [self-check]  = {y_log_raw_external - y_log_raw_external:.6f} ✓")
print(f"  pipe_for_gurobi() - predict_log_raw()              = {y_log_pipeline - y_log_raw_external:.6f}")
print()

# The key insight: what should y_log be?
# According to the code, y_log should be y_log_raw + b0_offset
# So y_log_raw is the "margin" (raw tree output)
# And y_log = y_log_raw + b0_offset

print("Expected scaling:")
print(f"  If y_log_raw_external = {y_log_raw_external:.6f}")
print(f"  And y_log = y_log_raw + b0")
print(f"  Then b0 = y_log - y_log_raw")
print()

b0_from_pipeline = y_log_pipeline - y_log_raw_external
print(f"  b0 (inferred from pipeline) = {b0_from_pipeline:.6f}")
