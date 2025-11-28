#!/usr/bin/env python3
"""
Test to verify the constraint flow for y_log_raw -> y_log -> y_price
"""
import sys
sys.path.insert(0, r'c:\Users\valen\OneDrive - uc.cl\UNIVERSIDAD\2024-1\Bases de datos\Optimizacion-Casas-Ames-Iowa-Capstone-\optimization\remodel')

import numpy as np
import pandas as pd
from xgb_predictor import XGBBundle
import pickle
import json

# Load the trained model
with open("../models/xgb/completa_present_log_p2_1800_ELEGIDO/xgb_model.pkl", "rb") as f:
    bundle = pickle.load(f)

# Get base_score from booster
try:
    bst = bundle.reg.get_booster()
except:
    bst = getattr(bundle.reg, "_Booster", None)

bs_attr = bst.attr("base_score")
print(f"base_score from booster.attr: {bs_attr}")
print(f"b0_offset attribute on bundle: {bundle.b0_offset}")

# Now verify by computing it manually
zeros = np.zeros((1, len(bst.feature_names)))
y_out_margin = float(bundle.reg.predict(zeros, output_margin=True)[0])
y_in_sum = bundle._eval_sum_leaves(zeros.ravel())
b0_computed = y_out_margin - y_in_sum

print(f"\nManual computation at origin:")
print(f"  predict(zeros, output_margin=True) = {y_out_margin}")
print(f"  _eval_sum_leaves(zeros) = {y_in_sum}")
print(f"  b0 = {y_out_margin} - {y_in_sum} = {b0_computed}")

# Now let's check if attach_to_gurobi updates b0_offset
print(f"\nBefore attach_to_gurobi: b0_offset = {bundle.b0_offset}")

import gurobipy as gp
m = gp.Model("test")
x_vars = [m.addVar(name=f"x_{i}") for i in range(len(bst.feature_names))]

# This should update b0_offset if it's None or near 0
bundle.attach_to_gurobi(m, x_vars, m.addVar(name="y_log_raw"), eps=0.0)

print(f"After attach_to_gurobi: b0_offset = {bundle.b0_offset}")

# Check if the constraint was added
for c in m.getConstrs():
    if "YLOG" in c.ConstrName:
        print(f"\nFound constraint: {c.ConstrName}")
        print(f"  Sense: {c.Sense}")
        print(f"  RHS: {c.RHS}")
