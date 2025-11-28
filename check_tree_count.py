#!/usr/bin/env python3
"""
Diagnóstico: verificar número de árboles usados por Gurobi vs XGBoost
"""
import sys
import os
os.chdir('optimization/remodel')
sys.path.insert(0, '.')

import pandas as pd
import pickle
from xgb_predictor import XGBBundle

# Load bundle
bundle = pickle.load(open("../../models/xgb/completa_present_log_p2_1800_ELEGIDO/xgb_model.pkl", "rb"))

# Check meta
import json
meta = json.load(open("../../models/xgb/completa_present_log_p2_1800_ELEGIDO/meta.json"))
print("Meta información:")
print(f"  best_iteration: {meta.get('best_iteration')}")
print(f"  best_ntree_limit: {meta.get('best_ntree_limit')}")
print(f"  best_score: {meta.get('best_score')}")

# Check booster
try:
    bst = bundle.reg.get_booster()
except:
    bst = getattr(bundle.reg, "_Booster", None)

if bst:
    dumps = bst.get_dump(with_stats=False, dump_format="json")
    print(f"\nTotal trees in booster: {len(dumps)}")
    print(f"bundle.n_trees_use: {bundle.n_trees_use}")
    
    if bundle.n_trees_use:
        print(f"\nUsing FIRST {bundle.n_trees_use} trees in embed")
    else:
        print(f"\nUsing ALL {len(dumps)} trees in embed")

# Now check what happens during predictions
test_data = pd.read_csv("../../data/processed/base_completa_sin_nulos.csv").iloc[:1]
print(f"\n\nTesting predictions:")

try:
    pred_full = bundle.predict_log_raw(test_data).values[0]
    print(f"predict_log_raw (full): {pred_full:.6f}")
except Exception as e:
    print(f"predict_log_raw error: {e}")

try:
    # Try with iteration_range
    Xp = bundle.pre.transform(test_data)
    if bundle.n_trees_use:
        pred_limited = bundle.reg.predict(Xp, output_margin=True, iteration_range=(0, bundle.n_trees_use))[0]
    else:
        pred_limited = bundle.reg.predict(Xp, output_margin=True)[0]
    print(f"Direct predict (with n_trees_use={bundle.n_trees_use}): {pred_limited:.6f}")
except Exception as e:
    print(f"Direct predict error: {e}")
