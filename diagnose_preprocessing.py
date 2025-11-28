#!/usr/bin/env python3
"""
Diagnostic: Verify if features need preprocessing before tree embedding.
"""

import sys
import pandas as pd
import numpy as np
sys.path.insert(0, r"c:\Users\valen\OneDrive - uc.cl\UNIVERSIDAD\2024-1\Bases de datos\Optimizacion-Casas-Ames-Iowa-Capstone-")

from optimization.remodel.xgb_predictor import XGBBundle

# Load bundle
try:
    bundle = XGBBundle()
except Exception as e:
    print(f"[PREPROC-DIAG] Error loading bundle: {e}")
    sys.exit(1)

print("[PREPROC-DIAG] XGBBundle preprocessing check")
print(f"[PREPROC-DIAG] bundle.pre type: {type(bundle.pre)}")

orig_feats = bundle.feature_names_in()
booster_feats = bundle.booster_feature_order()
print(f"[PREPROC-DIAG] Original features: {len(orig_feats)} features")
print(f"[PREPROC-DIAG]   First 5: {orig_feats[:5]}")
print(f"[PREPROC-DIAG] Booster features: {len(booster_feats)} features")
print(f"[PREPROC-DIAG]   First 5: {booster_feats[:5]}")

# Check if they're the same
if orig_feats == booster_feats:
    print("[PREPROC-DIAG] ✓ Original features == Booster features (NO TRANSFORMATION)")
else:
    print("[PREPROC-DIAG] ⚠️ Original features != Booster features")
    print("[PREPROC-DIAG] This suggests features ARE transformed!")
    
# Check the ColumnTransformer
try:
    ct = bundle.pre
    trs = ct.transformers_ if hasattr(ct, "transformers_") else ct.transformers
    print(f"\n[PREPROC-DIAG] ColumnTransformer has {len(trs)} transformers:")
    for name, transformer, cols in trs:
        print(f"  - {name}: {transformer.__class__.__name__} on {len(list(cols))} features")
        if name == "num":
            num_cols = list(cols)
            print(f"    Numeric features: {num_cols[:5]}...")
except Exception as e:
    print(f"[PREPROC-DIAG] Could not inspect ColumnTransformer: {e}")

print("\n[PREPROC-DIAG] Conclusion:")
if orig_feats == booster_feats:
    print("[PREPROC-DIAG] ✓ No transformation detected → x_vars should be in correct space")
else:
    print("[PREPROC-DIAG] ⚠️ Features ARE transformed!")
    print("[PREPROC-DIAG] The MIP x_vars are in ORIGINAL space but tree thresholds expect TRANSFORMED space!")
    print("[PREPROC-DIAG] This is likely the BUG causing the 0.387 divergence!")
