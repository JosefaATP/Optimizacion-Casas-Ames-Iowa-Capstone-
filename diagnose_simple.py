#!/usr/bin/env python3
"""
Script simple para investigar el mismatch de predicciones.
Enfoque: Verificar si los valores de hojas se extraen correctamente.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

WORKSPACE = Path(__file__).parent
sys.path.insert(0, str(WORKSPACE))

from optimization.remodel.xgb_predictor import XGBBundle
from optimization.remodel.config import PATHS

# Load bundle using XGBBundle()
bundle = XGBBundle()
print(f"[INFO] Bundle loaded")

# Get booster
bst = bundle.reg.get_booster()
reg = bundle.reg

# Get tree dumps
dumps = bst.get_dump(with_stats=False, dump_format="json")
print(f"[INFO] Total trees: {len(dumps)}")

# Calculate base_score
b0_offset = getattr(bundle, 'b0_offset', None)
print(f"[INFO] b0_offset stored: {b0_offset}")

# Load data
train_df = pd.read_csv(PATHS.base_csv)
print(f"[INFO] Data shape: {train_df.shape}")

# Test on specific property
pid = 528328100
if pid in train_df.index:
    X_prop = train_df.loc[[pid]].values
    print(f"\n[TEST] Property {pid}")
    print(f"[TEST] X shape: {X_prop.shape}")
    
    # Prediction with output margin
    y_margin = reg.predict(X_prop, output_margin=True)[0]
    print(f"[TEST] y_margin (with base_score): {y_margin:.8f}")
    
    # Calculate base_score by checking zero vector
    zeros = np.zeros((1, X_prop.shape[1]))
    y_margin_zero = reg.predict(zeros, output_margin=True)[0]
    print(f"[TEST] y_margin at zero point: {y_margin_zero:.8f}")
    
    # Manually walk trees to get leaf values
    print(f"\n[DETAIL] Walking first 5 trees manually:")
    tree_sum = 0.0
    for t_idx in range(min(5, len(dumps))):
        js = dumps[t_idx]
        node = json.loads(js)
        
        # Walk to leaf
        cur = node
        depth = 0
        leaf_val = None
        while True:
            if "leaf" in cur:
                leaf_val = float(cur["leaf"])
                break
            
            f_idx = int(str(cur.get("split", "0")).replace("f", ""))
            thr = float(cur.get("split_condition", 0.0))
            x_val = X_prop[0, f_idx]
            go_yes = (x_val < thr)
            
            print(f"  Tree {t_idx}, depth {depth}: feature f{f_idx}, value={x_val:.6f}, threshold={thr:.6f}, go_yes={go_yes}")
            
            # Find next node
            yes_id = cur.get("yes")
            no_id = cur.get("no")
            next_id = yes_id if go_yes else no_id
            
            # Find in children
            nxt = None
            for ch in cur.get("children", []):
                if ch.get("nodeid") == next_id:
                    nxt = ch
                    break
            
            if nxt is None:
                print(f"  [WARNING] Could not find next node!")
                break
            
            cur = nxt
            depth += 1
        
        if leaf_val is not None:
            print(f"  -> Tree {t_idx} leaf value: {leaf_val:.8f}")
            tree_sum += leaf_val
        print()
    
    print(f"[SUMMARY] Sum of first 5 tree leaves: {tree_sum:.8f}")
    print(f"[SUMMARY] y_margin - tree_sum = {y_margin - tree_sum:.8f} (should be base_score)")

