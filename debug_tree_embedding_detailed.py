#!/usr/bin/env python3
"""
Debug script to trace tree embedding and leaf selection in detail.
Shows exactly which leaves are selected at each step and why.
"""
import json
import math
import sys
sys.path.insert(0, str(__file__.rsplit('\\', 1)[0]))

import numpy as np
import pandas as pd
from optimization.remodel.config import PATHS
from optimization.remodel.xgb_predictor import XGBBundle

def test_tree_embedding_detailed():
    """
    Load the XGBBundle and test tree embedding with detailed logging.
    """
    print("\n" + "="*80)
    print("TREE EMBEDDING DETAILED DEBUG")
    print("="*80)
    
    # Load the bundle
    print("\n[LOAD] Loading XGBBundle...")
    bundle = XGBBundle()
    print(f"[LOAD] Loaded successfully. Trees: {bundle.n_trees_use}")
    
    # Get a sample house
    print("\n[DATA] Loading base data...")
    base_df = pd.read_csv(str(PATHS.base_csv))
    # Property 528328100
    pid = 528328100
    house_data = base_df[base_df['PID'] == pid]
    
    if len(house_data) == 0:
        print(f"[ERROR] Property {pid} not found")
        return
    
    house_data = house_data.iloc[0]
    print(f"[DATA] Found property {pid}")
    
    # Get X features - use all columns except PID and non-feature cols
    feature_cols = [col for col in base_df.columns if col != 'PID']
    X_array = house_data[feature_cols].values.reshape(1, -1)
    print(f"[DATA] Features shape: {X_array.shape}")
    print(f"[DATA] Feature columns: {len(feature_cols)}")
    
    # Now create a modified X where we change some features (simulating MIP optimization)
    X_modified = X_array.copy()
    
    # Simulate feature changes (these would come from MIP optimization)
    # Since we don't have feature names easily available, we'll just modify a few indices
    print("\n[CHANGES] Modified features (simulating MIP optimization):")
    modifications = {
        5: (X_array[0, 5], X_array[0, 5] + 0.1),   # Small numeric change
        10: (X_array[0, 10], X_array[0, 10] + 0.2), # Another numeric change
    }
    
    for feat_idx, (old_val, new_val) in modifications.items():
        if feat_idx < len(feature_cols):
            X_modified[0, feat_idx] = new_val
            print(f"  - Feature {feat_idx} ({feature_cols[feat_idx]}): {old_val:.6f} -> {new_val:.6f}")
    
    
    # Get predictions (external predictor)
    print("\n[EXTERNAL] External predictor results:")
    y_pred_original = float(bundle.predict(X_array)[0])
    y_log_original = float(bundle.predict_log_raw(X_array)[0])
    print(f"  Base case: y_pred={y_pred_original:.6f}, y_log_raw={y_log_original:.6f}")
    
    y_pred_modified = float(bundle.predict(X_modified)[0])
    y_log_modified = float(bundle.predict_log_raw(X_modified)[0])
    print(f"  Modified case: y_pred={y_pred_modified:.6f}, y_log_raw={y_log_modified:.6f}")
    print(f"  Δ y_log_raw = {y_log_modified - y_log_original:.6f}")
    
    # Now manually evaluate trees with debug output
    print("\n[TREE-EVAL] Manual tree traversal (external predictor logic):")
    manual_eval_base = evaluate_trees_manual(bundle, X_array, "BASE")
    manual_eval_modified = evaluate_trees_manual(bundle, X_modified, "MODIFIED")
    
    print(f"\n[TREE-EVAL] Summary:")
    print(f"  Base case sum_leaves: {manual_eval_base['sum_leaves']:.6f}")
    print(f"  Modified case sum_leaves: {manual_eval_modified['sum_leaves']:.6f}")
    print(f"  External y_log_raw base: {y_log_original:.6f}")
    print(f"  External y_log_raw modified: {y_log_modified:.6f}")

def evaluate_trees_manual(bundle, X, label):
    """
    Manually walk through trees and collect leaf selections.
    """
    print(f"\n[TRAVERSE-{label}] Walking trees with feature values:")
    
    try:
        bst = bundle.reg.get_booster()
    except Exception:
        bst = getattr(bundle.reg, "_Booster", None)
    
    dumps = bst.get_dump(with_stats=False, dump_format="json")
    n_use = int(bundle.n_trees_use) if bundle.n_trees_use is not None else len(dumps)
    if 0 < n_use < len(dumps):
        dumps = dumps[:n_use]
    
    total_leaves = 0.0
    tree_results = []
    
    for t_idx, js in enumerate(dumps[:5]):  # Show first 5 trees in detail
        node = json.loads(js)
        
        # Collect all leaves first
        leaves = []
        def walk_collect(nd, path):
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
                walk_collect(yes_child, path + [(f_idx, thr, True)])
            if no_child is not None:
                walk_collect(no_child, path + [(f_idx, thr, False)])
        
        walk_collect(node, [])
        
        # Find which leaf path matches this X
        selected_leaf_idx = None
        selected_leaf_value = None
        selected_leaf_path = None
        
        for k, (conds, leaf_val) in enumerate(leaves):
            all_match = True
            path_str = ""
            for (f_idx, thr, is_left) in conds:
                x_val = float(X[0, f_idx])
                
                if is_left:
                    if not (x_val < thr - 1e-9):
                        all_match = False
                        path_str += f"f{f_idx}={x_val:.4f} >= {thr:.4f}[NO] "
                        break
                    else:
                        path_str += f"f{f_idx}={x_val:.4f} < {thr:.4f}[YES] "
                else:
                    if not (x_val >= thr - 1e-9):
                        all_match = False
                        path_str += f"f{f_idx}={x_val:.4f} < {thr:.4f}[NO] "
                        break
                    else:
                        path_str += f"f{f_idx}={x_val:.4f} >= {thr:.4f}[YES] "
            
            if all_match:
                selected_leaf_idx = k
                selected_leaf_value = leaf_val
                selected_leaf_path = path_str
                break
        
        if selected_leaf_idx is not None:
            total_leaves += selected_leaf_value
            tree_results.append({
                'tree_idx': t_idx,
                'leaf_idx': selected_leaf_idx,
                'leaf_value': selected_leaf_value,
                'n_leaves': len(leaves),
                'path': selected_leaf_path
            })
            print(f"  Tree {t_idx}: selected leaf {selected_leaf_idx}/{len(leaves)-1}, value={selected_leaf_value:.6f}")
            print(f"    Path: {selected_leaf_path}")
    
    # Continue for remaining trees without detailed output
    for t_idx, js in enumerate(dumps[5:], start=5):
        node = json.loads(js)
        
        leaves = []
        def walk_collect(nd, path):
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
                walk_collect(yes_child, path + [(f_idx, thr, True)])
            if no_child is not None:
                walk_collect(no_child, path + [(f_idx, thr, False)])
        
        walk_collect(node, [])
        
        for k, (conds, leaf_val) in enumerate(leaves):
            all_match = True
            for (f_idx, thr, is_left) in conds:
                x_val = float(X[0, f_idx])
                if is_left and not (x_val < thr - 1e-9):
                    all_match = False
                    break
                elif not is_left and not (x_val >= thr - 1e-9):
                    all_match = False
                    break
            
            if all_match:
                total_leaves += leaf_val
                tree_results.append({
                    'tree_idx': t_idx,
                    'leaf_idx': k,
                    'leaf_value': leaf_val,
                    'n_leaves': len(leaves),
                    'path': '...'
                })
                break
    
    b0 = bundle.b0_offset if bundle.b0_offset is not None else 0.0
    y_log_raw_calc = total_leaves + b0
    
    print(f"  [SUMMARY-{label}] Total leaf values: {total_leaves:.6f}")
    print(f"  [SUMMARY-{label}] Base score offset: {b0:.6f}")
    print(f"  [SUMMARY-{label}] Calculated y_log_raw: {y_log_raw_calc:.6f}")
    
    return {
        'sum_leaves': total_leaves,
        'y_log_raw': y_log_raw_calc,
        'tree_results': tree_results
    }

if __name__ == "__main__":
    test_tree_embedding_detailed()
