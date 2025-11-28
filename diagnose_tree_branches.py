#!/usr/bin/env python
"""
Diagnosticar dónde los árboles elijen branches distintas entre XGB y Gurobi.
"""

import json
import numpy as np
import pandas as pd
from optimization.remodel.xgb_predictor import XGBBundle

def main():
    bundle = XGBBundle()
    bst = bundle.reg.get_booster()
    dumps = bst.get_dump(with_stats=False, dump_format="json")
    
    # Buscar árboles que splitean en features binarias (bounds 0-1)
    binary_splits = []
    for t_idx, js in enumerate(dumps[:10]):  # Check first 10 trees
        node = json.loads(js)
        
        def walk(nd, depth=0):
            if "leaf" in nd:
                return
            f_idx = int(str(nd["split"]).replace("f", ""))
            thr = float(nd["split_condition"])
            
            # Check if this could be a binary feature
            # Binary features typically have splits at 0.5 or integers
            if 0.0 <= thr <= 1.0:
                binary_splits.append((t_idx, f_idx, thr))
                print(f"[TREE {t_idx}] Feature {f_idx}: split_condition={thr}")
            
            for ch in nd.get("children", []):
                walk(ch, depth+1)
        
        walk(node)
    
    print(f"\nEncontrados {len(binary_splits)} splits en rango [0,1]")
    
    # Ahora verifica: cuando el código Gurobi overwrite thr=0.5,
    # ¿es correcto?
    print("\nPosibles problemas:")
    for t_idx, f_idx, thr in binary_splits:
        if thr != 0.5:
            print(f"  ⚠ TREE {t_idx}, FEATURE {f_idx}: actual_thr={thr}, pero Gurobi usaría 0.5 → MISMATCH")

if __name__ == "__main__":
    main()
