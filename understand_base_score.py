#!/usr/bin/env python
"""
Understand the base_score in XGBoost tree embedding.
"""

import numpy as np
from optimization.remodel.xgb_predictor import XGBBundle
import pandas as pd

def main():
    b = XGBBundle()
    
    # Get the booster
    bst = b.reg.get_booster()
    bs_attr = bst.attr("base_score")
    print(f"Booster base_score attr: {bs_attr}")
    
    # Test data: all zeros
    feats = b.feature_names_in()
    X_zeros = pd.DataFrame(np.zeros((1, len(feats))), columns=feats)
    
    # Get preprocessed data
    X_zeros_pre = b.pre.transform(X_zeros)
    
    # Method 1: XGB predict with output_margin=True
    y_margin_xgb = float(b.reg.predict(X_zeros_pre, output_margin=True)[0])
    print(f"\nXGB predict(output_margin=True): {y_margin_xgb:.6f}")
    
    # Method 2: XGB predict without output_margin (includes TTR expm1)
    y_price_xgb = float(b.reg.predict(X_zeros_pre)[0])
    print(f"XGB predict(): {y_price_xgb:.6f}")
    print(f"  (This includes TTR, so it's in price scale)")
    
    # Method 3: pipe_for_gurobi (XGBRegressor only, no TTR)
    y_pipe = float(b.pipe_for_gurobi().predict(X_zeros)[0])
    print(f"\npipe_for_gurobi().predict(): {y_pipe:.6f}")
    
    # Method 4: Manually sum tree leaves
    import json
    dumps = bst.get_dump(with_stats=False, dump_format="json")
    sum_leaves = 0.0
    for js in dumps:
        node = json.loads(js)
        cur = node
        while "leaf" in cur:
            sum_leaves += float(cur.get("leaf", 0.0))
            break
        # If root has no leaf, traverse to find it
        if "leaf" not in cur:
            cur = node
            while "children" in cur and cur["children"]:
                cur = cur["children"][0]  # Go left by default
            if "leaf" in cur:
                sum_leaves += float(cur.get("leaf", 0.0))
    
    print(f"\nManual sum of tree leaves: {sum_leaves:.6f}")
    print(f"Δ(y_margin_xgb - sum_leaves): {y_margin_xgb - sum_leaves:+.6f}")
    print(f"This should equal base_score: {float(bs_attr):.6f}")

if __name__ == "__main__":
    main()
