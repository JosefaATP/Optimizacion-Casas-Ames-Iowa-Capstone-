#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch check that scans the first N rows of the base CSV and compares:
 - base numeric row (build_base_input_row)
 - reconstructed numeric row (simulating fix_to_base=True but WITHOUT Gurobi)
Then computes predictions for both and writes results to CSV.
"""
import pandas as pd
import numpy as np
from optimization.remodel.xgb_predictor import XGBBundle
from optimization.remodel.gurobi_model import build_base_input_row
from optimization.remodel.config import PATHS
from optimization.remodel.features import MODIFIABLE

OUT = "batch_encoding_check.csv"
N = 200


def main(n=N, out=OUT):
    bundle = XGBBundle(PATHS.xgb_model_file)
    df = pd.read_csv(PATHS.base_csv, low_memory=False)
    rows = min(n, len(df))

    records = []
    feature_order = list(bundle.feature_names_in())

    for i in range(rows):
        row = df.iloc[i]
        base_X = build_base_input_row(bundle, row)

        # reconstruct: use base_X numeric for all features (simulating fix_to_base)
        row_vals = {}
        for fname in feature_order:
            try:
                row_vals[fname] = float(base_X.iloc[0].loc[fname])
            except Exception:
                row_vals[fname] = np.nan
        X_recon = pd.DataFrame([row_vals], columns=feature_order, dtype=float)

        # compare columnwise
        diffs = []
        for c in feature_order:
            v_base = base_X.iloc[0].loc[c]
            v_rec = X_recon.iloc[0].loc[c]
            if not (np.isfinite(v_base) and np.isfinite(v_rec)):
                if not (pd.isna(v_base) and pd.isna(v_rec)):
                    diffs.append((c, v_base, v_rec))
            else:
                if abs(float(v_base) - float(v_rec)) > 1e-8:
                    diffs.append((c, v_base, v_rec))

        pred_base = float(bundle.predict(base_X).iloc[0])
        pred_recon = float(bundle.predict(X_recon).iloc[0])

        rec = {
            'index': i,
            'pred_base': pred_base,
            'pred_recon': pred_recon,
            'equal': abs(pred_base - pred_recon) < 1e-8,
            'n_diffs': len(diffs),
            'sample_diffs': ';'.join([f"{d[0]}:{d[1]}->{d[2]}" for d in diffs[:5]])
        }
        records.append(rec)

    out_df = pd.DataFrame(records)
    out_df.to_csv(out, index=False)
    print(f'Wrote {out} with {len(records)} rows; mismatches={len(out_df[~out_df.equal])}')


if __name__ == '__main__':
    main()
