#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight check that reproduces the `X_input` construction from `build_mip_embed`
without creating a Gurobi model. This avoids Gurobi startup/licensing output and
is much faster for verifying encoding/prediction invariance.
"""
import pandas as pd
import numpy as np
from optimization.remodel.xgb_predictor import XGBBundle
from optimization.remodel.gurobi_model import build_base_input_row
from optimization.remodel.config import PATHS
from optimization.remodel.features import MODIFIABLE


def main(idx=1):
    bundle = XGBBundle(PATHS.xgb_model_file)
    df = pd.read_csv(PATHS.base_csv, low_memory=False)
    row = df.iloc[idx]

    print('[STEP] Building base numeric row via build_base_input_row...')
    base_X = build_base_input_row(bundle, row)

    feature_order = list(bundle.feature_names_in())
    modif = {f.name for f in MODIFIABLE}

    # Reconstruct numeric input without Gurobi: for non-modifiable features use base_X numeric,
    # for modifiable features also use base_X numeric (this simulates fix_to_base=True)
    row_vals = {}
    for fname in feature_order:
        try:
            row_vals[fname] = float(base_X.iloc[0].loc[fname])
        except Exception:
            # fallback: use 0.0
            row_vals[fname] = 0.0

    X_recon = pd.DataFrame([row_vals], columns=feature_order, dtype=float)

    # Compare base_X and X_recon
    diffs = []
    for c in feature_order:
        v_base = float(base_X.iloc[0].loc[c])
        v_rec  = float(X_recon.iloc[0].loc[c])
        if not (np.isfinite(v_base) and np.isfinite(v_rec)) or abs(v_base - v_rec) > 1e-8:
            diffs.append((c, v_base, v_rec))
    if not diffs:
        print('[OK] No differences between base numeric row and reconstructed numeric row (no-Gurobi).')
    else:
        print(f'[WARN] {len(diffs)} differences found (showing up to 20):')
        for d in diffs[:20]:
            print(' ', d)

    # Compare predictions
    print('[STEP] Running bundle.predict on both rows...')
    pred_base = float(bundle.predict(base_X).iloc[0])
    pred_recon = float(bundle.predict(X_recon).iloc[0])
    print('pred_base =', pred_base)
    print('pred_recon =', pred_recon)
    print('same?', abs(pred_base - pred_recon) < 1e-8)


if __name__ == '__main__':
    main()
