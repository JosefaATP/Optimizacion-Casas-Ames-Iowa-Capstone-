#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from optimization.remodel.xgb_predictor import XGBBundle
from optimization.remodel.gurobi_model import build_base_input_row, build_mip_embed
from optimization.remodel.config import PATHS
from optimization.remodel.costs import CostTables
import gurobipy as gp


def main():
    print('[STEP] Loading XGBBundle...')
    bundle = XGBBundle(PATHS.xgb_model_file)
    print('[STEP] Bundle loaded')
    df = pd.read_csv(PATHS.base_csv, low_memory=False)
    row = df.iloc[1]               # cambia índice si quieres otra fila

    # predicción base
    Xb = build_base_input_row(bundle, row)
    print('[STEP] Computing base prediction...')
    base_pred = float(bundle.predict(Xb).iloc[0])
    print('base_pred =', base_pred)

    # construir MIP fijando variables al estado base
    print('[STEP] Building MIP (fix_to_base=True)...')
    # silence Gurobi console output for this check
    try:
        gp.setParam('OutputFlag', 0)
    except Exception:
        pass
    m = build_mip_embed(row, budget=40000.0, ct=CostTables(), bundle=bundle, fix_to_base=True)
    print('[STEP] MIP built')

    # reconstruir entrada numérica desde m._X_input (lee LBs de las vars fijadas)
    Xn = m._X_input.copy()
    for c in Xn.columns:
        v = Xn.iloc[0].loc[c]
        try:
            Xn.iloc[0, Xn.columns.get_loc(c)] = float(v)
        except Exception:
            var = m.getVarByName(f'x_{c}')
            Xn.iloc[0, Xn.columns.get_loc(c)] = float(var.LB)

    print('[STEP] Computing prediction from reconstructed input...')
    pred_fixed = float(bundle.predict(Xn.astype(float)).iloc[0])
    print('pred_fixed =', pred_fixed)
    print('same?', abs(base_pred - pred_fixed) < 1e-6)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print('Exception during run:')
        traceback.print_exc()
        raise
