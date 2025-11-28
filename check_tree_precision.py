#!/usr/bin/env python
"""
Diagnostic: Verificar si gurobi_ml evalúa árboles igual que XGBoost.

Problema observado:
- bundle.predict(X) = 241,548
- MIP predice = 270,369 (después de cambios en Open Porch SF, Heating)

Hipótesis: Hay diferencia en evaluación de árboles entre gurobi_ml y XGBoost.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
PROJ_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJ_ROOT))

from optimization.remodel.xgb_predictor import XGBBundle
from optimization.remodel.gurobi_model import build_base_input_row, build_mip_embed
import xgboost as xgb

def main():
    # Cargar bundle (como lo hace run_opt.py)
    bundle = XGBBundle()
    print(f"Bundle loaded")
    print(f"  - base_score (b0_offset): {bundle.b0_offset}")
    print(f"  - n_trees_use: {bundle.n_trees_use}")
    
    # Check booster base_score
    bst = bundle.reg.get_booster()
    bs_attr = bst.attr("base_score")
    print(f"  - Booster.attr('base_score'): {bs_attr}")
    
    # Datos test (sin argumentos para ver estructura)
    from optimization.remodel.config import DATA_BASE
    df = pd.read_csv(DATA_BASE)
    X_base = df[df['PID'] == 526351010]
    print(f"\n[BASE] X shape: {X_base.shape}")
    if len(X_base) == 0:
        print("ERROR: PID not found, using first row")
        X_base = df.iloc[0:1]
    print(f"\n[BASE] X shape: {X_base.shape}")
    print(f"[BASE] Open Porch SF: {X_base['Open Porch SF'].values[0]}")
    print(f"[BASE] Heating_GasA: {X_base.get('Heating_GasA', pd.Series([np.nan])).values[0]}")
    
    # Predicción en base
    y_pred_base = float(bundle.predict(X_base).iloc[0])
    y_log_base_direct = float(bundle.predict_log_raw(X_base).iloc[0])
    print(f"\n[BASE] Predicción precio: {y_pred_base:,.2f}")
    print(f"[BASE] Log margin (raw): {y_log_base_direct:.6f}")
    
    # Modifica features como lo hizo el MIP
    X_mod = X_base.copy()
    X_mod['Open Porch SF'] = 39.6
    X_mod['Heating_GasA'] = 0.0
    
    y_pred_mod = float(bundle.predict(X_mod).iloc[0])
    y_log_mod_direct = float(bundle.predict_log_raw(X_mod).iloc[0])
    print(f"\n[MOD] Predicción precio: {y_pred_mod:,.2f}")
    print(f"[MOD] Log margin (raw): {y_log_mod_direct:.6f}")
    print(f"[MOD] Δ precio: {y_pred_mod - y_pred_base:+,.2f}")
    print(f"[MOD] Δ log: {y_log_mod_direct - y_log_base_direct:+.6f}")
    
    # Ahora evalúa manualmente el árbol para ver divergencia
    print(f"\n[TREE-MANUAL] Evaluating trees manually...")
    
    bst = bundle.reg.get_booster()
    
    # Evalúa en escala base
    y_base_with_margin = float(bst.predict(
        xgb.DMatrix(X_base.iloc[:, :].values), 
        output_margin=True
    )[0])
    print(f"  BASE via bst.predict(output_margin=True): {y_base_with_margin:.6f}")
    
    # Evalúa modificado
    y_mod_with_margin = float(bst.predict(
        xgb.DMatrix(X_mod.iloc[:, :].values),
        output_margin=True
    )[0])
    print(f"  MOD via bst.predict(output_margin=True): {y_mod_with_margin:.6f}")
    print(f"  Δ margin: {y_mod_with_margin - y_base_with_margin:+.6f}")
    
    # Compara con lo que el MIP reportó
    print(f"\n[COMPARISON to MIP]")
    print(f"  External (via bundle): {y_log_mod_direct:.6f}")
    print(f"  MIP reported: 12.507481")
    print(f"  Δ: {12.507481 - y_log_mod_direct:+.6f}")
    
    print(f"\nΔ precio final:")
    print(f"  External (via bundle): {y_pred_mod:,.2f}")
    print(f"  MIP reported y_price: 270,369")
    print(f"  Δ: {270369 - y_pred_mod:+,.2f}")

if __name__ == "__main__":
    main()
