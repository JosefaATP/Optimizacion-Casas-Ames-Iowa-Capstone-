from optimization.remodel.xgb_predictor import XGBBundle
import pandas as pd
import numpy as np
import json

bundle = XGBBundle()

# Load base data  
df_all = pd.read_csv('data/processed/base_completa_sin_nulos.csv')
X_test = df_all[df_all['PID'] == 526351010].drop(columns=['SalePrice', 'PID', 'SalePrice_Present'], errors='ignore')

# Get all columns needed for preprocessing
train_X = pd.read_csv('data/processed/base_completa_sin_nulos.csv').drop(columns=['SalePrice', 'PID', 'SalePrice_Present'], errors='ignore')

# Reindex X_test to match train_X columns
X_test_aligned = X_test.copy()
for col in train_X.columns:
    if col not in X_test_aligned.columns:
        X_test_aligned[col] = train_X[col].mode()[0] if col in train_X.columns else 0
        
X_test_aligned = X_test_aligned[train_X.columns]

# Preprocess
X_fixed = X_test_aligned.copy()
try:
    Xp = bundle.pre.transform(X_fixed)
    print(f'Preprocessing OK: shape {Xp.shape}')
    
    # Now test predictions
    y_pred_full = bundle.reg.predict(Xp)
    y_pred_margin = bundle.reg.predict(Xp, output_margin=True)
    
    print(f'\nreg.predict() = {y_pred_full[0]:.6f}')
    print(f'reg.predict(output_margin=True) = {y_pred_margin[0]:.6f}')
    print(f'Difference = {y_pred_full[0] - y_pred_margin[0]:.6f}')
    print(f'bundle.b0_offset = {bundle.b0_offset:.6f}')
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
