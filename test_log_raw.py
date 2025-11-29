from optimization.remodel.xgb_predictor import XGBBundle
import pandas as pd
import numpy as np

# Load bundle
bundle = XGBBundle()
print(f'bundle.b0_offset = {bundle.b0_offset}')

# Load base data
df_all = pd.read_csv('data/processed/base_completa_sin_nulos.csv')
X_test = df_all[df_all['PID'] == 526351010].drop(columns=['SalePrice', 'PID', 'SalePrice_Present'], errors='ignore')

# Preprocess
X_fixed = X_test.copy()
try:
    Xp = bundle.pre.transform(X_fixed)
except Exception as e:
    print(f'Preprocessing error: {e}')
    Xp = X_test.values

print(f'\nX preprocessed shape: {Xp.shape}')

# Get the row
if isinstance(Xp, np.ndarray):
    row = Xp[0]
else:
    row = Xp.values[0]

# Try _eval_sum_leaves  
try:
    sum_leaves = bundle._eval_sum_leaves(np.ravel(row))
    print(f'_eval_sum_leaves = {sum_leaves:.6f}')
    print(f'+ b0_offset ({bundle.b0_offset:.6f}) = {sum_leaves + bundle.b0_offset:.6f}')
except Exception as e:
    print(f'Error in _eval_sum_leaves: {e}')

# Compare with predict_log_raw
y_log_raw = bundle.predict_log_raw(X_test)
print(f'\npredict_log_raw result = {y_log_raw.values[0]:.6f}')

# Compare with reg.predict output_margin
y_margin = bundle.reg.predict(Xp, output_margin=True)
print(f'reg.predict(output_margin=True) = {y_margin[0]:.6f}')

# Test prediction
y_hat = bundle.predict(X_test)
print(f'\npredict (original scale) = {y_hat.values[0]:.2f}')
print(f'log(predict) = {np.log(y_hat.values[0]):.6f}')

print(f'\nExpected y_log (from MIP) should be around: 12.522918')
