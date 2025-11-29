from optimization.remodel.xgb_predictor import XGBBundle
import pandas as pd
import numpy as np

# Load the bundle
bundle = XGBBundle()

# Create a simple test - use the base house
df_all = pd.read_csv('data/processed/base_completa_sin_nulos.csv')
X_test = df_all[df_all['PID'] == 526351010].drop(columns=['SalePrice', 'PID', 'SalePrice_Present'], errors='ignore')

# Preprocess
X_fixed = X_test.copy()
try:
    Xp = bundle.pre.transform(X_fixed)
    print(f'Preprocessing OK: shape {Xp.shape}')
except Exception as e:
    print(f'Preprocessing failed: {e}')
    Xp = X_test.values

# Now test the different prediction methods
y_pred = bundle.reg.predict(Xp)
print(f'reg.predict(Xp) = {y_pred[0]:.6f}')

y_margin = bundle.reg.predict(Xp, output_margin=True)
print(f'reg.predict(Xp, output_margin=True) = {y_margin[0]:.6f}')

# What the bundle thinks
y_hat = bundle.predict(X_test)
print(f'bundle.predict() = {y_hat.values[0]:.2f}')
print(f'log1p of that = {np.log1p(y_hat.values[0]):.6f}')

# The b0_offset
print(f'\nbundle.b0_offset = {bundle.b0_offset:.6f}')
print(f'y_margin + b0 = {y_margin[0] + bundle.b0_offset:.6f}')
print(f'y_margin alone = {y_margin[0]:.6f}')

# So which is correct for y_log_raw?
# In XGB, output_margin is the raw margin BEFORE applying base_score
# BUT the MIP embed uses: y_log_raw_var + b0_offset where y_log_raw is what?
