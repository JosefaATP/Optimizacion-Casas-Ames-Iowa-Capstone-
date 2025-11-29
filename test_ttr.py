from optimization.remodel.xgb_predictor import XGBBundle
import pandas as pd
import numpy as np

bundle = XGBBundle()

# Check if there's a TTR wrapper
print(f"bundle._ttr is not None: {bundle._ttr is not None}")
print(f"bundle.log_target: {bundle.log_target}")
print(f"bundle.reg type: {type(bundle.reg)}")

# Load a test property - use the one from the optimization
df_base = pd.read_csv('data/processed/base_completa_sin_nulos.csv')
X_base = df_base[df_base['PID'] == 526351010].drop(columns=['SalePrice', 'PID', 'SalePrice_Present'], errors='ignore')

# Test what bundle.predict returns
y_pred = bundle.predict(X_base)
print(f"\nbundle.predict() = {y_pred.values[0]:.2f}")

# Test what the underlying regressor returns
X_fixed = X_base.copy()
Xp = bundle.pre.transform(X_fixed)

y_reg_pred = bundle.reg.predict(Xp)
print(f"bundle.reg.predict(Xp) = {y_reg_pred[0]:.6f}")

y_reg_margin = bundle.reg.predict(Xp, output_margin=True)
print(f"bundle.reg.predict(Xp, output_margin=True) = {y_reg_margin[0]:.6f}")

# Now check: if bundle._ttr exists, it might be doing log1p transformation
if bundle._ttr is not None:
    print(f"\nFound TTR wrapper!")
    print(f"TTR func: {bundle._ttr.func}")
    print(f"TTR inverse_func: {bundle._ttr.inverse_func}")
    
    # The TTR might be wrapping the regressor with log1p
    # So output_margin from the underlying regressor might need transformation
    
    # Let's check: does output_margin from TTR-wrapped regressor already include the transformation?
    print(f"\nDirect regressor type: {type(bundle._ttr.regressor_)}")
