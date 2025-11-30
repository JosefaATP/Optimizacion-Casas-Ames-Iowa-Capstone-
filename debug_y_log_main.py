"""Debug y_log mismatch - 528328100."""
import sys
sys.path.insert(0, '.')

from optimization.remodel.xgb_predictor import XGBBundle
import pandas as pd

# Load data
df = pd.read_csv('data/processed/base_completa_sin_nulos.csv')
casa_idx = df[df['PID'] == 528328100].index[0]
casa_row = df.iloc[casa_idx]

# Load model
bundle = XGBBundle()

# Get X and y_log predictions
X_single = pd.DataFrame([casa_row])
y_hat = bundle.predict(X_single).iloc[0]
y_log_out = bundle.predict_log(X_single).iloc[0]
y_log_raw_out = bundle.predict_log_raw(X_single).iloc[0]
b0 = bundle.b0_offset

print(f"External predictor:")
print(f"  y_hat (price): {y_hat:12.2f}")
print(f"  y_log (including base): {y_log_out:12.6f}")
print(f"  y_log_raw (excluding base): {y_log_raw_out:12.6f}")
print(f"  base_score: {b0:12.6f}")
print(f"  Verification: y_log_raw + base = {y_log_raw_out + b0:12.6f}")

print(f"\nMIP reported:")
print(f"  y_log_raw: 0.934545")
print(f"  y_log: 13.372293")
print(f"  Expected y_log from y_log_raw: {0.934545 + b0:12.6f}")
print(f"  Difference: {13.372293 - y_log_out:12.6f}")

print(f"\nNote: External y_log_raw should be {y_log_raw_out:12.6f}")
print(f"      but MIP used {0.934545:12.6f}")
print(f"      Difference: {0.934545 - y_log_raw_out:12.6f}")
