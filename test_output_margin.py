import xgboost as xgb
print(f"XGBoost version: {xgb.__version__}")

# Simple test: create a simple booster and check output_margin behavior
import numpy as np

# Create synthetic data
X = np.random.randn(10, 5)
y = np.random.randn(10)

# Train simple model
dtrain = xgb.DMatrix(X, label=y)
params = {
    'objective': 'reg:squarederror',
    'max_depth': 2,
    'base_score': 12.5,  # Explicit base score
}
bst = xgb.train(params, dtrain, num_boost_round=5)

# Test predictions
X_test = X[:1]
test = xgb.DMatrix(X_test)

# Get predictions
y_pred = bst.predict(test)[0]
y_margin = bst.predict(test, output_margin=True)[0]
y_raw = bst.predict(test, pred_leaf=False, output_margin=True)[0]

print(f"\nBase score set to: 12.5")
print(f"predict() = {y_pred:.6f}")
print(f"predict(output_margin=True) = {y_margin:.6f}")
print(f"Difference = {y_pred - y_margin:.6f}")
print(f"Expected difference (if margin includes base_score): ~{12.5:.6f}")
