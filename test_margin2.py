import xgboost as xgb
import numpy as np

print(f"XGBoost version: {xgb.__version__}")

# Create synthetic data
X = np.random.randn(10, 5)
y = np.ones(10) * 100  # Start with target around 100

# Train simple model
dtrain = xgb.DMatrix(X, label=y)
params = {
    'objective': 'reg:squarederror',
    'max_depth': 2,
    'base_score': 50.0,  # Explicit, relatively large base score
    'learning_rate': 0.1,
}
bst = xgb.train(params, dtrain, num_boost_round=10)

# Verify base_score  
base_score_set = 50.0

# Test predictions
X_test = np.random.randn(1, 5)
test = xgb.DMatrix(X_test)

y_pred = bst.predict(test)[0]
y_margin = bst.predict(test, output_margin=True)[0]

print(f"predict() = {y_pred:.6f}")
print(f"predict(output_margin=True) = {y_margin:.6f}")
print(f"Difference = {y_pred - y_margin:.6f}")
print(f"Base score = {base_score_set:.6f}")

# The relationship should be:
# predict() = exp1m(output_margin) if MSE is used
# or predict() = output_margin for linear objective

# Actually for squared error:
# The output is just the sum of tree predictions + base_score
# So output_margin should be without base_score
# And predict() should be output_margin + base_score

print(f"\nDoes predict - margin == base_score? {abs(y_pred - y_margin - 50.0) < 1e-6}")
