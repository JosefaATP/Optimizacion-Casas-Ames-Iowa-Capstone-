import pandas as pd, numpy as np
from optimization.remodel.xgb_predictor import XGBBundle
from optimization.remodel.gurobi_model import build_base_input_row

pid = 526351010
base = pd.read_csv("data/processed/base_completa_sin_nulos.csv")
base_row = base.loc[base["PID"] == pid].iloc[0]

bundle = XGBBundle()
base_X = build_base_input_row(bundle, base_row)
X_in = pd.read_csv("X_input_after_opt.csv")

pipe_log = float(bundle.pipe_for_gurobi().predict(X_in)[0])
Xp = bundle.pre.transform(X_in)
try:
    xp_arr = Xp.toarray()
except Exception:
    xp_arr = Xp
sum_leaves = bundle._eval_sum_leaves(np.ravel(xp_arr))
offset = pipe_log - sum_leaves

price_pipe = float(bundle.predict(X_in).iloc[0])

print(f"pipe_log={pipe_log:.6f}")
print(f"sum_leaves={sum_leaves:.6f}")
print(f"offset (pipe - leaves)={offset:.6f}")
print(f"price_pipe={price_pipe:,.2f}")
heat_cols = [c for c in X_in.columns if c.startswith('Heating_')]
print("Heating dummies:", {c: float(X_in.loc[0, c]) for c in heat_cols})
order = bundle.booster_feature_order()
missing = [c for c in order if c not in X_in.columns]
print("missing cols vs booster order:", missing)
