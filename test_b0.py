from optimization.remodel.xgb_predictor import XGBBundle

bundle = XGBBundle()
print(f'bundle.b0_offset = {bundle.b0_offset}')

try:
    bst = bundle.reg.get_booster()
    params = bst.get_params()
    print(f'Booster params base_score: {params.get("base_score")}')
except Exception as e:
    print(f'Error getting from booster: {e}')

# Try direct attribute
print(f'bundle.reg.base_score = {getattr(bundle.reg, "base_score", "NOTFOUND")}')
print(f'bundle.reg params: {bundle.reg.get_params().get("base_score")}')

# Try booster's JSON
try:
    bst = bundle.reg.get_booster()
    json_model = bst.save_raw("json")
    import json
    data = json.loads(json_model)
    bs = data.get("learner", {}).get("learner_model_param", {}).get("base_score")
    print(f'Base score from JSON: {bs}')
except Exception as e:
    print(f'Error from JSON: {e}')
