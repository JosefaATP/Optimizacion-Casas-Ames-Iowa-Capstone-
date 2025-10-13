from copy import deepcopy
from typing import Dict, Optional
from .xgb_predictor import XGBPricePredictor

def apply_plan(base_features: Dict[str, any], plan: Dict[str, any]) -> Dict[str, any]:
    f = deepcopy(base_features)

    if plan.get("Utilities"):   f["Utilities"]   = plan["Utilities"][0]
    if plan.get("RoofStyle"):   f["Roof Style"]  = plan["RoofStyle"][0]
    if plan.get("RoofMatl"):    f["Roof Matl"]   = plan["RoofMatl"][0]

    if plan.get("Exterior1st"): f["Exterior 1st"] = plan["Exterior1st"][0]
    if plan.get("Exterior2nd"): f["Exterior 2nd"] = plan["Exterior2nd"][0]

    if plan.get("MasVnrType"):  f["Mas Vnr Type"] = plan["MasVnrType"][0]
    if plan.get("Electrical"):  f["Electrical"]   = plan["Electrical"][0]
    if "CentralAir" in plan:    f["Central Air"]  = plan["CentralAir"]  # "Yes"/"No"
    if plan.get("Heating"):     f["Heating"]      = plan["Heating"][0]
    if plan.get("KitchenQual"): f["Kitchen Qual"] = plan["KitchenQual"][0]

    # pool
    if plan.get("AddPool",0) == 1:
        f["Pool Area"] = max(plan.get("PoolArea", 0.0), 0.0)
        qc = plan.get("PoolQC", 0)
        f["Pool QC"] = ["NA","Fa","TA","Gd","Ex"][qc] if isinstance(qc,int) else qc
    else:
        f["Pool Area"] = 0.0
        f["Pool QC"]   = "NA"
    return f


def score_plans(predictor: XGBPricePredictor, base_features: Dict[str, any], plans: list, base_value: Optional[float] = None):
    # si tienes el valor inicial por XGB o por dataset, calcula delta
    if base_value is None:
        base_value = predictor.predict_price(base_features)
    results = []
    for p in plans:
        f_new = apply_plan(base_features, p)
        pred = predictor.predict_price(f_new)
        profit = pred - base_value - p["Cost"]
        results.append({"plan": p, "pred_final": pred, "base_pred": base_value, "profit": profit})
    # ordenar por utilidad
    results.sort(key=lambda x: x["profit"], reverse=True)
    return results
