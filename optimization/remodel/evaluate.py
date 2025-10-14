from copy import deepcopy
from typing import Dict, Optional
from .xgb_predictor import XGBPricePredictor

ORDER_QC = ["NA","Fa","TA","Gd","Ex"]

def apply_plan(base_features: Dict[str, any], plan: Dict[str, any]) -> Dict[str, any]:
    f = deepcopy(base_features)

    # mapeo nombres del plan -> nombres crudos del CSV
    if plan.get("Utilities"):   f["Utilities"]    = plan["Utilities"][0]
    if plan.get("RoofStyle"):   f["Roof Style"]   = plan["RoofStyle"][0]
    if plan.get("RoofMatl"):    f["Roof Matl"]    = plan["RoofMatl"][0]
    if plan.get("Exterior1st"): f["Exterior 1st"] = plan["Exterior1st"][0]
    if plan.get("Exterior2nd"): f["Exterior 2nd"] = plan["Exterior2nd"][0] if plan["Exterior2nd"] else ""
    if plan.get("MasVnrType"):
        mv = plan["MasVnrType"][0]
        f["Mas Vnr Type"] = "None" if mv in ("No aplica","","NA",None) else mv
    if plan.get("Electrical"):  f["Electrical"]   = plan["Electrical"][0]
    if "CentralAir" in plan:    f["Central Air"]  = plan["CentralAir"]
    if plan.get("Heating"):     f["Heating"]      = plan["Heating"][0]
    if plan.get("KitchenQual"): f["Kitchen Qual"] = plan["KitchenQual"][0]

    # piscina
    if plan.get("AddPool",0) == 1:
        f["Pool Area"] = max(float(plan.get("PoolArea", 0.0)), 0.0)
        qc = plan.get("PoolQC", 0)
        f["Pool QC"] = ORDER_QC[int(qc)] if isinstance(qc,int) else qc
    else:
        f["Pool Area"] = 0.0
        f["Pool QC"]   = "NA"
    return f

def diff_changes(base: Dict[str, any], new: Dict[str, any]) -> Dict[str, tuple]:
    keys = [
        "Utilities","Roof Style","Roof Matl","Exterior 1st","Exterior 2nd",
        "Mas Vnr Type","Electrical","Central Air","Heating","Kitchen Qual",
        "Pool Area","Pool QC"
    ]
    out = {}
    for k in keys:
        bv = base.get(k, None)
        nv = new.get(k, None)
        if str(bv) != str(nv):
            out[k] = (bv, nv)
    return out

def score_plans(predictor: XGBPricePredictor, base_features: Dict[str, any], plans: list, base_value: Optional[float] = None):
    # pred base una sola vez
    if base_value is None:
        base_value = predictor.predict_price(base_features)

    results = []
    for p in plans:
        f_new = apply_plan(base_features, p)
        pred_new = predictor.predict_price(f_new)
        cost = float(p.get("Cost", 0.0))
        profit = pred_new - base_value - cost
        changes = diff_changes(base_features, f_new)
        results.append({
            "plan": p,
            "pred_final": pred_new,
            "base_pred": base_value,
            "cost": cost,
            "profit": profit,
            "changes": changes,
            "features_new": f_new,
        })
    # ordenar por utilidad
    results.sort(key=lambda x: x["profit"], reverse=True)
    return results
