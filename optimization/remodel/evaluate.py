from copy import deepcopy
from typing import Dict, Optional
from .xgb_predictor import XGBPricePredictor

ORDER_QC = ["NA","Fa","TA","Gd","Ex"]

def apply_plan(base_features: Dict[str, any], plan: Dict[str, any]) -> Dict[str, any]:
    f = deepcopy(base_features)

    if plan.get("Utilities"):   f["Utilities"]    = plan["Utilities"][0]
    if plan.get("RoofStyle"):   f["Roof Style"]   = plan["RoofStyle"][0]
    if plan.get("RoofMatl"):    f["Roof Matl"]    = plan["RoofMatl"][0]
    if plan.get("Exterior1st"): f["Exterior 1st"] = plan["Exterior1st"][0]
    if plan.get("Exterior2nd"):
        f["Exterior 2nd"] = plan["Exterior2nd"][0] if plan["Exterior2nd"] else f.get("Exterior 2nd","")
    if plan.get("MasVnrType"):
        mv = plan["MasVnrType"][0]
        # Mantén exactamente la misma convención que el modelo/entrenamiento
        if str(mv).strip() in ("No aplica","NA","","None"):
            # usa la MISMA cadena que haya en la base si ya es una de las equivalentes
            base_mv = str(f.get("Mas Vnr Type",""))
            f["Mas Vnr Type"] = base_mv if base_mv in ("No aplica","NA","","None","None ") else "None"
        else:
            f["Mas Vnr Type"] = mv

    if plan.get("Electrical"):  f["Electrical"]   = plan["Electrical"][0]
    if "CentralAir" in plan:
        v = plan["CentralAir"]
        f["Central Air"] = {"Y":"Yes","N":"No","Yes":"Yes","No":"No"}.get(str(v).strip(), str(v))

    if plan.get("Heating"):     f["Heating"]      = plan["Heating"][0]
    if plan.get("KitchenQual"): f["Kitchen Qual"] = plan["KitchenQual"][0]

    # piscina
    if plan.get("AddPool",0) == 1:
        f["Pool Area"] = float(plan.get("PoolArea", 0.0) or 0.0)
        qc = plan.get("PoolQC", 0)
        f["Pool QC"] = ["NA","Fa","TA","Gd","Ex"][int(qc)] if isinstance(qc,int) else str(qc)
    else:
        f["Pool Area"] = 0.0
        # mantén la misma convención de la base si ya trae NA/No aplica/None
        base_pq = str(f.get("Pool QC","NA")).strip()
        f["Pool QC"] = "NA" if base_pq in ("NA","No aplica","","None") else base_pq
    return f

def diff_changes(base: Dict[str, any], new: Dict[str, any]) -> Dict[str, tuple]:
    keys = [
        "Utilities","Roof Style","Roof Matl","Exterior 1st","Exterior 2nd",
        "Mas Vnr Type","Electrical","Central Air","Heating","Kitchen Qual",
        "Pool Area","Pool QC"
    ]
    out = {}
    for k in keys:
        b = base.get(k, None)
        n = new.get(k, None)
        # ignora diferencias 0 vs 0.0
        if k == "Pool Area":
            try:
                if abs(float(b or 0.0) - float(n or 0.0)) < 1e-6:
                    continue
            except Exception:
                pass
        # normaliza equivalentes
        def norm(col, v):
            s = "" if v is None else str(v).strip()
            if col == "Central Air":
                return {"Y":"Yes","N":"No","Yes":"Yes","No":"No"}.get(s, s)
            if col == "Mas Vnr Type":
                return "No aplica" if s in ("No aplica","NA","","None") else s
            if col == "Pool QC":
                return "NA" if s in ("NA","No aplica","","None","0") else s
            if col == "Exterior 2nd":
                return "NA" if s == "" else s
            return s
        if norm(k,b) != norm(k,n):
            out[k] = (norm(k,b), norm(k,n))
    return out

def score_plans(predictor: XGBPricePredictor, base_features: Dict[str, any], plans: list, base_value: Optional[float] = None):
    # Canonicaliza la base con la MISMA lógica de apply_plan (plan vacío)
    base_can = apply_plan(base_features, {})
    if base_value is None:
        base_value = predictor.predict_price(base_can)

    results = []
    for p in plans:
        if p.get("__baseline__", False):
            pred_new = base_value
            cost = 0.0
            profit = 0.0
            changes = {}
            f_new = base_can
        else:
            f_new = apply_plan(base_can, p)
            pred_new = predictor.predict_price(f_new)
            cost = float(p.get("Cost", 0.0))
            profit = pred_new - base_value - cost
            changes = diff_changes(base_can, f_new)

        f_new = apply_plan(base_can, p)  # aplica cambios sobre la base canonicalizada
        pred_new = predictor.predict_price(f_new)
        cost = float(p.get("Cost", 0.0))
        profit = pred_new - base_value - cost
        changes = diff_changes(base_can, f_new)
        results.append({
            "plan": p,
            "pred_final": pred_new,
            "base_pred": base_value,
            "cost": cost,
            "profit": profit,
            "changes": changes,
            "features_new": f_new,
        })
    results.sort(key=lambda x: x["profit"], reverse=True)
    return results
