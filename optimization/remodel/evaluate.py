from copy import deepcopy
from typing import Dict, Optional, Tuple
from .xgb_predictor import XGBPricePredictor
# Centralize canonicalization helpers
from .canon import canon, qscore

# Lista de etiquetas de calidad para mapear el score entero (0=NA, 1=Po, ..., 5=Ex)
ORDER_QC_POOL = ["NA", "Po", "Fa", "TA", "Gd", "Ex"] 


def apply_plan(base_features: Dict[str, any], plan: Dict[str, any]) -> Dict[str, any]:
    """
    Crea el nuevo diccionario de features aplicando las modificaciones del plan.
    Aplica canonicalización (normalización de formatos) en el proceso para 
    garantizar consistencia con el pipeline de XGBoost.
    """
    f = deepcopy(base_features)
    
    # ------------------------------------------------------------------
    # 1. Normalización de las features base (Punto Crítico)
    # ------------------------------------------------------------------
    
    KEYS_TO_CANONIZE = [
        "Utilities", "Roof Style", "Roof Matl", "Exterior 1st", "Exterior 2nd", 
        "Mas Vnr Type", "Electrical", "Heating", "Kitchen Qual", "Central Air", 
        "Pool QC", "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Pool Area",
        # Agregamos las que tienen riesgo de formato, incluso si no se tocan en el MILP
        "Bsmt Cond", "Fireplace Qu", "Paved Drive", "Garage Qual", "Garage Cond" 
    ]
    
    for key in KEYS_TO_CANONIZE:
        if key in f:
            # Sobreescribimos el valor base con su versión canonizada, lo que da una base limpia
            f[key] = canon(key, f.get(key))


    # ------------------------------------------------------------------
    # 2. Aplicar cambios Categóricos (desde el plan de Gurobi)
    # ------------------------------------------------------------------
    
    categorical_changes = {
        "Utilities": "Utilities", "Roof Style": "RoofStyle", "Roof Matl": "RoofMatl",
        "Exterior 1st": "Exterior1st", "Exterior 2nd": "Exterior2nd", 
        "Mas Vnr Type": "MasVnrType", "Electrical": "Electrical", "Heating": "Heating", 
        "Kitchen Qual": "KitchenQual"
    }

    for final_key, plan_key in categorical_changes.items():
        chosen_value = plan.get(plan_key)
        # Extraer el valor de la lista (Gurobi devuelve [valor])
        if isinstance(chosen_value, list) and chosen_value:
            chosen_value = chosen_value[0]
        
        if chosen_value is not None:
             # Asignar el valor canonizado al diccionario de features
             f[final_key] = canon(final_key, chosen_value)

    # Central Air (se maneja como string 'Yes'/'No' en el plan)
    if "CentralAir" in plan:
        f["Central Air"] = canon("Central Air", plan["CentralAir"])


    # ------------------------------------------------------------------
    # 3. Aplicar Cambios de Área (Sótano y Piscina)
    # ------------------------------------------------------------------

    # A. Sótano Terminado (CRÍTICO: Lógica de Deltas)
    if plan.get("FinishBSMT", 0) == 1:
        x_b1 = plan.get("x_to_BsmtFin1", 0.0)
        x_b2 = plan.get("x_to_BsmtFin2", 0.0)

        # Las áreas base ya están canonizadas/normalizadas y son floats (gracias al paso 1)
        bsmt_fin1_base = f.get("BsmtFin SF 1", 0.0)
        bsmt_fin2_base = f.get("BsmtFin SF 2", 0.0)
        
        # El nuevo valor es el base + lo añadido por Gurobi
        f["BsmtFin SF 1"] = bsmt_fin1_base + x_b1
        f["BsmtFin SF 2"] = bsmt_fin2_base + x_b2
        
        # El área no terminada pasa a cero
        f["Bsmt Unf SF"] = 0.0 # Esto asume que el MILP obliga a terminar todo lo no terminado

    # B. Piscina: PoolQC y PoolArea
    add_pool = plan.get("AddPool", 0)
    pool_area = plan.get("PoolArea", 0.0)
    pool_qc = plan.get("PoolQC", 0) # Viene como un entero (0=NA, 1=Po, ...)

    if add_pool == 1 and pool_area > 0:
        # Si se añade, actualiza el área y mapea la calidad entera a la categoría string
        f["Pool Area"] = pool_area
        
        # Mapea el score entero (0, 1, 2,...) a la etiqueta de calidad (NA, Po, Fa, ...)
        f["Pool QC"] = ORDER_QC_POOL[min(int(pool_qc), len(ORDER_QC_POOL) - 1)]
        
    elif f.get("Pool Area", 0.0) == 0.0:
        # Si no se añadió y la casa base no tenía, asegurar que el valor sea el canónico 'None'
        f["Pool Area"] = 0.0
        f["Pool QC"] = canon("Pool QC", "None") 

    return f


def diff_changes(base: Dict[str, any], new: Dict[str, any]) -> Dict[str, tuple]:
    """
    Compara dos diccionarios de features (base y nuevo) y reporta las diferencias,
    aplicando normalización en la comparación para ignorar diferencias de formato.
    """
    
    keys = [
        "Utilities","Roof Style","Roof Matl","Exterior 1st","Exterior 2nd",
        "Mas Vnr Type","Electrical","Central Air","Heating","Kitchen Qual",
        "Pool Area","Pool QC",
        # Las áreas del sótano son CRÍTICAS
        "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF"
    ]
    out = {}
    
    # La función normalize_for_diff usa canon para garantizar la comparación de formatos
    def normalize_for_diff(k, v):
        if k in ["BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Pool Area"]:
            # Las áreas ya fueron tratadas en apply_plan y son float, pero aseguramos
            try:
                return float(v or 0.0)
            except Exception:
                return 0.0
        return canon(k, v)


    for k in keys:
        b = base.get(k, None)
        n = new.get(k, None)
        
        b_norm = normalize_for_diff(k, b)
        n_norm = normalize_for_diff(k, n)
        
        # Comparación numérica (flotantes: áreas)
        if isinstance(b_norm, float) and isinstance(n_norm, float):
            if abs(b_norm - n_norm) > 1e-6:
                out[k] = (round(b_norm, 1), round(n_norm, 1))
        # Comparación categórica (cadenas normalizadas)
        elif b_norm != n_norm:
            out[k] = (b_norm, n_norm)
            
    return out


def score_plans(predictor: XGBPricePredictor, base_features: Dict[str, any], plans: list, base_value: Optional[float] = None):
    """
    Calcula la rentabilidad (profit) para un conjunto de planes de remodelación.
    """
    
    # CRÍTICO: 1. Obtener la base canonicalizada (aplicar apply_plan con plan vacío)
    base_can = apply_plan(base_features, {})
    if base_value is None:
        base_value = predictor.predict_price(base_can)

    results = []
    
    for p in plans:
        # print(">>> PLAN COMPLETO DE GUROBI:", p) # Se movió el debug al loop principal

        if p.get("__baseline__", False):
            # Caso base (no hacer nada)
            f_new = base_can
            pred_new = base_value
            cost = 0.0
            profit = 0.0
            changes = {}
        else:
            # 2. Aplicar cambios sobre la base ya normalizada
            f_new = apply_plan(deepcopy(base_can), p) 
            pred_new = predictor.predict_price(f_new)
            cost = float(p.get("Cost", 0.0))
            profit = pred_new - base_value - cost
            # 3. Usamos base_can como referencia para diff_changes
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