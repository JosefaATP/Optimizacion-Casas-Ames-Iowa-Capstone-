from copy import deepcopy
from typing import Dict, Optional, Tuple
from .xgb_predictor import XGBPricePredictor
# Importamos la función canon para asegurar la consistencia del formato
from .gurobi_model import canon 

# Lista de etiquetas para Pool QC, asumiendo mapeo 0=NA, 1=Po, 2=Fa, 3=TA, 4=Gd, 5=Ex
# Ya que el ub en Gurobi es 4, usaremos la jerarquía más probable que incluye NA.
# Basado en el listado de calidades: 
# (NA, Po, Fa, TA, Gd, Ex) -> (0, 1, 2, 3, 4, 5)
# Ya que el modelo en gurobi.py usa ub=4 para PoolQC, definimos el orden hasta ese índice.
ORDER_QC_POOL = ["NA", "Po", "Fa", "TA", "Gd", "Ex"] # Usamos esta como base


def apply_plan(base_features: Dict[str, any], plan: Dict[str, any]) -> Dict[str, any]:
    """
    Crea el nuevo diccionario de features aplicando las modificaciones del plan.
    Aplica canonicalización (normalización de formatos) en el proceso para 
    garantizar consistencia con el pipeline de XGBoost.
    """
    f = deepcopy(base_features)
    
    # ------------------------------------------------------------------
    # 1. Normalización de las features base
    # Aplicamos canon a las variables que manejan formatos especiales (NA, Yes/No, etc.)
    # ------------------------------------------------------------------
    
    # Claves que requieren normalización (incluimos las variables de área para consistencia)
    KEYS_TO_CANONIZE = [
        "Utilities", "Roof Style", "Roof Matl", "Exterior 1st", "Exterior 2nd", 
        "Mas Vnr Type", "Electrical", "Heating", "Kitchen Qual", "Central Air", 
        "Pool QC", "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Pool Area"
    ]
    
    for key in KEYS_TO_CANONIZE:
        if key in f:
            # Sobreescribimos el valor base con su versión canonizada
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

    # Central Air (se maneja como string 'Yes'/'No' en el plan, no como lista)
    if "CentralAir" in plan:
        f["Central Air"] = canon("Central Air", plan["CentralAir"])


    # ------------------------------------------------------------------
    # 3. Aplicar Cambios de Área (Sótano y Piscina)
    # ------------------------------------------------------------------

    # A. Sótano Terminado: x_b1 y x_b2 (usamos los valores exportados de Gurobi)
    if plan.get("FinishBSMT", 0) == 1:
        x_b1 = plan.get("x_to_BsmtFin1", 0.0)
        x_b2 = plan.get("x_to_BsmtFin2", 0.0)

        # Las áreas base ya están canonizadas/normalizadas
        bsmt_fin1_base = f.get("BsmtFin SF 1", 0.0) or 0.0 
        bsmt_fin2_base = f.get("BsmtFin SF 2", 0.0) or 0.0
        
        # Actualizar las áreas terminadas con la delta
        f["BsmtFin SF 1"] = bsmt_fin1_base + x_b1
        f["BsmtFin SF 2"] = bsmt_fin2_base + x_b2
        
        # El área no terminada pasa a cero (asumiendo que se termina completamente)
        f["Bsmt Unf SF"] = 0.0
    
    # B. Piscina: PoolQC y PoolArea
    add_pool = plan.get("AddPool", 0)
    pool_area = plan.get("PoolArea", 0.0)
    pool_qc = plan.get("PoolQC", 0) # Viene como un entero (0=NA, 1=Po, ...)

    if add_pool == 1 and pool_area > 0:
        # Si se añade, actualiza el área y mapea la calidad entera a la categoría string
        f["Pool Area"] = pool_area
        
        # Mapea el score entero (0, 1, 2,...) a la etiqueta de calidad (NA, Po, Fa, ...)
        # Usamos min(pool_qc, 5) para no exceder el índice 5 (Ex) de la lista.
        f["Pool QC"] = ORDER_QC_POOL[min(int(pool_qc), len(ORDER_QC_POOL) - 1)]
    elif f.get("Pool Area", 0.0) == 0.0:
        # Si no se añadió y la casa base no tenía, asegurar que sea el valor canónico "None"
        f["Pool Area"] = 0.0
        f["Pool QC"] = canon("Pool QC", "None") # Asegurar que sea el valor 'NA' o 'None' que espera el pipeline

    return f


def diff_changes(base: Dict[str, any], new: Dict[str, any]) -> Dict[str, tuple]:
    """
    Compara dos diccionarios de features (base y nuevo) y reporta las diferencias,
    aplicando normalización en la comparación para ignorar diferencias de formato.
    """
    # 
    keys = [
        "Utilities","Roof Style","Roof Matl","Exterior 1st","Exterior 2nd",
        "Mas Vnr Type","Electrical","Central Air","Heating","Kitchen Qual",
        "Pool Area","Pool QC",
        # <<< AGREGA LAS ÁREAS DEL SÓTANO AQUÍ >>>
        "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF"
    ]
    out = {}
    
    # La función canon aquí se usa para la COMPARACIÓN, asegurando que 
    # 'N' y 'No' sean tratados como iguales, y 'NA' y 'None' también.
    def normalize_for_diff(k, v):
         # Usamos la función canon importada. Para Pool Area, convertimos a float.
        if k in ["BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Pool Area"]:
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
        
        # Comparación numérica (flotantes)
        if isinstance(b_norm, float) and isinstance(n_norm, float):
            if abs(b_norm - n_norm) > 1e-6:
                out[k] = (b_norm, n_norm)
        # Comparación categórica (cadenas normalizadas)
        elif b_norm != n_norm:
            out[k] = (b_norm, n_norm)
            
    return out


def score_plans(predictor: XGBPricePredictor, base_features: Dict[str, any], plans: list, base_value: Optional[float] = None):
    """
    Calcula la rentabilidad (profit) para un conjunto de planes de remodelación.
    """
    
    # 1. Obtener la base canonicalizada y su predicción (base_can ahora es la casa normalizada)
    base_can = apply_plan(base_features, {})
    if base_value is None:
        base_value = predictor.predict_price(base_can)

    results = []
    
    for p in plans:
        # La casa base canonicalizada debe ser aplicada a CADA plan para asegurar un punto de partida consistente
        f_base_for_plan = deepcopy(base_can)
        print(">>> PLAN COMPLETO DE GUROBI:", p) #DEBUG

        if p.get("__baseline__", False):
            # Caso base (no hacer nada)
            pred_new = base_value
            cost = 0.0
            profit = 0.0
            changes = {}
            f_new = f_base_for_plan
        else:
            # Aplicar cambios sobre la base ya normalizada
            f_new = apply_plan(f_base_for_plan, p) 
            pred_new = predictor.predict_price(f_new)
            cost = float(p.get("Cost", 0.0))
            profit = pred_new - base_value - cost
            # Usamos base_can como referencia para diff_changes
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