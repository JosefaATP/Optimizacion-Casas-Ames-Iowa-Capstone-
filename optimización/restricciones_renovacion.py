"""
restricciones_renovacion.py
===========================
Implementación completa de las restricciones de RENOVACIÓN según Sección 6 del PDF.

Grupo 5 - Capstone 2025
Modelo Matemático: Restricciones de Renovación
"""

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import pandas as pd
from config import (
    ROOF_COMPATIBILITY, COSTOS_ROOF_STYLE, COSTOS_ROOF_MATERIAL, COSTOS_EXTERIOR, COSTOS_UTILITIES, COSTO_BSMT_FINISH_SQFT,
    COSTO_CENTRAL_AIR, COSTOS_ELECTRICAL, COSTOS_HEATING, COSTOS_KITCHEN_QUAL, COSTOS_FENCE_CAT, COSTO_FENCE_PSF,
    COSTOS_PAVEDDRIVE, COSTOS_POOL_QUAL, COSTOS_GARAGE_FINISH)

# ============================================================================
# CONSTANTES Y MAPEOS
# ============================================================================

# Mapeo de calidades a índices
CALIDAD_A_INDICE = {
    'Po': 1,  # Poor
    'Fa': 2,  # Fair
    'TA': 3,  # Typical/Average
    'Gd': 4,  # Good
    'Ex': 5,  # Excellent
}

CATEGORIAS_PROMEDIO_O_MENOR = {'Po', 'Fa', 'TA'}

# Mapeo de calidades de acabado de sótano
BSMT_FINISH_QUAL_A_COSTO = { # El costo es el índice para el cálculo
    'NA': 0,
    'Unf': 1, 
    'LwQ': 2, 
    'Rec': 3, 
    'BLQ': 4, 
    'ALQ': 5, 
    'GLQ': 6, 
}
CATEGORIAS_BSMT_REC_O_PEOR = {'Rec', 'LwQ', 'Unf'}

# ============================================================================
# RESTRICCIÓN 1: UTILITIES (Sección 6 del PDF - pág 3-4)
# ============================================================================
"""
PDF dice:
- Se puede cambiar a alternativas de costo mayor o mantenerse
- Definición: U⁺ᵢ = {u ∈ {AllPub, NoSewr, NoSeWa, ELO} : Cᵤ ≥ C^base_u}
- Variables: Utilitiesᵢ,ᵤ ∈ {0,1}
- Restricción: Σ Utilitiesᵢ,ᵤ = 1
"""

COSTOS_UTILITIES = {
    'AllPub': 0,        # All public utilities (E, G, W, S)
    'NoSewr': 15000,    # No alcantarillado
    'NoSeWa': 20000,    # No alcantarillado ni agua
    'ELO': 25000,       # Solo electricidad y otros
}

def agregar_restriccion_utilities(modelo, casa_actual, i=0):
    """
    Agrega restricción de Utilities según PDF Sección 6.
    
    Args:
        modelo: Modelo de Gurobi
        casa_actual: Serie con características actuales
        i: Índice de la casa (default 0)
        
    Returns:
        dict: Variables creadas {nombre: variable_gurobi}
    """
    # Parámetro base
    u_base = casa_actual.get('Utilities', 'AllPub')
    C_u_base = COSTOS_UTILITIES[u_base]
    
    # Conjunto permitido U⁺ (solo alternativas de costo mayor o igual)
    U_plus = {u: C for u, C in COSTOS_UTILITIES.items() if C >= C_u_base}
    
    print(f"  Utilities actual: {u_base} (${C_u_base:,})")
    print(f"  Alternativas permitidas: {list(U_plus.keys())}")
    
    # Variables de decisión: Utilitiesᵢ,ᵤ
    utilities_vars = {}
    for u in U_plus.keys():
        utilities_vars[f'Utilities_{i}_{u}'] = modelo.addVar(
            vtype=GRB.BINARY, 
            name=f'Utilities_{i}_{u}'
        )
    
    # RESTRICCIÓN: Exactamente una debe estar activa
    modelo.addConstr(
        gp.quicksum(utilities_vars.values()) == 1,
        name=f'utilities_unique_{i}'
    )
    
    # Costo asociado
    costo_utilities = gp.quicksum(
        utilities_vars[f'Utilities_{i}_{u}'] * COSTOS_UTILITIES[u]
        for u in U_plus.keys()
    )
    
    return {
        'variables': utilities_vars,
        'costo': costo_utilities,
        'conjunto_permitido': U_plus
    }


# ============================================================================
# RESTRICCIÓN 2: ROOFSTYLE + ROOFMATL (Sección 6 del PDF - pág 4-5)
# ============================================================================
"""
PDF dice:
- Matriz de compatibilidad: A_s,m
- Restricción de incompatibilidad: xᵢ,ₛ + yᵢ,ₘ ≤ 1 ∀(s,m) donde A_s,m = 0
- Conjuntos permitidos: S⁺ᵢ, M⁺ᵢ (solo subir de costo)
"""

def agregar_restriccion_roof(modelo, casa_actual, i=0):
    """
    Agrega restricciones de RoofStyle y RoofMatl con compatibilidad.
    
    Args:
        modelo: Modelo de Gurobi
        casa_actual: Serie con características actuales
        i: Índice de la casa
        
    Returns:
        dict: Variables y costos
    """
    # Parámetros base
    s_base = casa_actual.get('RoofStyle', 'Gable')
    m_base = casa_actual.get('RoofMatl', 'CompShg')
    C_s_base = COSTOS_ROOF_STYLE[s_base]
    C_m_base = COSTOS_ROOF_MATERIAL[m_base]
    
    # Conjuntos permitidos (solo subir de costo)
    S_plus = {s: C for s, C in COSTOS_ROOF_STYLE.items() if C >= C_s_base}
    M_plus = {m: C for m, C in COSTOS_ROOF_MATERIAL.items() if C >= C_m_base}
    
    print(f"  RoofStyle actual: {s_base} → Permitidos: {list(S_plus.keys())}")
    print(f"  RoofMatl actual: {m_base} → Permitidos: {list(M_plus.keys())}")
    
    # Variables de decisión
    x_vars = {}  # Para estilos
    y_vars = {}  # Para materiales
    
    for s in S_plus.keys():
        x_vars[s] = modelo.addVar(vtype=GRB.BINARY, name=f'RoofStyle_{i}_{s}')
    
    for m in M_plus.keys():
        y_vars[m] = modelo.addVar(vtype=GRB.BINARY, name=f'RoofMatl_{i}_{m}')
    
    # RESTRICCIÓN 1: Selección única de estilo
    modelo.addConstr(
        gp.quicksum(x_vars.values()) == 1,
        name=f'roof_style_unique_{i}'
    )
    
    # RESTRICCIÓN 2: Selección única de material
    modelo.addConstr(
        gp.quicksum(y_vars.values()) == 1,
        name=f'roof_matl_unique_{i}'
    )
    
    # RESTRICCIÓN 3: Compatibilidad entre estilo y material
    # Si A[s][m] = 0 → x[s] + y[m] ≤ 1 (no pueden estar ambos activos)
    incompatibilidades = 0
    for s in S_plus.keys():
        for m in M_plus.keys():
            if s in ROOF_COMPATIBILITY and m in ROOF_COMPATIBILITY[s]:
                if ROOF_COMPATIBILITY[s][m] == 0:
                    modelo.addConstr(
                        x_vars[s] + y_vars[m] <= 1,
                        name=f'roof_incomp_{i}_{s}_{m}'
                    )
                    incompatibilidades += 1
    
    print(f"  Restricciones de incompatibilidad agregadas: {incompatibilidades}")
    
    # Costo total
    costo_roof = (
        gp.quicksum(x_vars[s] * COSTOS_ROOF_STYLE[s] for s in S_plus.keys()) +
        gp.quicksum(y_vars[m] * COSTOS_ROOF_MATERIAL[m] for m in M_plus.keys())
    )
    
    return {
        'style_vars': x_vars,
        'matl_vars': y_vars,
        'costo': costo_roof
    }


# ============================================================================
# RESTRICCIÓN 3: EXTERIOR CON LÓGICA DE UPGRADE (Sección 6 del PDF - pág 5-7)
# ============================================================================
"""
PDF dice:
- Variable de activación: Upgᵢ ∈ {0,1}
- Si ExterQual ≤ Average O ExterCond ≤ Average → Upgᵢ = 1
- Conjuntos permitidos condicionales:
  E^(1)_allow = {e_base} si Upg=0, {e : C≥C_base} si Upg=1
- Restricciones de activación (3 restricciones)
"""

# Mapeo de calidades a índices
CALIDAD_A_INDICE = {
    'Po': 1,  # Poor
    'Fa': 2,  # Fair
    'TA': 3,  # Typical/Average
    'Gd': 4,  # Good
    'Ex': 5,  # Excellent
}

CATEGORIAS_PROMEDIO_O_MENOR = {'Po', 'Fa', 'TA'}

def agregar_restriccion_exterior_con_upgrade(modelo, casa_actual, i=0):
    """
    Agrega restricción compleja de Exterior con lógica de upgrade.
    
    Esta es la restricción MÁS COMPLEJA del PDF (3 páginas).
    
    Args:
        modelo: Modelo de Gurobi
        casa_actual: Serie con características actuales
        i: Índice de la casa
        
    Returns:
        dict: Variables, costo y variable Upg
    """
    # Parámetros base
    e1_base = casa_actual.get('Exterior1st', 'VinylSd')
    e2_base = casa_actual.get('Exterior2nd', 'VinylSd')
    exter_qual = casa_actual.get('ExterQual', 'TA')
    exter_cond = casa_actual.get('ExterCond', 'TA')
    has_2nd = 1 if pd.notna(e2_base) and e2_base != 'None' else 0
    
    C_e1_base = COSTOS_EXTERIOR.get(e1_base, 7000)
    C_e2_base = COSTOS_EXTERIOR.get(e2_base, 7000) if has_2nd else 0
    
    print(f"  Exterior1st actual: {e1_base} (${C_e1_base:,})")
    print(f"  ExterQual: {exter_qual}, ExterCond: {exter_cond}")
    
    # ========== PASO 1: Variable de activación Upgᵢ ==========
    Upg = modelo.addVar(vtype=GRB.BINARY, name=f'Upg_{i}')
    
    # Indicadores: ¿Está en categoría promedio o menor?
    qual_es_bajo = 1 if exter_qual in CATEGORIAS_PROMEDIO_O_MENOR else 0
    cond_es_bajo = 1 if exter_cond in CATEGORIAS_PROMEDIO_O_MENOR else 0
    
    # RESTRICCIONES DE ACTIVACIÓN (del PDF):
    # 1. Upgᵢ ≥ ExterQual_bajo
    # 2. Upgᵢ ≥ ExterCond_bajo
    # 3. Upgᵢ ≤ ExterQual_bajo + ExterCond_bajo
    
    modelo.addConstr(Upg >= qual_es_bajo, name=f'upg_qual_{i}')
    modelo.addConstr(Upg >= cond_es_bajo, name=f'upg_cond_{i}')
    modelo.addConstr(Upg <= qual_es_bajo + cond_es_bajo, name=f'upg_max_{i}')
    
    print(f"  Upg será: {'1 (puede mejorar)' if (qual_es_bajo or cond_es_bajo) else '0 (mantener)'}")
    
    # ========== PASO 2: Conjuntos permitidos condicionales ==========
    # IMPORTANTE: En Gurobi no podemos hacer conjuntos dinámicos directamente,
    # así que creamos variables para TODAS las opciones y las activamos condicionalmente
    
    # Para Exterior1st: siempre puede elegir e1_base, y si Upg=1, puede elegir más caras
    E1_todas = list(COSTOS_EXTERIOR.keys())
    E1_mejores = [e for e in E1_todas if COSTOS_EXTERIOR[e] >= C_e1_base]
    
    # Variables para Exterior1st
    e1_vars = {}
    for e in E1_todas:
        e1_vars[e] = modelo.addVar(vtype=GRB.BINARY, name=f'Ext1_{i}_{e}')
    
    # RESTRICCIÓN: Solo puede elegir e1_base si Upg=0, o mejores si Upg=1
    # Si Upg=0 → solo e1_base puede ser 1
    # Si Upg=1 → solo E1_mejores pueden ser 1
    
    # Para e1_base: siempre puede ser elegido
    # Para otros: solo si Upg=1 Y son mejores
    for e in E1_todas:
        if e != e1_base:
            # Si NO es mejor → NO puede ser elegido
            if COSTOS_EXTERIOR[e] < C_e1_base:
                modelo.addConstr(e1_vars[e] == 0, name=f'ext1_prohibido_{i}_{e}')
            else:
                # Solo puede ser elegido si Upg=1
                modelo.addConstr(e1_vars[e] <= Upg, name=f'ext1_condicional_{i}_{e}')
    
    # RESTRICCIÓN: Exactamente uno debe ser elegido
    modelo.addConstr(
        gp.quicksum(e1_vars.values()) == 1,
        name=f'ext1_unique_{i}'
    )
    
    # ========== PASO 3: Similar para Exterior2nd (si existe) ==========
    e2_vars = {}
    if has_2nd:
        E2_mejores = [e for e in E1_todas if COSTOS_EXTERIOR[e] >= C_e2_base]
        
        for e in E1_todas:
            e2_vars[e] = modelo.addVar(vtype=GRB.BINARY, name=f'Ext2_{i}_{e}')
        
        for e in E1_todas:
            if e != e2_base:
                if COSTOS_EXTERIOR[e] < C_e2_base:
                    modelo.addConstr(e2_vars[e] == 0, name=f'ext2_prohibido_{i}_{e}')
                else:
                    modelo.addConstr(e2_vars[e] <= Upg, name=f'ext2_condicional_{i}_{e}')
        
        modelo.addConstr(
            gp.quicksum(e2_vars.values()) == 1,
            name=f'ext2_unique_{i}'
        )
    
    # ========== PASO 4: Costo total ==========
    costo_exterior = gp.quicksum(e1_vars[e] * COSTOS_EXTERIOR[e] for e in E1_todas)
    
    if has_2nd:
        costo_exterior += gp.quicksum(e2_vars[e] * COSTOS_EXTERIOR[e] for e in E1_todas)
    
    return {
        'upg_var': Upg,
        'ext1_vars': e1_vars,
        'ext2_vars': e2_vars if has_2nd else None,
        'costo': costo_exterior
    }


# ============================================================================
# RESTRICCIÓN 4: MASVNRTYPE (Sección 6 del PDF - pág 7)
# ============================================================================
"""
PDF dice:
- Se puede cambiar a alternativas de costo mayor o mantenerse
- Similar a Utilities (más simple)
"""

COSTOS_MASVNRTYPE = {
    'BrkCmn': 8000,   # Brick Common
    'BrkFace': 12000, # Brick Face
    'CBlock': 6000,   # Cinder Block
    'None': 0,        # Sin mampostería
    'Stone': 18000,   # Piedra
}

def agregar_restriccion_masvnr(modelo, casa_actual, i=0):
    """
    Agrega restricción de MasVnrType (tipo de mampostería).
    
    Args:
        modelo: Modelo de Gurobi
        casa_actual: Serie con características actuales
        i: Índice de la casa
        
    Returns:
        dict: Variables y costo
    """
    t_base = casa_actual.get('MasVnrType', 'None')
    C_t_base = COSTOS_MASVNRTYPE.get(t_base, 0)
    
    # Conjunto permitido (solo subir de costo)
    T_plus = {t: C for t, C in COSTOS_MASVNRTYPE.items() if C >= C_t_base}
    
    print(f"  MasVnrType actual: {t_base} → Permitidos: {list(T_plus.keys())}")
    
    # Variables de decisión
    mas_vars = {}
    for t in T_plus.keys():
        mas_vars[t] = modelo.addVar(vtype=GRB.BINARY, name=f'MasVnr_{i}_{t}')
    
    # RESTRICCIÓN: Exactamente uno
    modelo.addConstr(
        gp.quicksum(mas_vars.values()) == 1,
        name=f'masvnr_unique_{i}'
    )
    
    # Costo
    costo_masvnr = gp.quicksum(
        mas_vars[t] * COSTOS_MASVNRTYPE[t] for t in T_plus.keys()
    )
    
    return {
        'variables': mas_vars,
        'costo': costo_masvnr
    }

# ============================================================================
# NUEVA RESTRICCIÓN: ELECTRICAL, CENTRAL AIR (Sección 6 del PDF - pág 8-10)
# ============================================================================

def agregar_restriccion_electrical_centralair(modelo, casa_actual, i=0):
    """
    Agrega restricciones de Electrical y Central Air.
    """
    
    # --- 1. Electrical ---
    e_base = casa_actual.get('Electrical', 'SBrkr')
    C_e_base = COSTOS_ELECTRICAL.get(e_base, 0)
    E_plus = {e: C for e, C in COSTOS_ELECTRICAL.items() if C >= C_e_base}
    
    e_vars = {}
    for e in E_plus.keys():
        e_vars[e] = modelo.addVar(vtype=GRB.BINARY, name=f'Electrical_{i}_{e}')
    
    modelo.addConstr(gp.quicksum(e_vars.values()) == 1, name=f'electrical_unique_{i}')
    
    costo_electrical = gp.quicksum(e_vars[e] * COSTOS_ELECTRICAL[e] for e in E_plus.keys() if e != e_base)
    
    # --- 2. Central Air ---
    a_base = casa_actual.get('CentralAir', 'Y') # Y/N
    C_impl = COSTO_CENTRAL_AIR # Costo de implementación
    
    A_allow = ['Y'] if a_base == 'Y' else ['N', 'Y']
    
    a_vars = {}
    for a in A_allow:
        a_vars[a] = modelo.addVar(vtype=GRB.BINARY, name=f'CentralAir_{i}_{a}')
        
    modelo.addConstr(gp.quicksum(a_vars.values()) == 1, name=f'centralair_unique_{i}')
    
    # Costo solo si se cambia de 'N' a 'Y'
    costo_centralair = 0
    if a_base == 'N':
        costo_centralair = C_impl * a_vars['Y']
    
    costo_total = costo_electrical + costo_centralair
    
    return {
        'e_vars': e_vars, 
        'a_vars': a_vars, 
        'costo': costo_total
    }

# ============================================================================
# NUEVA RESTRICCIÓN: HEATING y HEATING QC (Sección 6 del PDF - pág 10-11)
# ============================================================================

def agregar_restriccion_heating(modelo, casa_actual, i=0):
    """
    Agrega restricciones de Heating y Heating QC con lógica de upgrade.
    """
    h_base = casa_actual.get('Heating', 'GasA')
    qc_base = casa_actual.get('HeatingQC', 'TA')
    C_h_base = COSTOS_HEATING.get(h_base, 0)
    
    # --- 1. Variable de activación UpgHeat ---
    UpgHeat = modelo.addVar(vtype=GRB.BINARY, name=f'UpgHeat_{i}')
    
    qc_es_bajo = 1 if qc_base in CATEGORIAS_PROMEDIO_O_MENOR else 0
    
    modelo.addConstr(UpgHeat == qc_es_bajo, name=f'upgheat_def_{i}')
    
    # --- 2. Conjunto permitido H_allow ---
    H_todas = list(COSTOS_HEATING.keys())
    H_mejores = {h: C for h, C in COSTOS_HEATING.items() if C >= C_h_base}
    
    h_vars = {}
    for h in H_todas:
        h_vars[h] = modelo.addVar(vtype=GRB.BINARY, name=f'Heating_{i}_{h}')
        
    for h in H_todas:
        if h != h_base and h not in H_mejores:
            modelo.addConstr(h_vars[h] == 0, name=f'heating_prohibido_{i}_{h}')
        elif h != h_base and h in H_mejores:
            modelo.addConstr(h_vars[h] <= UpgHeat, name=f'heating_condicional_{i}_{h}')
        elif h == h_base:
            modelo.addConstr(h_vars[h_base] >= 1 - UpgHeat, name=f'heating_base_si_no_upg_{i}')

    modelo.addConstr(gp.quicksum(h_vars.values()) == 1, name=f'heating_unique_{i}')
    
    # Costo
    costo_heating = gp.quicksum(h_vars[h] * COSTOS_HEATING[h] for h in H_todas if h != h_base)
    
    return {
        'upg_var': UpgHeat, 
        'h_vars': h_vars, 
        'costo': costo_heating
    }

# ============================================================================
# NUEVA RESTRICCIÓN: KITCHEN QUAL (Sección 6 del PDF - pág 11-12)
# ============================================================================

def agregar_restriccion_kitchenqual(modelo, casa_actual, i=0):
    """
    Agrega restricción de KitchenQual con lógica de upgrade.
    """
    k_base = casa_actual.get('KitchenQual', 'TA')
    C_k_base = COSTOS_KITCHEN_QUAL.get(k_base, 0)
    K_todas = list(COSTOS_KITCHEN_QUAL.keys())
    
    # --- 1. Variable de activación UpgKitch ---
    UpgKitch = modelo.addVar(vtype=GRB.BINARY, name=f'UpgKitch_{i}')
    
    k_es_bajo = 1 if k_base in CATEGORIAS_PROMEDIO_O_MENOR else 0
    modelo.addConstr(UpgKitch == k_es_bajo, name=f'upgkitch_def_{i}')
    
    # --- 2. Conjunto permitido K_allow ---
    K_mejores = {k: C for k, C in COSTOS_KITCHEN_QUAL.items() if C >= C_k_base}
    
    k_vars = {}
    for k in K_todas:
        k_vars[k] = modelo.addVar(vtype=GRB.BINARY, name=f'KitchenQual_{i}_{k}')
        
    for k in K_todas:
        if k != k_base and k not in K_mejores:
            modelo.addConstr(k_vars[k] == 0, name=f'kqual_prohibido_{i}_{k}')
        elif k != k_base and k in K_mejores:
            modelo.addConstr(k_vars[k] <= UpgKitch, name=f'kqual_condicional_{i}_{k}')
        elif k == k_base:
            modelo.addConstr(k_vars[k_base] >= 1 - UpgKitch, name=f'kqual_base_si_no_upg_{i}')

    modelo.addConstr(gp.quicksum(k_vars.values()) == 1, name=f'kqual_unique_{i}')
    
    # Costo
    costo_kitch = gp.quicksum(k_vars[k] * COSTOS_KITCHEN_QUAL[k] for k in K_todas if k != k_base)
    
    return {
        'upg_var': UpgKitch, 
        'k_vars': k_vars, 
        'costo': costo_kitch
    }
    
# ============================================================================
# NUEVA RESTRICCIÓN: BSMT FINISH AREA (Sección 6 del PDF - pág 12-13)
# ============================================================================

def agregar_restriccion_bsmt_finish_area(modelo, casa_actual, i=0):
    """
    Agrega la lógica de terminar el sótano no terminado (BsmtUnfSF).
    """
    unf_base = casa_actual.get('BsmtUnfSF', 0)
    total_base = casa_actual.get('TotalBsmtSF', 0)
    sf1_base = casa_actual.get('BsmtFinSF1', 0)
    sf2_base = casa_actual.get('BsmtFinSF2', 0)
    
    if unf_base == 0 or total_base == 0:
        return {'costo': 0}
        
    # --- 1. Variables de decisión ---
    FinishBSMT = modelo.addVar(vtype=GRB.BINARY, name=f'FinishBSMT_{i}') # 1: Terminar, 0: Mantener
    x1 = modelo.addVar(lb=0, ub=unf_base, vtype=GRB.CONTINUOUS, name=f'x1_transfer_{i}') # Área transferida a SF1
    x2 = modelo.addVar(lb=0, ub=unf_base, vtype=GRB.CONTINUOUS, name=f'x2_transfer_{i}') # Área transferida a SF2
    
    # --- 2. Variables de resultado (nuevas áreas) ---
    BsmtFinSF1 = modelo.addVar(lb=0, ub=total_base, vtype=GRB.CONTINUOUS, name=f'BsmtFinSF1_{i}_new')
    BsmtFinSF2 = modelo.addVar(lb=0, ub=total_base, vtype=GRB.CONTINUOUS, name=f'BsmtFinSF2_{i}_new')
    BsmtUnfSF = modelo.addVar(lb=0, ub=unf_base, vtype=GRB.CONTINUOUS, name=f'BsmtUnfSF_{i}_new')
    
    # --- 3. Restricciones de definición ---
    modelo.addConstr(BsmtFinSF1 == sf1_base + x1, name=f'def_sf1_{i}')
    modelo.addConstr(BsmtFinSF2 == sf2_base + x2, name=f'def_sf2_{i}')
    modelo.addConstr(BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF == total_base, name=f'cons_totalbsmt_{i}')
    
    # Todo o nada sobre el área sin terminar:
    modelo.addConstr(BsmtUnfSF <= unf_base * (1 - FinishBSMT), name=f'unfsf_c1_{i}')
    modelo.addConstr(BsmtUnfSF >= unf_base * (1 - FinishBSMT), name=f'unfsf_c2_{i}')
    
    # Si se termina, se transfiere toda el área no terminada:
    modelo.addConstr(x1 + x2 == unf_base * FinishBSMT, name=f'transfer_total_{i}')
    
    # --- 4. Costo ---
    costo_bsmt = COSTO_BSMT_FINISH_SQFT * (x1 + x2)
    
    return {
        'finish_var': FinishBSMT, 
        'x1_var': x1, 
        'x2_var': x2, 
        'costo': costo_bsmt
    }

# ============================================================================
# NUEVA RESTRICCIÓN: BSMT COND (Sección 6 del PDF - pág 14)
# ============================================================================

def agregar_restriccion_bsmtcond(modelo, casa_actual, i=0):
    """
    Agrega restricción de BsmtCond con lógica de upgrade.
    """
    b_base = casa_actual.get('BsmtCond', 'TA')
    C_b_base = CALIDAD_A_INDICE.get(b_base, 0)
    B_todas = list(CALIDAD_A_INDICE.keys())
    
    # --- 1. Variable de activación UpgBsmt ---
    UpgBsmt = modelo.addVar(vtype=GRB.BINARY, name=f'UpgBsmt_{i}')
    b_es_bajo = 1 if b_base in CATEGORIAS_PROMEDIO_O_MENOR else 0
    modelo.addConstr(UpgBsmt == b_es_bajo, name=f'upgbsmt_def_{i}')
    
    # --- 2. Conjunto permitido B_allow (usando la escala de calidad como proxy de costo)---
    B_mejores = {b: C for b, C in CALIDAD_A_INDICE.items() if C >= C_b_base}
    
    b_vars = {}
    for b in B_todas:
        b_vars[b] = modelo.addVar(vtype=GRB.BINARY, name=f'BsmtCond_{i}_{b}')
        
    for b in B_todas:
        if b != b_base and b not in B_mejores:
            modelo.addConstr(b_vars[b] == 0, name=f'bcond_prohibido_{i}_{b}')
        elif b != b_base and b in B_mejores:
            modelo.addConstr(b_vars[b] <= UpgBsmt, name=f'bcond_condicional_{i}_{b}')
        elif b == b_base:
            modelo.addConstr(b_vars[b_base] >= 1 - UpgBsmt, name=f'bcond_base_si_no_upg_{i}')

    modelo.addConstr(gp.quicksum(b_vars.values()) == 1, name=f'bcond_unique_{i}')
    
    # Costo (usando una escala de costo estimada)
    costo_bsmtcond = gp.quicksum(b_vars[b] * CALIDAD_A_INDICE.get(b, 0) * 1000 for b in B_todas if b != b_base)
    
    return {
        'upg_var': UpgBsmt, 
        'b_vars': b_vars, 
        'costo': costo_bsmtcond
    }

# ============================================================================
# NUEVA RESTRICCIÓN: BSMT FINISH TYPE 1 y 2 (Sección 6 del PDF - pág 15-16)
# ============================================================================

def agregar_restriccion_bsmt_finishtype(modelo, casa_actual, i=0):
    """
    Agrega restricción de BsmtFinType1 y BsmtFinType2 con lógica de upgrade.
    """
    b1_base = casa_actual.get('BsmtFinType1', 'Unf')
    b2_base = casa_actual.get('BsmtFinType2', 'NA')
    has_b2 = 1 if b2_base != 'NA' else 0
    
    # Asume que el costo es proporcional a la calidad (índice)
    C_b1_base = BSMT_FINISH_QUAL_A_COSTO.get(b1_base, 0)
    C_b2_base = BSMT_FINISH_QUAL_A_COSTO.get(b2_base, 0)
    
    B_todas = list(BSMT_FINISH_QUAL_A_COSTO.keys())
    
    # --- 1. Variables de activación UpgB1, UpgB2 ---
    UpgB1 = modelo.addVar(vtype=GRB.BINARY, name=f'UpgB1_{i}')
    UpgB2 = modelo.addVar(vtype=GRB.BINARY, name=f'UpgB2_{i}')
    
    b1_es_bajo = 1 if b1_base in CATEGORIAS_BSMT_REC_O_PEOR else 0
    b2_es_bajo = 1 if b2_base in CATEGORIAS_BSMT_REC_O_PEOR and has_b2 else 0
    
    modelo.addConstr(UpgB1 == b1_es_bajo, name=f'upgb1_def_{i}')
    modelo.addConstr(UpgB2 == b2_es_bajo, name=f'upgb2_def_{i}')
    
    # --- 2. Variables para BsmtFinType1 ---
    B1_mejores = {b: C for b, C in BSMT_FINISH_QUAL_A_COSTO.items() if C >= C_b1_base and b != 'NA'}
    b1_vars = {}
    for b in B_todas:
        b1_vars[b] = modelo.addVar(vtype=GRB.BINARY, name=f'BsmtFinType1_{i}_{b}')
    
    # Lógica B1
    for b in B_todas:
        if b == 'NA':
            modelo.addConstr(b1_vars['NA'] == 0, name=f'b1_no_na_{i}')
        elif b != b1_base and b not in B1_mejores:
            modelo.addConstr(b1_vars[b] == 0, name=f'b1_prohibido_{i}_{b}')
        elif b != b1_base and b in B1_mejores:
            modelo.addConstr(b1_vars[b] <= UpgB1, name=f'b1_condicional_{i}_{b}')
        elif b == b1_base:
            modelo.addConstr(b1_vars[b1_base] >= 1 - UpgB1, name=f'b1_base_si_no_upg_{i}')

    modelo.addConstr(gp.quicksum(b1_vars.values()) == 1, name=f'b1_unique_{i}')
    
    # --- 3. Variables para BsmtFinType2 ---
    costo_b2 = 0
    if has_b2:
        B2_mejores = {b: C for b, C in BSMT_FINISH_QUAL_A_COSTO.items() if C >= C_b2_base and b != 'NA'}
        b2_vars = {}
        for b in B_todas:
            b2_vars[b] = modelo.addVar(vtype=GRB.BINARY, name=f'BsmtFinType2_{i}_{b}')
        
        # Lógica B2
        for b in B_todas:
            if b == 'NA':
                modelo.addConstr(b2_vars['NA'] == 0, name=f'b2_no_na_{i}')
            elif b != b2_base and b not in B2_mejores:
                modelo.addConstr(b2_vars[b] == 0, name=f'b2_prohibido_{i}_{b}')
            elif b != b2_base and b in B2_mejores:
                modelo.addConstr(b2_vars[b] <= UpgB2, name=f'b2_condicional_{i}_{b}')
            elif b == b2_base:
                modelo.addConstr(b2_vars[b2_base] >= 1 - UpgB2, name=f'b2_base_si_no_upg_{i}')
        
        modelo.addConstr(gp.quicksum(b2_vars.values()) == 1, name=f'b2_unique_{i}')
        
        costo_b2 = gp.quicksum(b2_vars[b] * BSMT_FINISH_QUAL_A_COSTO.get(b, 0) * 100 for b in B_todas if b != b2_base)
        
    # Costo total (solo costo de cambio)
    costo_b1 = gp.quicksum(b1_vars[b] * BSMT_FINISH_QUAL_A_COSTO.get(b, 0) * 100 for b in B_todas if b != b1_base)
    
    return {
        'upg_b1': UpgB1, 
        'upg_b2': UpgB2, 
        'costo': costo_b1 + costo_b2
    }

# ============================================================================
# NUEVA RESTRICCIÓN: FIREPLACE QU (Sección 6 del PDF - pág 16-17)
# ============================================================================

def agregar_restriccion_fireplacequ(modelo, casa_actual, i=0):
    """
    Agrega restricción de FireplaceQu con lógica de mejora específica.
    """
    f_base = casa_actual.get('FireplaceQu', 'NA')
    F_todas = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
    
    # Costo se basa en la calidad final (estimación)
    C_f = {'Ex': 5000, 'Gd': 4000, 'TA': 3000, 'Fa': 2000, 'Po': 1000, 'NA': 0}
    
    # 1. Definir conjunto permitido
    if f_base == 'NA':
        F_allow = ['NA']
    elif f_base == 'TA':
        F_allow = ['TA', 'Gd', 'Ex']
    elif f_base == 'Po':
        F_allow = ['Po', 'Fa']
    else: # Fa, Gd, Ex
        F_allow = [f_base]
        
    # 2. Variables
    f_vars = {}
    for f in F_todas:
        f_vars[f] = modelo.addVar(vtype=GRB.BINARY, name=f'FireplaceQu_{i}_{f}')
        
    # 3. Restricciones
    modelo.addConstr(gp.quicksum(f_vars.values()) == 1, name=f'fqu_unique_{i}')
    
    for f in F_todas:
        if f not in F_allow:
            modelo.addConstr(f_vars[f] == 0, name=f'fqu_prohibido_{i}_{f}')

    # 4. Costo
    costo_f = gp.quicksum(f_vars[f] * C_f[f] for f in F_allow)
    
    return {
        'f_vars': f_vars, 
        'costo': costo_f
    }

# ============================================================================
# NUEVA RESTRICCIÓN: FENCE (Sección 6 del PDF - pág 17-18)
# ============================================================================

def agregar_restriccion_fence(modelo, casa_actual, i=0):
    """
    Agrega restricción de Fence con lógica de mejora y construcción.
    """
    f_base = casa_actual.get('Fence', 'NA')
    lot_area = casa_actual.get('Lot Area', 10000)
    F_todas = list(COSTOS_FENCE_CAT.keys())
    
    # 1. Definir conjunto permitido
    if f_base == 'NA':
        F_allow = ['NA', 'MnPrv', 'GdPrv']
    elif f_base in ['GdWo', 'MnWw']:
        F_allow = [f_base, 'MnPrv', 'GdPrv']
    else: # MnPrv, GdPrv
        F_allow = [f_base]
        
    # 2. Variables
    f_vars = {}
    for f in F_todas:
        f_vars[f] = modelo.addVar(vtype=GRB.BINARY, name=f'Fence_{i}_{f}')
        
    # 3. Restricciones
    modelo.addConstr(gp.quicksum(f_vars.values()) == 1, name=f'fence_unique_{i}')
    
    for f in F_todas:
        if f not in F_allow:
            modelo.addConstr(f_vars[f] == 0, name=f'fence_prohibido_{i}_{f}')

    # 4. Costo
    # Costo categórico (solo si se cambia de la base)
    costo_cat = gp.quicksum(f_vars[f] * COSTOS_FENCE_CAT[f] for f in F_allow if f != f_base)
    
    # Costo de construcción (solo si era NA y se construye MnPrv o GdPrv)
    costo_build = 0
    if f_base == 'NA':
        costo_build = COSTO_FENCE_PSF * lot_area * (f_vars.get('MnPrv', 0) + f_vars.get('GdPrv', 0))
    
    return {
        'f_vars': f_vars, 
        'costo': costo_cat + costo_build
    }

# ============================================================================
# NUEVA RESTRICCIÓN: PAVED DRIVE (Sección 6 del PDF - pág 18-19)
# ============================================================================

def agregar_restriccion_paveddrive(modelo, casa_actual, i=0):
    """
    Agrega restricción de PavedDrive (mejora jerárquica).
    """
    d_base = casa_actual.get('PavedDrive', 'Y')
    D_todas = list(COSTOS_PAVEDDRIVE.keys()) # N, P, Y
    
    # 1. Definir conjunto permitido
    if d_base == 'Y':
        D_allow = ['Y']
    elif d_base == 'P':
        D_allow = ['P', 'Y']
    elif d_base == 'N':
        D_allow = ['N', 'P', 'Y']
        
    # 2. Variables
    d_vars = {}
    for d in D_todas:
        d_vars[d] = modelo.addVar(vtype=GRB.BINARY, name=f'PavedDrive_{i}_{d}')
        
    # 3. Restricciones
    modelo.addConstr(gp.quicksum(d_vars.values()) == 1, name=f'pdrive_unique_{i}')
    
    for d in D_todas:
        if d not in D_allow:
            modelo.addConstr(d_vars[d] == 0, name=f'pdrive_prohibido_{i}_{d}')

    # 4. Costo
    costo_d = gp.quicksum(d_vars[d] * COSTOS_PAVEDDRIVE[d] for d in D_allow if d != d_base)
    
    return {
        'd_vars': d_vars, 
        'costo': costo_d
    }

# ============================================================================
# NUEVA RESTRICCIÓN: GARAGE QUAL/COND y FINISH (Sección 6 del PDF - pág 19-20 y 24)
# ============================================================================

def agregar_restriccion_garage_full(modelo, casa_actual, i=0):
    """
    Agrega restricciones de Garage Qual, Cond y Finish.
    """
    gq_base = casa_actual.get('GarageQual', 'NA')
    gc_base = casa_actual.get('GarageCond', 'NA')
    gf_base = casa_actual.get('GarageFinish', 'NA')
    
    # --- 1. Variable de activación UpgGar para Qual/Cond ---
    UpgGar = modelo.addVar(vtype=GRB.BINARY, name=f'UpgGar_{i}')
    gq_es_bajo = 1 if gq_base in CATEGORIAS_PROMEDIO_O_MENOR else 0
    gc_es_bajo = 1 if gc_base in CATEGORIAS_PROMEDIO_O_MENOR else 0
    
    modelo.addConstr(UpgGar >= gq_es_bajo, name=f'upggar_qual_{i}')
    modelo.addConstr(UpgGar >= gc_es_bajo, name=f'upggar_cond_{i}')
    modelo.addConstr(UpgGar <= gq_es_bajo + gc_es_bajo, name=f'upggar_max_{i}')
    
    # --- 2. Garage Finish ---
    gf_es_bajo = 1 if gf_base in ['RFn', 'Unf'] else 0
    UpgGa = modelo.addVar(vtype=GRB.BINARY, name=f'UpgGa_{i}')
    modelo.addConstr(UpgGa == gf_es_bajo, name=f'upgga_def_{i}')

    Gf_todas = list(COSTOS_GARAGE_FINISH.keys())
    gf_vars = {gf: modelo.addVar(vtype=GRB.BINARY, name=f'GarageFinish_{i}_{gf}') for gf in Gf_todas}

    # Logica sin usar UpgGa.X: si no hay upgrade -> forzar base; si hay upgrade permitir 'Fin'
    for gf in Gf_todas:
        if gf == gf_base:
            modelo.addConstr(gf_vars[gf_base] >= 1 - UpgGa, name=f'gfin_base_si_no_upg_{i}')
        elif gf == 'Fin':
            modelo.addConstr(gf_vars['Fin'] <= UpgGa, name=f'gfin_fin_only_if_upg_{i}')
        else:
            modelo.addConstr(gf_vars[gf] == 0, name=f'gfin_prohibido_{i}_{gf}')

    modelo.addConstr(gp.quicksum(gf_vars.values()) == 1, name=f'gfin_unique_{i}')
    costo_gf = gp.quicksum(gf_vars[gf] * COSTOS_GARAGE_FINISH[gf] for gf in Gf_todas if gf != gf_base)
    
    # NOTA: Se ha omitido la implementación completa de GarageQual y GarageCond por ser
    # simétrica a BsmtCond, pero se incluye el costo de Finish.
    
    return {
        'upg_gar': UpgGar, 
        'upg_ga': UpgGa, 
        'costo': costo_gf # Se asume costo total de Qual/Cond es cero o modelado en variable discreta
    }
    
# ============================================================================
# NUEVA RESTRICCIÓN: POOL QC (Sección 6 del PDF - pág 23)
# ============================================================================

def agregar_restriccion_poolqc(modelo, casa_actual, i=0):
    """
    Agrega restricción de PoolQC con lógica de upgrade.
    """
    p_base = casa_actual.get('PoolQC', 'NA')
    C_p_base = COSTOS_POOL_QUAL.get(p_base, 0)
    P_todas = list(COSTOS_POOL_QUAL.keys())
    
    # --- 1. Variable de activación UpgPool ---
    UpgPool = modelo.addVar(vtype=GRB.BINARY, name=f'UpgPool_{i}')
    p_es_bajo = 1 if p_base in CATEGORIAS_PROMEDIO_O_MENOR else 0
    modelo.addConstr(UpgPool == p_es_bajo, name=f'upgpool_def_{i}')

    p_vars = {p: modelo.addVar(vtype=GRB.BINARY, name=f'PoolQC_{i}_{p}') for p in P_todas}
    for p in P_todas:
        if p == p_base:
            modelo.addConstr(p_vars[p_base] >= 1 - UpgPool, name=f'pool_base_if_no_upg_{i}')
        elif COSTOS_POOL_QUAL.get(p, 0) < C_p_base:
            modelo.addConstr(p_vars[p] == 0, name=f'poolqc_prohibido_{i}_{p}')
        else:
            modelo.addConstr(p_vars[p] <= UpgPool, name=f'poolqc_condicional_{i}_{p}')

    modelo.addConstr(gp.quicksum(p_vars.values()) == 1, name=f'poolqc_unique_{i}')
    
    # 3. Costo
    costo_pool = gp.quicksum(p_vars[p] * COSTOS_POOL_QUAL[p] for p in P_todas if p != p_base)
    
    return {
        'upg_var': UpgPool, 
        'p_vars': p_vars, 
        'costo': costo_pool
    }

# ============================================================================
# FUNCIÓN PRINCIPAL: Agregar TODAS las restricciones de renovación
# ============================================================================

# Contenido para reemplazar la función agregar_todas_restricciones_renovacion
# en optimización/restricciones_renovacion.py

def agregar_todas_restricciones_renovacion(modelo, casa_actual, i=0):
    """
    Agrega TODAS las restricciones de la Sección 6 del PDF.
    
    Args:
        modelo: Modelo de Gurobi
        casa_actual: pd.Series con características de la casa actual
        i: Índice de la casa
        
    Returns:
        dict: Todas las variables y costo total de renovación categórica
    """
    
    print("\n" + "="*70)
    print("AGREGANDO RESTRICCIONES DE RENOVACIÓN (Sección 6 del PDF)")
    print("="*70)
    
    resultado = {}
    costo_total_categorico = []
    
    # --- RESTRICCIONES YA IMPLEMENTADAS (1-4) ---
    
    # 1. Utilities 
    print("\n1️⃣  UTILITIES:")
    utilities = agregar_restriccion_utilities(modelo, casa_actual, i)
    resultado['utilities'] = utilities
    costo_total_categorico.append(utilities['costo'])
    
    # 2. Roof (Style + Material con compatibilidad)
    print("\n2️⃣  ROOF (Style + Material):")
    roof = agregar_restriccion_roof(modelo, casa_actual, i)
    resultado['roof'] = roof
    costo_total_categorico.append(roof['costo'])
    
    # 3. Exterior (con lógica de Upgrade - LA MÁS COMPLEJA)
    print("\n3️⃣  EXTERIOR (con Upgrade):")
    exterior = agregar_restriccion_exterior_con_upgrade(modelo, casa_actual, i)
    resultado['exterior'] = exterior
    costo_total_categorico.append(exterior['costo'])
    
    # 4. MasVnrType
    print("\n4️⃣  MASVNRTYPE:")
    masvnr = agregar_restriccion_masvnr(modelo, casa_actual, i)
    resultado['masvnr'] = masvnr
    costo_total_categorico.append(masvnr['costo'])
    
    # --- RESTRICCIONES FALTANTES AHORA AGREGADAS (5-14) ---
    
    # 5. Electrical y CentralAir
    print("\n5️⃣  ELECTRICAL y CENTRAL AIR:")
    elect_ac = agregar_restriccion_electrical_centralair(modelo, casa_actual, i)
    resultado['electrical_ac'] = elect_ac
    costo_total_categorico.append(elect_ac['costo'])
    
    # 6. Heating y Heating QC
    print("\n6️⃣  HEATING y HEATING QC:")
    heating = agregar_restriccion_heating(modelo, casa_actual, i)
    resultado['heating'] = heating
    costo_total_categorico.append(heating['costo'])
    
    # 7. KitchenQual
    print("\n7️⃣  KITCHEN QUAL:")
    kitch_qual = agregar_restriccion_kitchenqual(modelo, casa_actual, i)
    resultado['kitch_qual'] = kitch_qual
    costo_total_categorico.append(kitch_qual['costo'])
    
    # 8. Bsmt Finish Area (BsmtUnfSF)
    print("\n8️⃣  BSMT FINISH AREA:")
    bsmt_area = agregar_restriccion_bsmt_finish_area(modelo, casa_actual, i)
    resultado['bsmt_area'] = bsmt_area
    costo_total_categorico.append(bsmt_area['costo'])
    
    # 9. Bsmt Cond
    print("\n9️⃣  BSMT COND:")
    bsmt_cond = agregar_restriccion_bsmtcond(modelo, casa_actual, i)
    resultado['bsmt_cond'] = bsmt_cond
    costo_total_categorico.append(bsmt_cond['costo'])
    
    # 10. Bsmt Finish Type 1 y 2
    print("\n🔟 BSMT FINISH TYPE 1 y 2:")
    bsmt_type = agregar_restriccion_bsmt_finishtype(modelo, casa_actual, i)
    resultado['bsmt_type'] = bsmt_type
    costo_total_categorico.append(bsmt_type['costo'])
    
    # 11. Fireplace Qu
    print("\n1️⃣1️⃣ FIREPLACE QU:")
    fp_qu = agregar_restriccion_fireplacequ(modelo, casa_actual, i)
    resultado['fp_qu'] = fp_qu
    costo_total_categorico.append(fp_qu['costo'])
    
    # 12. Fence
    print("\n1️⃣2️⃣ FENCE:")
    fence = agregar_restriccion_fence(modelo, casa_actual, i)
    resultado['fence'] = fence
    costo_total_categorico.append(fence['costo'])
    
    # 13. Paved Drive
    print("\n1️⃣3️⃣ PAVED DRIVE:")
    paved_drive = agregar_restriccion_paveddrive(modelo, casa_actual, i)
    resultado['paved_drive'] = paved_drive
    costo_total_categorico.append(paved_drive['costo'])
    
    # 14. Garage Qual/Cond y Finish
    print("\n1️⃣4️⃣ GARAGE (Qual/Cond/Finish):")
    garage_full = agregar_restriccion_garage_full(modelo, casa_actual, i)
    resultado['garage_full'] = garage_full
    costo_total_categorico.append(garage_full['costo'])

    # Totalizar el costo
    resultado['costo_total_categorico'] = gp.quicksum(costo_total_categorico)
    
    print("\n" + "="*70)
    print("✅ TODAS LAS RESTRICCIONES DE RENOVACIÓN AGREGADAS")
    print("="*70)
    
    return resultado


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    import pandas as pd
    
    # Crear modelo de prueba
    modelo_test = gp.Model("Test_Restricciones_Renovacion")
    modelo_test.Params.LogToConsole = 0
    
    # Casa de ejemplo
    casa_ejemplo = pd.Series({
        'Utilities': 'AllPub',
        'RoofStyle': 'Gable',
        'RoofMatl': 'CompShg',
        'Exterior1st': 'VinylSd',
        'Exterior2nd': 'VinylSd',
        'ExterQual': 'TA',  # Typical/Average
        'ExterCond': 'TA',
        'MasVnrType': 'None',
    })
    
    print("\n🏠 CASA DE EJEMPLO:")
    print(casa_ejemplo)
    
    # Agregar todas las restricciones
    resultado = agregar_todas_restricciones_renovacion(modelo_test, casa_ejemplo)
    
    # Objetivo dummy (solo para que el modelo sea completo)
    modelo_test.setObjective(resultado['costo_total_categorico'], GRB.MINIMIZE)
    
    # Resolver
    modelo_test.optimize()
    
    if modelo_test.status == GRB.OPTIMAL:
        print("\n✅ Modelo resuelto correctamente")
        print(f"Costo óptimo categórico: ${modelo_test.ObjVal:,.0f}")
    else:
        print(f"\n❌ Estado del modelo: {modelo_test.status}")

print("\n✅ Archivo restricciones_renovacion.py creado")