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
from config import ROOF_COMPATIBILITY, COSTOS_ROOF_STYLE, COSTOS_ROOF_MATERIAL, COSTOS_EXTERIOR

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
# FUNCIÓN PRINCIPAL: Agregar TODAS las restricciones de renovación
# ============================================================================

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
    import pandas as pd
    
    print("\n" + "="*70)
    print("AGREGANDO RESTRICCIONES DE RENOVACIÓN (Sección 6 del PDF)")
    print("="*70)
    
    resultado = {}
    costo_total_categorico = 0
    
    # 1. Utilities
    print("\n1️⃣  UTILITIES:")
    utilities = agregar_restriccion_utilities(modelo, casa_actual, i)
    resultado['utilities'] = utilities
    costo_total_categorico += utilities['costo']
    
    # 2. Roof (Style + Material con compatibilidad)
    print("\n2️⃣  ROOF (Style + Material):")
    roof = agregar_restriccion_roof(modelo, casa_actual, i)
    resultado['roof'] = roof
    costo_total_categorico += roof['costo']
    
    # 3. Exterior (con lógica de Upgrade - LA MÁS COMPLEJA)
    print("\n3️⃣  EXTERIOR (con Upgrade):")
    exterior = agregar_restriccion_exterior_con_upgrade(modelo, casa_actual, i)
    resultado['exterior'] = exterior
    costo_total_categorico += exterior['costo']
    
    # 4. MasVnrType
    print("\n4️⃣  MASVNRTYPE:")
    masvnr = agregar_restriccion_masvnr(modelo, casa_actual, i)
    resultado['masvnr'] = masvnr
    costo_total_categorico += masvnr['costo']
    
    print("\n" + "="*70)
    print("✅ TODAS LAS RESTRICCIONES DE RENOVACIÓN AGREGADAS")
    print("="*70)
    
    resultado['costo_total_categorico'] = costo_total_categorico
    
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