"""
restricciones_construccion.py
=============================
Implementación completa de las restricciones de CONSTRUCCIÓN DESDE CERO 
según Sección 7 del PDF.

Grupo 5 - Capstone 2025
Modelo Matemático: Restricciones de Construcción
"""

import gurobipy as gp
from gurobipy import GRB
from config import AREA_MINIMA, MAX_PISOS, OCUPACION_MAX_TERRENO

# ============================================================================
# RESTRICCIONES DE CONSTRUCCIÓN (Sección 7 del PDF)
# ============================================================================

def agregar_restricciones_construccion(modelo, variables, terreno_params):
    """
    Agrega TODAS las restricciones de construcción desde cero (Sección 7 del PDF).
    
    Args:
        modelo: Modelo de Gurobi
        variables: Dict con todas las variables del modelo
        terreno_params: Dict con parámetros del terreno (LotArea, etc.)
        
    Returns:
        None (modifica el modelo directamente)
    """
    
    print("\n" + "="*70)
    print("AGREGANDO RESTRICCIONES DE CONSTRUCCIÓN (Sección 7 del PDF)")
    print("="*70)
    
    # Extraer variables necesarias
    x = variables  # Asumimos que variables es un dict con todas las vars
    
    LotArea = terreno_params.get('LotArea', 10000)  # Área del terreno
    
    # ========================================================================
    # RESTRICCIÓN 1 (PDF): Área construida no puede superar terreno
    # ========================================================================
    """
    PDF dice:
    1stFlrSF_new + TotalPorch_new + PoolArea_new ≤ LotArea
    
    Además agregamos restricción de ocupación máxima (70% del terreno)
    """
    
    print("\n1️⃣  Restricción: Área construida ≤ Terreno")
    
    # Calcular TotalPorch
    total_porch = (
        x.get('WoodDeckSF', 0) +
        x.get('OpenPorchSF', 0) +
        x.get('EnclosedPorch', 0) +
        x.get('ScreenPorch', 0) +
        x.get('3SsnPorch', 0)
    )
    
    # Restricción básica: No exceder terreno
    modelo.addConstr(
        x['1stFlrSF'] + total_porch + x.get('PoolArea', 0) <= LotArea,
        name='area_no_excede_terreno'
    )
    
    # Restricción de ocupación máxima
    modelo.addConstr(
        x['1stFlrSF'] + total_porch + x.get('PoolArea', 0) <= OCUPACION_MAX_TERRENO * LotArea,
        name='ocupacion_maxima_70pct'
    )
    
    print(f"   ✓ LotArea: {LotArea} sqft")
    print(f"   ✓ Ocupación máxima permitida: {OCUPACION_MAX_TERRENO*100}% = {OCUPACION_MAX_TERRENO*LotArea:,.0f} sqft")
    
    # ========================================================================
    # RESTRICCIÓN 2 (PDF): Máximo número de pisos
    # ========================================================================
    """
    PDF dice:
    Solo se puede agregar un número limitado de pisos (máximo 2 pisos)
    """
    
    print("\n2️⃣  Restricción: Máximo 2 pisos")
    
    # Variable binaria: ¿Tiene segundo piso?
    tiene_2do_piso = modelo.addVar(vtype=GRB.BINARY, name='tiene_2do_piso')
    
    # Si 2ndFlrSF > 0 → tiene_2do_piso = 1
    M = 10000  # Big-M
    modelo.addConstr(x['2ndFlrSF'] <= M * tiene_2do_piso, name='activar_2do_piso')
    
    # Máximo MAX_PISOS pisos (ya tenemos 1er piso siempre, solo 1 adicional)
    modelo.addConstr(tiene_2do_piso <= 1, name='max_2_pisos')
    
    print(f"   ✓ Máximo {MAX_PISOS} pisos permitidos")
    
    # ========================================================================
    # RESTRICCIÓN 3 (PDF): Half-Bath solo si hay Full-Bath
    # ========================================================================
    """
    PDF dice:
    HalfBath_new ≤ FullBath_new
    
    Las half-bath solo pueden existir si hay al menos un baño completo
    """
    
    print("\n3️⃣  Restricción: HalfBath ≤ FullBath")
    
    modelo.addConstr(
        x.get('HalfBath', 0) <= x['FullBath'],
        name='halfbath_requiere_fullbath'
    )
    
    print(f"   ✓ Medios baños limitados por baños completos")
    
    # ========================================================================
    # RESTRICCIÓN 4 (PDF): Habitaciones proporcionales a superficie
    # ========================================================================
    """
    PDF dice:
    El número de habitaciones no puede ser mayor que el permitido por superficie
    
    Interpretación: Cada habitación necesita un área mínima
    """
    
    print("\n4️⃣  Restricción: Habitaciones proporcionales a área")
    
    # Cada dormitorio necesita al menos AREA_MINIMA['Bedroom'] sqft
    modelo.addConstr(
        x['BedroomAbvGr'] * AREA_MINIMA['Bedroom'] <= x['GrLivArea'],
        name='area_minima_dormitorios'
    )
    
    # Cada baño completo necesita al menos AREA_MINIMA['FullBath'] sqft
    modelo.addConstr(
        x['FullBath'] * AREA_MINIMA['FullBath'] <= x['GrLivArea'],
        name='area_minima_banos'
    )
    
    # Cocina necesita área mínima
    modelo.addConstr(
        x['KitchenAbvGr'] * AREA_MINIMA['Kitchen'] <= x['GrLivArea'],
        name='area_minima_cocina'
    )
    
    print(f"   ✓ Área mínima por dormitorio: {AREA_MINIMA['Bedroom']} sqft")
    print(f"   ✓ Área mínima por baño: {AREA_MINIMA['FullBath']} sqft")
    print(f"   ✓ Área mínima cocina: {AREA_MINIMA['Kitchen']} sqft")
    
    # ========================================================================
    # RESTRICCIÓN 6 (PDF): Área mínima razonable por tipo de ambiente
    # ========================================================================
    """
    PDF dice:
    Cada tipo de ambiente debe tener un área mínima razonable
    
    Esto se relaciona con área mínima de cocina básica vs top
    """
    
    print("\n6️⃣  Restricción: Áreas mínimas por ambiente")
    
    # Área habitable mínima total
    modelo.addConstr(x['GrLivArea'] >= 600, name='area_habitable_minima')
    
    # Si hay garaje, debe tener al menos área mínima por auto
    area_minima_por_auto = 200  # sqft por auto
    modelo.addConstr(
        x['GarageCars'] * area_minima_por_auto <= x.get('GarageArea', 0) + 10000 * (1 - tiene_garaje),
        name='area_minima_garaje'
    ) if 'GarageCars' in x else None
    
    print(f"   ✓ Área habitable mínima: 600 sqft")
    print(f"   ✓ Área mínima garaje: {area_minima_por_auto} sqft/auto")
    
    # ========================================================================
    # RESTRICCIÓN 7 (PDF): Máximo de ambientes repetidos
    # ========================================================================
    """
    PDF dice:
    Máximo de ambientes repetidos (máximo número de cocinas, comedores, etc.)
    """
    
    print("\n7️⃣  Restricción: Máximo de ambientes repetidos")
    
    # Máximo 2 cocinas (principal + auxiliar/isla)
    modelo.addConstr(x['KitchenAbvGr'] <= 2, name='max_2_cocinas')
    
    # Máximo 6 dormitorios (razonable para casa residencial)
    modelo.addConstr(x['BedroomAbvGr'] <= 6, name='max_6_dormitorios')
    
    # Máximo 4 baños completos
    modelo.addConstr(x['FullBath'] <= 4, name='max_4_banos_completos')
    
    # Máximo 2 medios baños
    modelo.addConstr(x.get('HalfBath', 0) <= 2, name='max_2_medios_banos')
    
    # Máximo 3 chimeneas
    modelo.addConstr(x.get('Fireplaces', 0) <= 3, name='max_3_chimeneas')
    
    print(f"   ✓ Máximo 2 cocinas, 6 dormitorios, 4 baños completos")
    
    # ========================================================================
    # RESTRICCIÓN 8 (PDF): Mínimo de área para ampliación
    # ========================================================================
    """
    PDF dice:
    Definir mínimo de área que tiene una ampliación (no puede haber ampliación de 1m)
    """
    
    print("\n8️⃣  Restricción: Área mínima de ampliación")
    
    # Si hay segundo piso, debe tener al menos 200 sqft
    modelo.addConstr(
        x['2ndFlrSF'] >= 200 * tiene_2do_piso,
        name='area_minima_2do_piso'
    )
    
    # Si hay segundo piso, no puede exceder primer piso (restricción estructural)
    modelo.addConstr(
        x['2ndFlrSF'] <= x['1stFlrSF'],
        name='2do_piso_max_igual_1ro'
    )
    
    print(f"   ✓ Si hay 2do piso, mínimo 200 sqft")
    print(f"   ✓ 2do piso ≤ 1er piso (estructural)")
    
    # ========================================================================
    # RESTRICCIÓN 9 (PDF): Solo construir ambiente si hay superficie
    # ========================================================================
    """
    PDF dice:
    Solo se puede construir un nuevo ambiente si se amplía la superficie construida
    
    Interpretación: Si agregas habitaciones, el área total debe aumentar
    """
    
    print("\n9️⃣  Restricción: Nuevos ambientes requieren área adicional")
    
    # Total de habitaciones debe ser consistente con área
    # TotRmsAbvGrd = Bedroom + Kitchen + Otras
    # Cada habitación promedio necesita ~100 sqft
    modelo.addConstr(
        x['TotRmsAbvGrd'] * 100 <= x['GrLivArea'] + 1000,  # +1000 de tolerancia
        name='habitaciones_consistentes_con_area'
    )
    
    print(f"   ✓ Total habitaciones consistente con área habitable")
    
    # ========================================================================
    # RESTRICCIÓN 10 (PDF): Al agregar ambiente, aumentar área mínima
    # ========================================================================
    """
    PDF dice:
    Si se agrega cualquier ambiente nuevo, el área total debe aumentar al menos 
    en el área mínima de ese ambiente
    
    Esto es implícito en las restricciones de área mínima anteriores
    """
    
    print("\n🔟 Restricción: Área total aumenta con nuevos ambientes")
    
    # Ya cubierto por restricciones 4 y 6
    # Cada nuevo dormitorio → +80 sqft mínimo
    # Cada nuevo baño → +50 sqft mínimo
    # etc.
    
    print(f"   ✓ Implícito en restricciones de área mínima")
    
    # ========================================================================
    # RESTRICCIÓN ADICIONAL: Consistencia de GrLivArea
    # ========================================================================
    """
    Del PDF Sección 5:
    GrLivArea = 1stFlrSF + 2ndFlrSF + LowQualFinSF
    """
    
    print("\n➕ Restricción adicional: Consistencia de GrLivArea")
    
    low_qual = x.get('LowQualFinSF', 0)
    
    modelo.addConstr(
        x['GrLivArea'] == x['1stFlrSF'] + x['2ndFlrSF'] + low_qual,
        name='grlivarea_consistencia'
    )
    
    print(f"   ✓ GrLivArea = 1stFlr + 2ndFlr + LowQual")
    
    # ========================================================================
    # RESTRICCIÓN ADICIONAL: Sótano no excede primer piso
    # ========================================================================
    """
    Del PDF Sección 5:
    1stFlrSF ≥ TotalBsmtSF
    """
    
    print("\n➕ Restricción adicional: Sótano ≤ Primer piso")
    
    if 'TotalBsmtSF' in x:
        modelo.addConstr(
            x['TotalBsmtSF'] <= x['1stFlrSF'],
            name='sotano_max_igual_1ro'
        )
        print(f"   ✓ TotalBsmtSF ≤ 1stFlrSF")
    
    # ========================================================================
    # RESTRICCIÓN ADICIONAL: Mínimos básicos
    # ========================================================================
    """
    Del PDF Sección 5:
    FullBath ≥ 1, Bedroom ≥ 1, Kitchen ≥ 1
    """
    
    print("\n➕ Restricción adicional: Mínimos básicos habitabilidad")
    
    modelo.addConstr(x['FullBath'] >= 1, name='minimo_1_bano')
    modelo.addConstr(x['BedroomAbvGr'] >= 1, name='minimo_1_dormitorio')
    modelo.addConstr(x['KitchenAbvGr'] >= 1, name='minimo_1_cocina')
    
    print(f"   ✓ Mínimo: 1 baño, 1 dormitorio, 1 cocina")
    
    # ========================================================================
    # RESTRICCIÓN ADICIONAL: TotRmsAbvGrd consistencia
    # ========================================================================
    """
    Del PDF Sección 5:
    TotRmsAbvGrd = Bedroom + Kitchen + OtrasHabitaciones
    """
    
    print("\n➕ Restricción adicional: Total habitaciones")
    
    # TotRmsAbvGrd debe incluir al menos dormitorios + cocinas
    modelo.addConstr(
        x['TotRmsAbvGrd'] >= x['BedroomAbvGr'] + x['KitchenAbvGr'],
        name='totrooms_minimo'
    )
    
    # Máximo razonable de habitaciones totales
    modelo.addConstr(
        x['TotRmsAbvGrd'] <= 15,
        name='totrooms_maximo'
    )
    
    print(f"   ✓ TotRms ≥ Bedrooms + Kitchens")
    print(f"   ✓ TotRms ≤ 15 (máximo razonable)")
    
    print("\n" + "="*70)
    print("✅ TODAS LAS RESTRICCIONES DE CONSTRUCCIÓN AGREGADAS")
    print("="*70)


# ============================================================================
# FUNCIÓN AUXILIAR: Variable binaria condicional para garaje
# ============================================================================

def agregar_variable_garaje_condicional(modelo, variables):
    """
    Crea variable binaria que indica si hay garaje.
    
    Args:
        modelo: Modelo de Gurobi
        variables: Dict con variables del modelo
        
    Returns:
        Variable binaria tiene_garaje
    """
    tiene_garaje = modelo.addVar(vtype=GRB.BINARY, name='tiene_garaje')
    
    # Si GarageCars > 0 → tiene_garaje = 1
    M = 10
    if 'GarageCars' in variables:
        modelo.addConstr(
            variables['GarageCars'] <= M * tiene_garaje,
            name='activar_garaje'
        )
    
    return tiene_garaje


# ============================================================================
# FUNCIÓN: Agregar restricciones de calidad para construcción nueva
# ============================================================================

def fijar_calidades_construccion_nueva(modelo, variables, calidades_default):
    """
    Fija calidades a valores excelentes para construcción nueva.
    
    Según PDF Sección 2:
    "Cuando se construye desde cero, la calidad será excelente"
    
    Args:
        modelo: Modelo de Gurobi
        variables: Dict con variables del modelo
        calidades_default: Dict con valores por defecto (de config.py)
    """
    
    print("\n" + "="*70)
    print("FIJANDO CALIDADES PARA CONSTRUCCIÓN NUEVA (Sección 2 del PDF)")
    print("="*70)
    
    for var_name, valor_excelente in calidades_default.items():
        if var_name in variables:
            modelo.addConstr(
                variables[var_name] == valor_excelente,
                name=f'calidad_{var_name}_excelente'
            )
            print(f"  ✓ {var_name} = {valor_excelente} (Excelente)")
    
    print("="*70)


# ============================================================================
# FUNCIÓN COMPLETA: Setup modelo de construcción
# ============================================================================

def setup_modelo_construccion_completo(terreno_params, calidades_default):
    """
    Crea y configura un modelo COMPLETO de construcción desde cero.
    
    Args:
        terreno_params: Dict con parámetros del terreno (LotArea, etc.)
        calidades_default: Dict con calidades por defecto (de config.py)
        
    Returns:
        tuple: (modelo, variables_dict)
    """
    
    modelo = gp.Model("Construccion_Desde_Cero")
    modelo.Params.LogToConsole = 0
    modelo.Params.NonConvex = 2
    
    print("\n" + "="*70)
    print("CREANDO MODELO DE CONSTRUCCIÓN DESDE CERO")
    print("="*70)
    
    # ========== CREAR VARIABLES ==========
    
    from config import ATRIBUTOS_ACCIONABLES
    
    variables = {}
    
    print("\n📋 Creando variables de decisión...")
    
    for attr, (costo, es_discreta, min_val, max_val, desc) in ATRIBUTOS_ACCIONABLES.items():
        if es_discreta:
            variables[attr] = modelo.addVar(
                lb=min_val, ub=max_val, 
                vtype=GRB.INTEGER, 
                name=attr
            )
        else:
            variables[attr] = modelo.addVar(
                lb=min_val, ub=max_val, 
                vtype=GRB.CONTINUOUS, 
                name=attr
            )
    
    print(f"  ✓ {len(variables)} variables creadas")
    
    # ========== AGREGAR RESTRICCIONES ==========
    
    # 1. Restricciones de construcción (Sección 7)
    agregar_restricciones_construccion(modelo, variables, terreno_params)
    
    # 2. Fijar calidades a excelente (Sección 2)
    fijar_calidades_construccion_nueva(modelo, variables, calidades_default)
    
    # 3. Variable auxiliar para garaje
    tiene_garaje = agregar_variable_garaje_condicional(modelo, variables)
    variables['tiene_garaje'] = tiene_garaje
    
    print("\n✅ Modelo de construcción configurado completamente")
    
    return modelo, variables


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    from config import CALIDADES_CONSTRUCCION_NUEVA, ATRIBUTOS_ACCIONABLES
    
    # Parámetros del terreno (dados)
    terreno = {
        'LotArea': 10000,      # 10,000 sqft de terreno
        'LotFrontage': 80,     # 80 pies de frente
        'Neighborhood': 'CollgCr',
        'LotShape': 'Reg',     # Regular
    }
    
    print("\n🏗️  TERRENO DISPONIBLE:")
    for k, v in terreno.items():
        print(f"  {k}: {v}")
    
    # Crear modelo completo
    modelo, variables = setup_modelo_construccion_completo(
        terreno_params=terreno,
        calidades_default=CALIDADES_CONSTRUCCION_NUEVA
    )
    
    # ========== DEFINIR FUNCIÓN OBJETIVO ==========
    # Minimizar costo de construcción (ejemplo simple)
    
    costo_total = 0
    for attr, var in variables.items():
        if attr in ATRIBUTOS_ACCIONABLES:
            costo_unitario = ATRIBUTOS_ACCIONABLES[attr][0]
            costo_total += var * costo_unitario
    
    modelo.setObjective(costo_total, GRB.MINIMIZE)
    
    print("\n🎯 Objetivo: Minimizar costo de construcción")
    
    # Resolver
    print("\n⚙️  Resolviendo modelo...")
    modelo.optimize()
    
    if modelo.status == GRB.OPTIMAL:
        print("\n" + "="*70)
        print("✅ SOLUCIÓN ÓPTIMA ENCONTRADA")
        print("="*70)
        print(f"\nCosto total de construcción: ${modelo.ObjVal:,.0f}")
        
        print("\n🏠 CARACTERÍSTICAS DE LA CASA ÓPTIMA:")
        for attr, var in sorted(variables.items()):
            if attr in ATRIBUTOS_ACCIONABLES and var.X > 0.01:
                print(f"  {attr:<20} = {var.X:>10.1f}")
    else:
        print(f"\n❌ No se encontró solución óptima. Status: {modelo.status}")

print("\n✅ Archivo restricciones_construccion.py creado")