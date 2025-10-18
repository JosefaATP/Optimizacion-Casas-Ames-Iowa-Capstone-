import gurobipy as gp
from gurobipy import GRB
from costos import COSTOS
from variables import definir_variables, FIXED, VARIABLES 
from variables import definir_variables  # solo esto es obligatorio

# Fallbacks si no existen en variables.py
try:
    from variables import DEFAULTS
except Exception:
    DEFAULTS = {}  # inicia vacío; idealmente lo definimos bien en Opción B


import joblib
import pandas as pd
import itertools
import random


# Parámetros del proyecto
presupuesto = 120000
LotArea = 800 # m2  (no usado aquí)

# Modelo
m = gp.Model("diseno_casa")

(
    MSSubClass, Utilities, BldgType, HouseStyle, RoofStyle, RoofMatl, Exterior1st, Exterior2nd,
    MasVnrType, MasVnrArea, Foundation, BsmtExposure, BsmtFinType1, BsmtFinSF1,
    BsmtFinType2, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, Heating, CentralAir, Electrical,
    FirstFlrSF, SecondFlrSF, LowQualFinSF, GrLivArea, BsmtFullBath, BsmtHalfBath,
    FullBath, HalfBath, Bedroom, Kitchen, TotRmsAbvGrd, Fireplaces,
    GarageType, GarageFinish, GarageCars, GarageArea,
    PavedDrive, WoodDeckSF, OpenPorchSF, EnclosedPorch, ThreeSsnPorch, ScreenPorch, PoolArea,
    MiscFeature, SaleType, SaleCondition, FenceLength, AreaAmpliacionFt2, AreaDemolicionFt2,
    HeatingQC, BasementCond, KitchenQual, FireplaceQu, PoolQC, GarageQu, GarageCarsCat, Fence
) = definir_variables(m)


for grupo, dicc in COSTOS.items():
    if isinstance(dicc, dict):
        # Detecta si hay desajuste entre COSTOS y variables
        var_name = grupo.lower().replace("_psf", "").replace("_remodel", "")
        if var_name in locals():
            var = locals()[var_name]
            missing = [k for k in var.keys() if k not in dicc and k.upper() not in dicc]
            if missing:
                print(f"⚠️  En {grupo} faltan costos para: {missing}")




############# DEFINICIONES #######################

def costo_total_expr(vars_dict):
    """
    Construye una expresión lineal de Gurobi con el costo total
    consistente con COSTOS y con lo que hiciste en costo_total_de_diseno().
    """
    # Desempaqueta lo que definiste en definir_variables(m)
    # (nombres típicos; ajusta si tus claves son distintas)
    FirstFlrSF = vars_dict.get("FirstFlrSF") or vars_dict.get("1st Flr SF")  # a veces lo nombramos distinto
    GrLivArea  = vars_dict.get("GrLivArea")  or vars_dict.get("Gr Liv Area")

    Utilities  = vars_dict.get("Utilities", {})
    Heating    = vars_dict.get("Heating", {})
    HeatingQC  = vars_dict.get("HeatingQC", {}) or vars_dict.get("Heating QC", {})
    CentralAir = vars_dict.get("CentralAir", {}) or vars_dict.get("Central Air", {})
    RoofMatl   = vars_dict.get("RoofMatl", {})
    Exterior1  = vars_dict.get("Exterior1st", {}) or vars_dict.get("Exterior 1st", {})
    MasVnrType = vars_dict.get("MasVnrType", {}) or vars_dict.get("Mas Vnr Type", {})
    MasVnrArea = vars_dict.get("MasVnrArea")  or vars_dict.get("Mas Vnr Area")
    MiscFeat   = vars_dict.get("MiscFeature", {}) or vars_dict.get("Misc Feature", {})

    expr = gp.LinExpr(0.0)

    # UTILITIES (costo fijo por categoría)
    if isinstance(Utilities, dict) and "UTILITIES" in COSTOS:
        for k, var in Utilities.items():
            if k in COSTOS["UTILITIES"]:
                expr += COSTOS["UTILITIES"][k] * var

    # HEATING (costo fijo por categoría)
    if isinstance(Heating, dict) and "HEATING" in COSTOS:
        for k, var in Heating.items():
            if k in COSTOS["HEATING"]:
                expr += COSTOS["HEATING"][k] * var

    # HEATING_QC (costo fijo por categoría)
    if isinstance(HeatingQC, dict) and "HEATING_QC" in COSTOS:
        for k, var in HeatingQC.items():
            if k in COSTOS["HEATING_QC"]:
                expr += COSTOS["HEATING_QC"][k] * var

    # CENTRAL AIR (costo fijo por categoría)
    if isinstance(CentralAir, dict) and "CENTRALAIR" in COSTOS:
        for k, var in CentralAir.items():
            if k in COSTOS["CENTRALAIR"]:
                expr += COSTOS["CENTRALAIR"][k] * var

    # ROOF MATL (costo por pie2 de 1stFlrSF)
    if isinstance(RoofMatl, dict) and "ROOFMATL_PSF" in COSTOS and FirstFlrSF is not None:
        for k, var in RoofMatl.items():
            if k in COSTOS["ROOFMATL_PSF"]:
                expr += COSTOS["ROOFMATL_PSF"][k] * var * FirstFlrSF

    # EXTERIOR 1st (costo por pie2 de GrLivArea)
    if isinstance(Exterior1, dict) and "EXTERIOR_PSF" in COSTOS and GrLivArea is not None:
        for k, var in Exterior1.items():
            if k in COSTOS["EXTERIOR_PSF"]:
                expr += COSTOS["EXTERIOR_PSF"][k] * var * GrLivArea

    # MASONRY VENEER (costo por pie2 de MasVnrArea)
    if isinstance(MasVnrType, dict) and "MASVNRTYPE_PSF" in COSTOS and MasVnrArea is not None:
        for k, var in MasVnrType.items():
            if k in COSTOS["MASVNRTYPE_PSF"]:
                expr += COSTOS["MASVNRTYPE_PSF"][k] * var * MasVnrArea

    # MISC FEATURE (costo fijo por categoría)
    if isinstance(MiscFeat, dict) and "MISCFEATURE" in COSTOS:
        for k, var in MiscFeat.items():
            if k in COSTOS["MISCFEATURE"]:
                expr += COSTOS["MISCFEATURE"][k] * var

    # Si ocupas ampliaciones/demoliciones (opcional):
    # AreaAmpliacionFt2 = vars_dict.get("AreaAmpliacionFt2")
    # AreaDemolicionFt2 = vars_dict.get("AreaDemolicionFt2")
    # if AreaAmpliacionFt2 is not None and "AMPLIACION_PSF" in COSTOS:
    #     expr += COSTOS["AMPLIACION_PSF"] * AreaAmpliacionFt2
    # if AreaDemolicionFt2 is not None and "DEMOLICION_PSF" in COSTOS:
    #     expr += COSTOS["DEMOLICION_PSF"] * AreaDemolicionFt2

    return expr

# ---- Helper para costos categóricos (dict de costos x tupledict de vars) ----
def costo_cat(nombre_costos, var_tupledict):
    """Devuelve sum_k COSTOS[nombre_costos][k] * var_tupledict[k] (0 si falta)."""
    tabla = COSTOS.get(nombre_costos, {})
    if not hasattr(var_tupledict, "keys"):
        # Por si alguna variable es escalar y no diccionario
        coef = tabla if isinstance(tabla, (int, float)) else 0.0
        return coef * var_tupledict
    return gp.quicksum(tabla.get(k, 0.0) * var_tupledict[k] for k in var_tupledict.keys())



# Cargar modelo XGBoost entrenado
MODEL_PATH = "models/xgb/Caso_bayesiano_top/model_xgb.joblib"
xgb_model = joblib.load(MODEL_PATH)

def evaluar_solution_pool(m, vars_dict, max_solutions=100):
    """
    Recorre el Solution Pool de Gurobi, construye la 'decision' para cada solución
    y la evalúa con XGBoost + costos. Devuelve (mejor_fila_df, df_resultados).
    """
    import pandas as pd

    solcount = int(m.SolCount)
    if solcount == 0:
        print("⚠️ No hay soluciones en el pool.")
        return None, pd.DataFrame()

    solcount = min(solcount, max_solutions)
    resultados = []

    # Helper para extraer una decision pero para la solución k (usando Xn en vez de X)
    def extraer_decision_k(vars_dict, k):
        decision = dict(DEFAULTS)

        def argmax_cat_k(d):
            if not isinstance(d, dict) or not d:
                return None
            best_k, best_v = None, -1
            for kk, var in d.items():
                val = var.Xn  # valor en la solución k
                if val > best_v:
                    best_k, best_v = kk, val
            return best_k

        cat_map = {
            "MS Zoning": vars_dict.get("MSZoning", {}),
            "Street": vars_dict.get("Street", {}),
            "Alley": vars_dict.get("Alley", {}),
            "Lot Shape": vars_dict.get("LotShape", {}),
            "Land Contour": vars_dict.get("LandContour", {}),
            "Utilities": vars_dict.get("Utilities", {}),
            "Lot Config": vars_dict.get("LotConfig", {}),
            "Land Slope": vars_dict.get("LandSlope", {}),
            "Neighborhood": vars_dict.get("Neighborhood", {}),
            "Condition 1": vars_dict.get("Condition1", {}),
            "Condition 2": vars_dict.get("Condition2", {}),
            "Bldg Type": vars_dict.get("BldgType", {}),
            "House Style": vars_dict.get("HouseStyle", {}),
            "Roof Style": vars_dict.get("RoofStyle", {}),
            "Roof Matl": vars_dict.get("RoofMatl", {}),
            "Exterior 1st": vars_dict.get("Exterior1st", {}),
            "Exterior 2nd": vars_dict.get("Exterior2nd", {}),
            "Mas Vnr Type": vars_dict.get("MasVnrType", {}),
            "Exter Qual": vars_dict.get("ExterQual", {}),
            "Exter Cond": vars_dict.get("ExterCond", {}),
            "Foundation": vars_dict.get("Foundation", {}),
            "Bsmt Qual": vars_dict.get("BsmtQual", {}),
            "Bsmt Cond": vars_dict.get("BsmtCond", {}),
            "Bsmt Exposure": vars_dict.get("BsmtExposure", {}),
            "BsmtFin Type 1": vars_dict.get("BsmtFinType1", {}),
            "BsmtFin Type 2": vars_dict.get("BsmtFinType2", {}),
            "Heating": vars_dict.get("Heating", {}),
            "Heating QC": vars_dict.get("HeatingQC", {}) or vars_dict.get("Heating QC", {}),
            "Central Air": vars_dict.get("CentralAir", {}) or vars_dict.get("Central Air", {}),
            "Electrical": vars_dict.get("Electrical", {}),
            "Kitchen Qual": vars_dict.get("KitchenQual", {}),
            "FunctioNo aplical": vars_dict.get("Functional", {}) or vars_dict.get("FunctioNo aplical", {}),
            "Fireplace Qu": vars_dict.get("FireplaceQu", {}),
            "Garage Type": vars_dict.get("GarageType", {}),
            "Garage Finish": vars_dict.get("GarageFinish", {}),
            "Garage Qual": vars_dict.get("GarageQual", {}),
            "Garage Cond": vars_dict.get("GarageCond", {}),
            "Paved Drive": vars_dict.get("PavedDrive", {}),
            "Pool QC": vars_dict.get("PoolQC", {}),
            "Fence": vars_dict.get("Fence", {}),
            "Misc Feature": vars_dict.get("MiscFeature", {}),
            "Sale Type": vars_dict.get("SaleType", {}),
            "Sale Condition": vars_dict.get("SaleCondition", {}),
        }

        for col, d in cat_map.items():
            val = argmax_cat_k(d)
            if val is not None:
                decision[col] = val

        num_map = {
            "MS SubClass": vars_dict.get("MSSubClass"),
            "Lot Frontage": vars_dict.get("LotFrontage"),
            "Lot Area": vars_dict.get("LotArea"),
            "Overall Qual": vars_dict.get("OverallQual"),
            "Overall Cond": vars_dict.get("OverallCond"),
            "Year Built": vars_dict.get("YearBuilt"),
            "Year Remod/Add": vars_dict.get("YearRemodAdd"),
            "Mas Vnr Area": vars_dict.get("MasVnrArea"),
            "BsmtFin SF 1": vars_dict.get("BsmtFinSF1"),
            "BsmtFin SF 2": vars_dict.get("BsmtFinSF2"),
            "Bsmt Unf SF": vars_dict.get("BsmtUnfSF"),
            "Total Bsmt SF": vars_dict.get("TotalBsmtSF"),
            "1st Flr SF": vars_dict.get("FirstFlrSF") or vars_dict.get("1st Flr SF"),
            "2nd Flr SF": vars_dict.get("SecondFlrSF") or vars_dict.get("2nd Flr SF"),
            "Low Qual Fin SF": vars_dict.get("LowQualFinSF"),
            "Gr Liv Area": vars_dict.get("GrLivArea") or vars_dict.get("Gr Liv Area"),
            "Bsmt Full Bath": vars_dict.get("BsmtFullBath"),
            "Bsmt Half Bath": vars_dict.get("BsmtHalfBath"),
            "Full Bath": vars_dict.get("FullBath"),
            "Half Bath": vars_dict.get("HalfBath"),
            "Bedroom AbvGr": vars_dict.get("BedroomAbvGr"),
            "Kitchen AbvGr": vars_dict.get("KitchenAbvGr"),
            "TotRms AbvGrd": vars_dict.get("TotRmsAbvGrd"),
            "Fireplaces": vars_dict.get("Fireplaces"),
            "Garage Yr Blt": vars_dict.get("GarageYrBlt"),
            "Garage Cars": vars_dict.get("GarageCars"),
            "Garage Area": vars_dict.get("GarageArea"),
            "Wood Deck SF": vars_dict.get("WoodDeckSF"),
            "Open Porch SF": vars_dict.get("OpenPorchSF"),
            "Enclosed Porch": vars_dict.get("EnclosedPorch"),
            "3Ssn Porch": vars_dict.get("ThreeSsnPorch") or vars_dict.get("3Ssn Porch"),
            "Screen Porch": vars_dict.get("ScreenPorch"),
            "Pool Area": vars_dict.get("PoolArea"),
            "Misc Val": vars_dict.get("MiscVal"),
            "Mo Sold": vars_dict.get("MoSold"),
            "Yr Sold": vars_dict.get("YrSold"),
        }
        for col, var in num_map.items():
            if var is not None:
                decision[col] = var.Xn  # valor en solución k

        # Por seguridad, reimpone FIXED si usas ese esquema:
        try:
            from variables import FIXED
            for k_fixed, v_fixed in FIXED.items():
                decision[k_fixed] = v_fixed
        except Exception:
            pass

        return decision

    # Recorremos el pool
    for k in range(solcount):
        m.Params.SolutionNumber = k  # activamos solución k
        decision_k = extraer_decision_k(vars_dict, k)
        precio, costo = evaluar_casa(decision_k)
        resultados.append({
            **{kk: decision_k.get(kk) for kk in decision_k.keys()},
            "Precio Predicho": precio,
            "Costo Total": costo,
            "Rentabilidad": precio - costo
        })

    df = pd.DataFrame(resultados)
    mejor = df.iloc[df["Rentabilidad"].idxmax()]
    print("\n🏆 Mejor del pool por rentabilidad:")
    print(mejor[["Precio Predicho", "Costo Total", "Rentabilidad"]])

    return mejor, df


def costo_total_de_diseno(decision: dict) -> float:
    """
    Calcula el costo total de construcción/remodelación según el diccionario COSTOS.
    """
    costo_total = 0.0

    # Ejemplo: costo por tipo de calefacción
    heating = decision.get("Heating")
    if heating in COSTOS["HEATING"]:
        costo_total += COSTOS["HEATING"][heating]

    # Ejemplo: costo por tipo de techo
    roofmatl = decision.get("Roof Matl")
    if roofmatl in COSTOS["ROOFMATL_PSF"]:
        area = decision.get("1st Flr SF", 0)
        costo_total += COSTOS["ROOFMATL_PSF"][roofmatl] * area

    # Ejemplo: costo por material exterior
    exterior = decision.get("Exterior 1st")
    if exterior in COSTOS["EXTERIOR_PSF"]:
        area = decision.get("Gr Liv Area", 0)
        costo_total += COSTOS["EXTERIOR_PSF"][exterior] * area

    # Ejemplo: costo por central air
    central_air = decision.get("Central Air")
    if central_air in COSTOS["CENTRALAIR"]:
        costo_total += COSTOS["CENTRALAIR"][central_air]

    # (Aquí se pueden seguir agregando más componentes si se desea)
    return costo_total

def construir_modelo_con_restricciones(presupuesto=None):
    """
    Crea el modelo de Gurobi, define variables, añade tus restricciones (las que ya existen en tu archivo)
    y agrega la restricción de presupuesto basada en COSTOS.
    Devuelve: (m, vars_dict, expr_costo_total)
    """
    m = gp.Model("diseno_casa_hibrido")



    # 1) variables
    vars_out = definir_variables(m)      # <- lo que sea que retorne…
    vars_dict = _vars_as_dict(vars_out)  # <- …lo normalizamos a dict
  # esta función es tuya (en variables.py)
    if not vars_dict:
        raise RuntimeError(
            "definir_variables(m) no retorna un dict ni (dict, ...). "
            "Revisa variables.py o ajusta _vars_as_dict para el formato real."
        )

    # 2) AQUÍ: asegúrate de que en tu archivo ya agregaste TODAS las restricciones físicas.
    #    Si ya las tenías codificadas (áreas, ocupación del terreno, relaciones lógicas, etc.),
    #    déjalas tal cual. Este bloque no las reescribe; solo añadimos el presupuesto.
    #
    #    Ejemplo clásico que ya vi en tu código (no repitas si ya lo tienes):
    #    m.addConstr(vars_dict["FirstFlrSF"] + vars_dict["TotalPorchSF"] + vars_dict["PoolArea"] <= vars_dict["LotArea"],
    #                name="OcupacionTerreno")

    # 3) costo total expresado para Gurobi
    expr_costo = costo_total_expr(vars_dict)

    # 4) restricción de presupuesto
    if presupuesto is None:
        presupuesto = presupuesto  # si lo tienes en variables.py
    m.addConstr(expr_costo <= presupuesto, name="RestriccionPresupuesto")

    # 5) objetivo neutro, solo queremos factibilidad por ahora
    m.setObjective(0.0, GRB.MINIMIZE)

    # algunos parámetros suaves para encontrar factibles rápido
    m.setParam("OutputFlag", 1)
    m.setParam("MIPFocus", 1)       # foco en factibilidad
    m.setParam("Heuristics", 0.2)
    m.setParam("PoolSearchMode", 2)   # busca diversas soluciones
    m.setParam("PoolSolutions", 200)  # cuántas queremos guardar (ajusta)

    return m, vars_dict, expr_costo

def extraer_decision(vars_dict):
    """
    Convierte los valores de las variables de Gurobi en un dict 'decision'
    con los NOMBRES EXACTOS que espera tu XGBoost.
    - Para categóricas one-hot: toma la categoría con valor 1 (o la mayor).
    - Para numéricas: toma .X
    Completa con DEFAULTS cuando falte.
    """
    decision = dict(DEFAULTS)  # punto de partida

    # helper para categóricas binarias
    def argmax_cat(d):
        if not isinstance(d, dict) or not d:
            return None
        best_k, best_v = None, -1
        for k, var in d.items():
            val = var.X if hasattr(var, "X") else 0.0
            if val > best_v:
                best_k, best_v = k, val
        return best_k

    # mapea lo más común (ajusta nombres si en definir_variables usaste otros)
    cat_map = {
        "MS Zoning": vars_dict.get("MSZoning", {}),
        "Street": vars_dict.get("Street", {}),
        "Alley": vars_dict.get("Alley", {}),
        "Lot Shape": vars_dict.get("LotShape", {}),
        "Land Contour": vars_dict.get("LandContour", {}),
        "Utilities": vars_dict.get("Utilities", {}),
        "Lot Config": vars_dict.get("LotConfig", {}),
        "Land Slope": vars_dict.get("LandSlope", {}),
        "Neighborhood": vars_dict.get("Neighborhood", {}),
        "Condition 1": vars_dict.get("Condition1", {}),
        "Condition 2": vars_dict.get("Condition2", {}),
        "Bldg Type": vars_dict.get("BldgType", {}),
        "House Style": vars_dict.get("HouseStyle", {}),
        "Roof Style": vars_dict.get("RoofStyle", {}),
        "Roof Matl": vars_dict.get("RoofMatl", {}),
        "Exterior 1st": vars_dict.get("Exterior1st", {}),
        "Exterior 2nd": vars_dict.get("Exterior2nd", {}),
        "Mas Vnr Type": vars_dict.get("MasVnrType", {}),
        "Exter Qual": vars_dict.get("ExterQual", {}),
        "Exter Cond": vars_dict.get("ExterCond", {}),
        "Foundation": vars_dict.get("Foundation", {}),
        "Bsmt Qual": vars_dict.get("BsmtQual", {}),
        "Bsmt Cond": vars_dict.get("BsmtCond", {}),
        "Bsmt Exposure": vars_dict.get("BsmtExposure", {}),
        "BsmtFin Type 1": vars_dict.get("BsmtFinType1", {}),
        "BsmtFin Type 2": vars_dict.get("BsmtFinType2", {}),
        "Heating": vars_dict.get("Heating", {}),
        "Heating QC": vars_dict.get("HeatingQC", {}),
        "Central Air": vars_dict.get("CentralAir", {}),
        "Electrical": vars_dict.get("Electrical", {}),
        "Kitchen Qual": vars_dict.get("KitchenQual", {}),
        "FunctioNo aplical": vars_dict.get("Functional", {}) or vars_dict.get("FunctioNo aplical", {}),
        "Fireplace Qu": vars_dict.get("FireplaceQu", {}),
        "Garage Type": vars_dict.get("GarageType", {}),
        "Garage Finish": vars_dict.get("GarageFinish", {}),
        "Garage Qual": vars_dict.get("GarageQual", {}),
        "Garage Cond": vars_dict.get("GarageCond", {}),
        "Paved Drive": vars_dict.get("PavedDrive", {}),
        "Pool QC": vars_dict.get("PoolQC", {}),
        "Fence": vars_dict.get("Fence", {}),
        "Misc Feature": vars_dict.get("MiscFeature", {}),
        "Sale Type": vars_dict.get("SaleType", {}),
        "Sale Condition": vars_dict.get("SaleCondition", {}),
    }

    for col, d in cat_map.items():
        val = argmax_cat(d)
        if val is not None:
            decision[col] = val

    # numéricas directas (ajusta los nombres si difieren)
    num_map = {
        "MS SubClass": vars_dict.get("MSSubClass"),
        "Lot Frontage": vars_dict.get("LotFrontage"),
        "Lot Area": vars_dict.get("LotArea"),
        "Overall Qual": vars_dict.get("OverallQual"),
        "Overall Cond": vars_dict.get("OverallCond"),
        "Year Built": vars_dict.get("YearBuilt"),
        "Year Remod/Add": vars_dict.get("YearRemodAdd"),
        "Mas Vnr Area": vars_dict.get("MasVnrArea"),
        "BsmtFin SF 1": vars_dict.get("BsmtFinSF1"),
        "BsmtFin SF 2": vars_dict.get("BsmtFinSF2"),
        "Bsmt Unf SF": vars_dict.get("BsmtUnfSF"),
        "Total Bsmt SF": vars_dict.get("TotalBsmtSF"),
        "1st Flr SF": vars_dict.get("FirstFlrSF") or vars_dict.get("1st Flr SF"),
        "2nd Flr SF": vars_dict.get("SecondFlrSF") or vars_dict.get("2nd Flr SF"),
        "Low Qual Fin SF": vars_dict.get("LowQualFinSF"),
        "Gr Liv Area": vars_dict.get("GrLivArea") or vars_dict.get("Gr Liv Area"),
        "Bsmt Full Bath": vars_dict.get("BsmtFullBath"),
        "Bsmt Half Bath": vars_dict.get("BsmtHalfBath"),
        "Full Bath": vars_dict.get("FullBath"),
        "Half Bath": vars_dict.get("HalfBath"),
        "Bedroom AbvGr": vars_dict.get("BedroomAbvGr"),
        "Kitchen AbvGr": vars_dict.get("KitchenAbvGr"),
        "TotRms AbvGrd": vars_dict.get("TotRmsAbvGrd"),
        "Fireplaces": vars_dict.get("Fireplaces"),
        "Garage Yr Blt": vars_dict.get("GarageYrBlt"),
        "Garage Cars": vars_dict.get("GarageCars"),
        "Garage Area": vars_dict.get("GarageArea"),
        "Wood Deck SF": vars_dict.get("WoodDeckSF"),
        "Open Porch SF": vars_dict.get("OpenPorchSF"),
        "Enclosed Porch": vars_dict.get("EnclosedPorch"),
        "3Ssn Porch": vars_dict.get("ThreeSsnPorch") or vars_dict.get("3Ssn Porch"),
        "Screen Porch": vars_dict.get("ScreenPorch"),
        "Pool Area": vars_dict.get("PoolArea"),
        "Misc Val": vars_dict.get("MiscVal"),
        "Mo Sold": vars_dict.get("MoSold"),
        "Yr Sold": vars_dict.get("YrSold"),
    }
    for col, var in num_map.items():
        if var is not None:
            decision[col] = var.X

    return decision

def costo_total_de_diseno(decision: dict) -> float:
    """
    Calcula el costo total estimado de construcción según las características del diseño.
    """
    costo_total = 0.0

    # ---- Ejemplo de costos por categoría ----
    # Tipo de calefacción
    heating = decision.get("Heating")
    if heating in COSTOS["HEATING"]:
        costo_total += COSTOS["HEATING"][heating]

    # Calidad de calefacción
    heating_qc = decision.get("Heating QC")
    if heating_qc in COSTOS["HEATING_QC"]:
        costo_total += COSTOS["HEATING_QC"][heating_qc]

    # Aire acondicionado central
    central_air = decision.get("Central Air")
    if central_air in COSTOS["CENTRALAIR"]:
        costo_total += COSTOS["CENTRALAIR"][central_air]

    # Material del techo
    roof_matl = decision.get("Roof Matl")
    area_techo = decision.get("1st Flr SF", 0)
    if roof_matl in COSTOS["ROOFMATL_PSF"]:
        costo_total += COSTOS["ROOFMATL_PSF"][roof_matl] * area_techo

    # Revestimiento exterior principal
    exterior1 = decision.get("Exterior 1st")
    area_exterior = decision.get("Gr Liv Area", 0)
    if exterior1 in COSTOS["EXTERIOR_PSF"]:
        costo_total += COSTOS["EXTERIOR_PSF"][exterior1] * area_exterior

    # Tipo de mampostería
    masvnr = decision.get("Mas Vnr Type")
    area_masvnr = decision.get("Mas Vnr Area", 0)
    if masvnr in COSTOS["MASVNRTYPE_PSF"]:
        costo_total += COSTOS["MASVNRTYPE_PSF"][masvnr] * area_masvnr

    # Servicios básicos
    utilities = decision.get("Utilities")
    if utilities in COSTOS["UTILITIES"]:
        costo_total += COSTOS["UTILITIES"][utilities]

    # Características misceláneas
    misc = decision.get("Misc Feature")
    if misc in COSTOS["MISCFEATURE"]:
        costo_total += COSTOS["MISCFEATURE"][misc]

    return costo_total

def fijar_campos_en_modelo(m, vars_dict, fixed):
    # Categóricas one-hot
    def fijar_cat(nombre_cat_modelo, valor_objetivo):
        d = vars_dict.get(nombre_cat_modelo, {})
        # forzamos a 1 la categoría objetivo, y 0 el resto
        for k, var in d.items():
            m.addConstr(var == (1.0 if k == valor_objetivo else 0.0),
                        name=f"fix_{nombre_cat_modelo}_{k}")

    # Numéricas directas
    def fijar_num(nombre_var_modelo, valor):
        v = vars_dict.get(nombre_var_modelo)
        if v is not None:
            m.addConstr(v == valor, name=f"fix_{nombre_var_modelo}")

    # Mapea nombres “humanos” → nombres de variables del modelo
    # (ajusta según tus nombres reales en definir_variables)
    mapa_cat = {
        "MS Zoning": "MSZoning",
        "Street": "Street",
        "Neighborhood": "Neighborhood",
    }
    mapa_num = {
        "Lot Area": "LotArea",
        "Lot Frontage": "LotFrontage",
    }

    for k, v in fixed.items():
        if k in mapa_cat:
            fijar_cat(mapa_cat[k], v)
        elif k in mapa_num:
            fijar_num(mapa_num[k], v)
        else:
            # si el fijo es otro campo, lo puedes ampliar aquí
            pass

def buscar_mejor_combinacion(presupuesto: float, base: dict, n_iteraciones: int = 2000, semilla: int = 42):

    """
    Búsqueda aleatoria (heurística) de combinaciones dentro del presupuesto.
    """
    random.seed(semilla)

    opciones = {
        "Bldg Type": ["1Fam", "TwnhsE", "TwnhsI"],
        "House Style": ["1Story", "2Story", "SLvl"],
        "Roof Matl": ["CompShg", "Metal", "WdShngl", "TarGrv"],
        "Heating": ["GasA", "GasW", "Floor", "Grav"],
        "Central Air": ["Y", "N"],
        "Exterior 1st": ["VinylSd", "BrkFace", "MetalSd", "Wd Sdng"],
        "Exterior 2nd": ["VinylSd", "Plywood", "Wd Sdng", "MetalSd"],
        "Mas Vnr Type": ["None", "BrkFace", "Stone"],
        "Kitchen Qual": ["Ex", "Gd", "TA", "Fa"],
        "Garage Type": ["Attchd", "BuiltIn", "Detchd"],
        "Roof Style": ["Gable", "Hip", "Flat"],
        "Sale Condition": ["Normal", "Partial", "Abnorml"],
    }

    resultados = []
    mejor = None
    mejor_valor = -float("inf")

    print(f"🎲 Explorando {n_iteraciones} combinaciones aleatorias...")

    for i in range(n_iteraciones):
        decision = base.copy()

        # Asignar un valor aleatorio a cada variable
        for k, v in opciones.items():
            decision[k] = random.choice(v)

        # Evaluar casa
        precio, costo = evaluar_casa(decision)

        if costo <= presupuesto:
            rentabilidad = precio - costo
            resultados.append({
                **{k: decision[k] for k in opciones},
                "Precio Predicho": precio,
                "Costo Total": costo,
                "Rentabilidad": rentabilidad,
            })

            # Actualizar mejor combinación
            if rentabilidad > mejor_valor:
                mejor_valor = rentabilidad
                mejor = decision.copy()

        # Log ocasional
        if (i + 1) % 500 == 0:
            print(f"  → Iteración {i+1}/{n_iteraciones}")

    if not resultados:
        print("⚠️  No se encontró ninguna combinación dentro del presupuesto.")
        return None, pd.DataFrame()

    df = pd.DataFrame(resultados)
    mejor_df = df.loc[df["Rentabilidad"].idxmax()]

    print("\n✅ Mejor combinación encontrada:")
    print(mejor_df)

    return mejor_df, df

def _vars_as_dict(vars_out):
    """
    Normaliza la salida de definir_variables(m).
    - Si ya es dict, lo retorna tal cual.
    - Si es tuple/list, intenta usar el primer elemento como dict.
    - Si falla, devuelve {}.
    """
    if isinstance(vars_out, dict):
        return vars_out
    if isinstance(vars_out, (list, tuple)) and len(vars_out) > 0 and isinstance(vars_out[0], dict):
        return vars_out[0]
    return {}


# =============================================
# Función principal de evaluación
# =============================================
def evaluar_casa(decision: dict) -> tuple:
    """
    Evalúa una casa con las características dadas.
    Devuelve:
        - precio_predicho (float)
        - costo_total (float)
    """
    # Convertir en DataFrame
    df_input = pd.DataFrame([decision])

    # Predecir precio con el modelo XGB
    precio_predicho = float(xgb_model.predict(df_input)[0])

    # Calcular costo total
    costo_total = costo_total_de_diseno(decision)

    return precio_predicho, costo_total
# ============================================================
# 8  Restricciones de Construcción
# ============================================================

# ----------------------------------------------------------------
# Auxiliares (créanse si no existen en tu modelo)
# ----------------------------------------------------------------
TotalArea      = m.addVar(lb=0, name="TotalArea")        # ft^2 totales (1er + 2do + basement)
TotalPorchSF   = m.addVar(lb=0, name="TotalPorchSF")     # suma de porches/terrazas

# Definición auxiliar de TotalPorchSF
m.addConstr(
    TotalPorchSF == (WoodDeckSF + OpenPorchSF + EnclosedPorch + ThreeSsnPorch + ScreenPorch),
    name="8.1.2_aux_TotalPorchSF_Def"
)

# ============================================================
# 8.1 Restricciones de Área y Dimensiones
# ============================================================

# 8.1.1  Límites por Vecindario y Terreno
# (2) TotalArea_i ≤ Ā^Total_{Neighborhood_i}
#     Si tienes un tope por vecindario, úsalo acá. Si no, usa un tope global.
Amax_total = 1e9  # <-- reemplaza por tu parámetro real (p. ej. según vecindario)
m.addConstr(
    TotalArea <= Amax_total,
    name="8.1.1_(2)_AreaTotal_MaxBarrio"
)

# (3) 1stFlrSF + TotalPorchSF + PoolArea ≤ LotArea
m.addConstr(
    FirstFlrSF + TotalPorchSF + PoolArea <= LotArea,
    name="8.1.1_(3)_Ocupacion_Terreno"
)

# (4) 2ndFlrSF ≤ 1stFlrSF
m.addConstr(
    SecondFlrSF <= FirstFlrSF,
    name="8.1.1_(4)_SegundoMenorQuePrimero"
)

# 8.1.2  Consistencias de Área
# (5) TotalBsmtSF = BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF
m.addConstr(
    TotalBsmtSF == BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF,
    name="8.1.2_(5)_Consistencia_Basement"
)

# (6) GrLivArea = 1stFlrSF + 2ndFlrSF + LowQualFinSF
m.addConstr(
    GrLivArea == FirstFlrSF + SecondFlrSF + LowQualFinSF,
    name="8.1.2_(6)_Def_GrLivArea"
)

# (7) TotalArea = 1stFlrSF + 2ndFlrSF + TotalBsmtSF
m.addConstr(
    TotalArea == FirstFlrSF + SecondFlrSF + TotalBsmtSF,
    name="8.1.2_(7)_Def_TotalArea"
)

# ============================================================
# 8.1.3 Piscina Condicionada a Espacio
# ============================================================

# Variable binaria: 1 si se instala piscina, 0 si no
HasPool = m.addVar(vtype=GRB.BINARY, name="HasPool")

# Restricción (8)
# PoolArea ≤ (LotArea − GrLivArea − GarageArea − WoodDeckSF − OpenPorchSF) * HasPool
m.addConstr(
    PoolArea <= (LotArea - GrLivArea - GarageArea - WoodDeckSF - OpenPorchSF) * HasPool,
    name="8.1.3_(8)_Piscina_Espacio"
)

# ============================================================
# 8.2  Restricciones de Ambientes
# ============================================================

# ------------------------------------------------------------
# 8.2.1  Variables de Área por Ambiente
# ------------------------------------------------------------
# Áreas totales por tipo de ambiente (ft^2)
AreaKitchen  = m.addVar(lb=0, name="AreaKitchen")
AreaFullBath = m.addVar(lb=0, name="AreaFullBath")
AreaHalfBath = m.addVar(lb=0, name="AreaHalfBath")
AreaBedroom  = m.addVar(lb=0, name="AreaBedroom")
AreaPool     = m.addVar(lb=0, name="AreaPool")

# Enlazamos AreaPool con la variable existente de piscina
m.addConstr(AreaPool == PoolArea, name="8.2.1_AreaPool_eq_PoolArea")

# ------------------------------------------------------------
# 8.2.2  Áreas Mínimas por Ambiente
# ------------------------------------------------------------
m.addConstr(AreaKitchen  >= 75,  name="8.2.2_(9)_Min_AreaKitchen")
m.addConstr(AreaFullBath >= 40,  name="8.2.2_(10)_Min_AreaFullBath")
m.addConstr(AreaHalfBath >= 20,  name="8.2.2_(11)_Min_AreaHalfBath")
m.addConstr(AreaBedroom  >= 70,  name="8.2.2_(12)_Min_AreaBedroom")
m.addConstr(AreaPool >= 161 * HasPool, name="8.2.2_(13)_Min_AreaPool_cond")

# (14) reservado si tu documento incluye otra cota mínima adicional

# ------------------------------------------------------------
# 8.2.3  Máximo de Ambientes Repetidos
# ------------------------------------------------------------
m.addConstr(Kitchen  <= 3, name="8.2.3_(15)_Max_Kitchen")
m.addConstr(Bedroom  <= 6, name="8.2.3_(16)_Max_Bedroom")
m.addConstr(FullBath <= 4, name="8.2.3_(17)_Max_FullBath")

# ============================================================
# 8.2.4  Relación Baños/Pisos
# ============================================================

# Variables binarias: 1 si la casa tiene 1 piso / 2 pisos
Floor1 = m.addVar(vtype=GRB.BINARY, name="Floor1")
Floor2 = m.addVar(vtype=GRB.BINARY, name="Floor2")

# (18) FullBath + HalfBath ≥ Floor1 + 2·Floor2
m.addConstr(
    FullBath + HalfBath >= Floor1 + 2 * Floor2,
    name="8.2.4_(18)_Banos_por_Piso"
)

# (19) HalfBath ≤ FullBath
m.addConstr(
    HalfBath <= FullBath,
    name="8.2.4_(19)_HalfLEFull"
)

# (20) FullBath ≥ (2/3) · Bedroom
m.addConstr(
    FullBath >= (2.0/3.0) * Bedroom,
    name="8.2.4_(20)_Full_vs_Bedroom"
)

# (Opcional, si quieres forzar que haya al menos un piso)******************
m.addConstr(Floor1 + Floor2 >= 1, name="8.2.4_opc_AlMenosUnPiso")
# (Opcional, si quieres prohibir más de 2 pisos)******************
m.addConstr(Floor1 + Floor2 <= 2, name="8.2.4_opc_A_LoMas_DosPisos")


# ============================================================
# 8.2.5  Funcionalidad Mínima
# ============================================================

# (21) Kitchen ≥ 1
m.addConstr(
    Kitchen >= 1,
    name="8.2.5_(21)_Min_Kitchen"
)

# (22) Bedroom ≥ 1
m.addConstr(
    Bedroom >= 1,
    name="8.2.5_(22)_Min_Bedroom"
)

# (23) FullBath ≥ 1
m.addConstr(
    FullBath >= 1,
    name="8.2.5_(23)_Min_FullBath"
)

# ============================================================
# 8.3  Restricciones del Sótano
# ============================================================

# ------------------------------------------------------------
# Variables adicionales
# ------------------------------------------------------------
HasBsmtFin1 = m.addVar(vtype=GRB.BINARY, name="HasBsmtFin1")
HasBsmtFin2 = m.addVar(vtype=GRB.BINARY, name="HasBsmtFin2")

# ------------------------------------------------------------
# Restricciones
# ------------------------------------------------------------

# (24) BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF = TotalBsmtSF
# (ya existe como parte de 8.1.2_(5), pero se repite aquí por consistencia)
m.addConstr(
    BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF == TotalBsmtSF,
    name="8.3_(24)_TotalBasement"
)

# (25) BsmtFinSF1 ≤ TotalBsmtSF · HasBsmtFin1
m.addConstr(
    BsmtFinSF1 <= TotalBsmtSF * HasBsmtFin1,
    name="8.3_(25)_Fin1_condicional"
)

# (26) BsmtFinSF2 ≤ TotalBsmtSF · HasBsmtFin2
m.addConstr(
    BsmtFinSF2 <= TotalBsmtSF * HasBsmtFin2,
    name="8.3_(26)_Fin2_condicional"
)

# (27) BsmtFinSF1 ≥ 0 , BsmtFinSF2 ≥ 0
m.addConstr(BsmtFinSF1 >= 0, name="8.3_(27a)_Fin1_NoNegativo")
m.addConstr(BsmtFinSF2 >= 0, name="8.3_(27b)_Fin2_NoNegativo")

# ============================================================
# 8.4  Restricciones de Sistemas
# ============================================================

# ------------------------------------------------------------
# 8.4.1  Sistema de Calefacción
# ------------------------------------------------------------
# Variables binarias Heating_t ∈ {0,1} para t ∈ {Floor, GasA, GasW, Grav, OthW, Wall}
# (estas variables ya están definidas en definir_variables(m))

# (28)  Σ_t Heating_t = 1   → se debe elegir exactamente un tipo de calefacción
m.addConstr(
    gp.quicksum(Heating[t] for t in Heating.keys()) == 1,
    name="8.4.1_(28)_Heating_Unique"
)

# ------------------------------------------------------------
# 8.4.2  Sistema Eléctrico
# ------------------------------------------------------------
# Variables binarias Electrical_e ∈ {0,1} para e ∈ {SBrkr, FuseA, FuseF, FuseP, Mix}
# (estas variables ya están definidas en definir_variables(m))

# (29)  Σ_e Electrical_e = 1   → se debe elegir exactamente un tipo de sistema eléctrico
m.addConstr(
    gp.quicksum(Electrical[e] for e in Electrical.keys()) == 1,
    name="8.4.2_(29)_Electrical_Unique"
)

# ============================================================
# 8.4.3  Aire Acondicionado
# ============================================================

# Variable binaria: CentralAir = 1 si tiene aire acondicionado central
# (ya está definida en definir_variables(m))
# No requiere restricción adicional, pero se deja documentada:
# (Solo la incluimos si quieres forzar su existencia mínima o condicional)
# m.addConstr(CentralAir >= 0, name="8.4.3_(NA)_CentralAir")

# ============================================================
# 8.5  Restricciones del Techo
# ============================================================

# ------------------------------------------------------------
# 8.5.1  Estilo de Techo
# ------------------------------------------------------------
# Variables binarias RoofStyle_s ∈ {0,1} para s ∈ {Flat, Gable, Gambrel, Hip, Mansard, Shed}
# (ya definidas en definir_variables(m))

# (30)  Σ_s RoofStyle_s = 1  → se elige exactamente un estilo de techo
m.addConstr(
    gp.quicksum(RoofStyle[s] for s in RoofStyle.keys()) == 1,
    name="8.5.1_(30)_RoofStyle_Unique"
)

# ------------------------------------------------------------
# 8.5.2  Material de Techo
# ------------------------------------------------------------
# Variables binarias RoofMatl_m ∈ {0,1} para m ∈ {ClyTile, CompShg, Membran, Metal, Roll, TarGrv, WdShake, WdShngl}
# (ya definidas en definir_variables(m))

# (31)  Σ_m RoofMatl_m = 1  → se elige exactamente un material de techo
m.addConstr(
    gp.quicksum(RoofMatl[m_] for m_ in RoofMatl.keys()) == 1,
    name="8.5.2_(31)_RoofMatl_Unique"
)

# ------------------------------------------------------------
# 8.5.3  Compatibilidad Estilo–Material
# ------------------------------------------------------------
# Parámetros Asm ∈ {0,1} que indican compatibilidad entre estilo y material.
#  Si Asm = 0 → combinación no permitida
#  Si Asm = 1 → combinación válida
# Restricción (32): RoofStyle_s + RoofMatl_m ≤ 1 + A_s,m

# Definimos una matriz de compatibilidad de ejemplo (ajústala a tus datos reales):
A_compat = {
    ("Flat",     "Membran"): 1,
    ("Flat",     "Metal"):   1,
    ("Flat",     "CompShg"): 0,
    ("Gable",    "CompShg"): 1,
    ("Gable",    "WdShngl"): 1,
    ("Hip",      "CompShg"): 1,
    ("Hip",      "WdShake"): 1,
    ("Mansard",  "WdShake"): 1,
    ("Gambrel",  "WdShngl"): 1,
    ("Shed",     "Metal"):   1,
    # Las demás combinaciones no listadas se asumen incompatibles (A_s,m = 0)
}

# (32) Compatibilidad: RoofStyle_s + RoofMatl_m ≤ 1 + A_s,m
for s in RoofStyle.keys():
    for m_ in RoofMatl.keys():
        A_sm = A_compat.get((s, m_), 0)
        m.addConstr(
            RoofStyle[s] + RoofMatl[m_] <= 1 + A_sm,
            name=f"8.5.3_(32)_Compat_{s}_{m_}"
        )
# ============================================================
# 8.6  Restricciones de Estilo de Vivienda
# ============================================================

# Alias por legibilidad (usando las llaves del vector HouseStyle)
HS_1 = HouseStyle["1Story"]   # HouseStyle_1Story ∈ {0,1}
HS_2 = HouseStyle["2Story"]   # HouseStyle_2Story ∈ {0,1}

# (33) HouseStyle_1Story + HouseStyle_2Story = 1
#     → La vivienda es de 1 o 2 pisos (excluyentes).
m.addConstr(
    HS_1 + HS_2 == 1,
    name="8.6_(33)_OneOrTwoStory"
)

# (34) FullBath + HalfBath ≥ HouseStyle_1Story + 2·HouseStyle_2Story
#     → Al menos 1 baño total si es de 1 piso; al menos 2 si es de 2 pisos.
m.addConstr(
    FullBath + HalfBath >= HS_1 + 2*HS_2,
    name="8.6_(34)_Baths_vs_Stories"
)

# --- (OPCIONAL) Si tu vector HouseStyle incluye otros estilos (p.ej. '1.5Fin', '1.5Unf', '2.5Unf'),
#     puedes forzarlos a 0 para que la 8.6 sea consistente con el documento:
# for k in HouseStyle.keys():
#     if k not in ("1Story", "2Story"):
#         m.addConstr(HouseStyle[k] == 0, name=f"8.6_opc_HouseStyleFix_{k}")

# Vincular pisos con estilo de vivienda (opcional)
m.addConstr(Floor1 == HS_1, name="8.6_link_Floor1_1Story")
m.addConstr(Floor2 == HS_2, name="8.6_link_Floor2_2Story")

# ============================================================
# 8.7  Restricciones de Exterior
# ============================================================

# ------------------------------------------------------------
# 8.7.1  Revestimiento Exterior
# ------------------------------------------------------------
# Conjunto M = {AsbShng, AsphShn, ..., WdShing}
# Variables binarias:
#   Exterior1st_m ∈ {0,1}  para material primario
#   Exterior2nd_m ∈ {0,1}  para material secundario
#   UseExterior1st, UseExterior2nd ∈ {0,1}
#   SameMaterial ∈ {0,1}

UseExterior1st = m.addVar(vtype=GRB.BINARY, name="UseExterior1st")
UseExterior2nd = m.addVar(vtype=GRB.BINARY, name="UseExterior2nd")
SameMaterial   = m.addVar(vtype=GRB.BINARY, name="SameMaterial")

# (35) Σ_m Exterior1st_m = UseExterior1st
m.addConstr(
    gp.quicksum(Exterior1st[m_] for m_ in Exterior1st.keys()) == UseExterior1st,
    name="8.7.1_(35)_Exterior1st_sum"
)

# (36) Σ_m Exterior2nd_m = UseExterior2nd
m.addConstr(
    gp.quicksum(Exterior2nd[m_] for m_ in Exterior2nd.keys()) == UseExterior2nd,
    name="8.7.1_(36)_Exterior2nd_sum"
)

# (37) UseExterior1st = 1
m.addConstr(
    UseExterior1st == 1,
    name="8.7.1_(37)_UseExterior1st_fixed"
)

# (38) SameMaterial ≥ Exterior1st_m + Exterior2nd_m − 1   ∀m ∈ M
for m_ in Exterior1st.keys():
    m.addConstr(
        SameMaterial >= Exterior1st[m_] + Exterior2nd[m_] - 1,
        name=f"8.7.1_(38)_SameMaterial_{m_}"
    )

# (39) UseExterior2nd ≤ 1 − SameMaterial
m.addConstr(
    UseExterior2nd <= 1 - SameMaterial,
    name="8.7.1_(39)_UseExterior2nd_cond"
)

# ============================================================
# 8.7.2  Revestimiento de Mampostería
# ============================================================

# Conjunto T_masvnr = {BrkCmn, BrkFace, CBlock, None, Stone}
# Variables ya existentes: MasVnrType[·], MasVnrArea, TotalArea
HasMasVnr = m.addVar(vtype=GRB.BINARY, name="HasMasVnr")

# (40)  Σ_t MasVnrType_t = 1
m.addConstr(
    gp.quicksum(MasVnrType[t] for t in MasVnrType.keys()) == 1,
    name="8.7.2_(40)_MasVnrType_Unique"
)

# (41)  HasMasVnr = 1 − MasVnrType_None
m.addConstr(
    HasMasVnr == 1 - MasVnrType["None"],
    name="8.7.2_(41)_HasMasVnr_vs_None"
)

# (42)  MasVnrArea ≤ TotalArea · HasMasVnr
m.addConstr(
    MasVnrArea <= TotalArea * HasMasVnr,
    name="8.7.2_(42)_Area_if_HasMasVnr"
)

# (43)  MasVnrArea ≥ 0   (redundante si ya es lb=0, lo dejamos documentado)
m.addConstr(
    MasVnrArea >= 0,
    name="8.7.2_(43)_AreaNonNegative"
)


# ============================================================
# 8.8  Restricciones Adicionales
# ============================================================
# 8.8.1  Acceso Pavimentado
# ============================================================

# En Ames existe PavedDrive con claves típicas {'Y','P','N'}.
# El documento (44) usa sólo Y y N:  PavedDriveY + PavedDriveN = 1
# Dejamos una versión robusta que funciona con o sin 'P'.

if set(PavedDrive.keys()) >= {"Y", "N"} and "P" not in PavedDrive.keys():
    # (44) sólo Y/N
    m.addConstr(
        PavedDrive["Y"] + PavedDrive["N"] == 1,
        name="8.8.1_(44)_PavedDrive_YN_Only"
    )
else:
    # Si existe 'P' (Partial), o cualquier otro valor, forzamos elección única
    m.addConstr(
        gp.quicksum(PavedDrive[k] for k in PavedDrive.keys()) == 1,
        name="8.8.1_(44)_PavedDrive_Unique"
    )
    # (Opcional) si quieres replicar exactamente la página (sin 'P'), fija P=0:
    # if "P" in PavedDrive.keys():
    #     m.addConstr(PavedDrive["P"] == 0, name="8.8.1_opc_PavedDrive_BlockP")

# ============================================================
# 8.8.2  Servicios Públicos
# ============================================================

# Variables binarias Utilities_u ∈ {0,1} para u ∈ {AllPub, NoSewr, NoSeWa, ELO}
# (ya definidas en definir_variables(m))

# (45) Σ_u Utilities_u = 1
m.addConstr(
    gp.quicksum(Utilities[u] for u in Utilities.keys()) == 1,
    name="8.8.2_(45)_Utilities_Unique"
)


# ============================================================
# 8.8.3  Cerca / Reja
# ============================================================

# Binario "Reja" para indicar si se instala reja
Reja = m.addVar(vtype=GRB.BINARY, name="Reja")

# Enlace Reja <-> categorías de Fence
if "NA" in Fence.keys():
    # Reja existe si NO está la categoría NA
    m.addConstr(Reja == 1 - Fence["NA"], name="8.8.3_Reja_vs_NA")
else:
    # Si no existe NA, Reja es la suma de las categorías elegidas
    m.addConstr(
        Reja == gp.quicksum(Fence[k] for k in Fence.keys()),
        name="8.8.3_Reja_sum"
    )
    # (si tu modelo no fuerza unicidad en Fence, puedes añadirla)
    # m.addConstr(gp.quicksum(Fence[k] for k in Fence.keys()) <= 1, name="8.8.3_Fence_Unique")

# Condicionar la longitud de la reja a que exista reja (Big-M)
M_fence = 1000  # ajusta si tienes un mejor tope
m.addConstr(FenceLength <= M_fence * Reja, name="8.8.3_FenceLength_if_Reja")

# ============================================================
# 8.10  Restricciones del Garaje
# ============================================================

# ------------------------------------------------------------
# 8.10.1 Capacidad mínima
# ------------------------------------------------------------
# Variable binaria: 1 si la vivienda tiene garaje
HasGarage = m.addVar(vtype=GRB.BINARY, name="HasGarage")

# (48) GarageCars ≥ HasGarage
m.addConstr(
    GarageCars >= HasGarage,
    name="8.10.1_(48)_GarageCars_vs_HasGarage"
)

# ------------------------------------------------------------
# 8.10.2 Consistencia Área–Capacidad
# ------------------------------------------------------------
# (49) GarageArea ≥ 150 * GarageCars
m.addConstr(
    GarageArea >= 150 * GarageCars,
    name="8.10.2_(49)_GarageArea_Min"
)

# (50) GarageArea ≤ 250 * GarageCars
m.addConstr(
    GarageArea <= 250 * GarageCars,
    name="8.10.2_(50)_GarageArea_Max"
)

# ------------------------------------------------------------
# 8.10.3 Garaje Terminado
# ------------------------------------------------------------
# Restricción (51): GarageFinish = Fin si HasGarage = 1
#   → Si no hay garaje, no puede haber acabado.
#   → Si hay garaje, debe elegirse un acabado (Fin, RFn, Unf)
#   → Modelamos con: sum(GarageFinish) == HasGarage

m.addConstr(
    gp.quicksum(GarageFinish[f] for f in GarageFinish.keys()) == HasGarage,
    name="8.10.3_(51)_GarageFinish_if_HasGarage"
)

# ============================================================
# 8.10.4  Exclusividad de Tipo de Garaje
# ============================================================

# Variables binarias ya definidas en definir_variables:
# GarageType[t] para t ∈ {2Types, Attchd, Basment, BuiltIn, CarPort, Detchd, NA}

# (53) Selección única de tipo de garaje
m.addConstr(
    gp.quicksum(GarageType[t] for t in GarageType.keys()) == 1,
    name="8.10.4_(53)_GarageType_Unique"
)

# (54) Consistencia con existencia de garaje
# HasGarage = 1 - GarageType_NA
if "NA" in GarageType.keys():
    m.addConstr(
        HasGarage == 1 - GarageType["NA"],
        name="8.10.4_(54)_HasGarage_vs_NA"
    )

# ============================================================
# 8.11  Restricciones de Tipo de Vivienda
# ============================================================

# ------------------------------------------------------------
# 8.11.1  Exclusividad del Tipo de Vivienda
# ------------------------------------------------------------
# Variables binarias: BldgType[t] ∈ {0,1} para t ∈ {1Fam, 2FmCon, Duplx, TwnhsE, TwnhsI}

# (52) Selección única de tipo de vivienda
m.addConstr(
    gp.quicksum(BldgType[t] for t in BldgType.keys()) == 1,
    name="8.11.1_(52)_BldgType_Unique"
)

# ============================================================
# 8.12  Restricciones de Cimentación
# ============================================================

# 8.12.1  Exclusividad del Tipo de Cimentación
# Foundation[f] ∈ {0,1} para f ∈ {BrkTil, CBlock, PConc, Slab, Stone, Wood}

# (55) Selección única de cimentación
m.addConstr(
    gp.quicksum(Foundation[f] for f in Foundation.keys()) == 1,
    name="8.12.1_(55)_Foundation_Unique"
)

# ============================================================
# 8.13  Restricciones de Exposición del Sótano
# ============================================================

# 8.13.1  Exclusividad del Nivel de Exposición
# BsmtExposure[e] ∈ {0,1} para e ∈ {Gd, Av, Mn, No, NA}

# (56) Selección única de exposición de sótano
m.addConstr(
    gp.quicksum(BsmtExposure[e] for e in BsmtExposure.keys()) == 1,
    name="8.13.1_(56)_BsmtExposure_Unique"
)

# --- (Opcional pero útil) Consistencia con existencia de sótano ---
# Si eliges 'NA' → no hay sótano (TotalBsmtSF = 0)
if "NA" in BsmtExposure.keys():
    M_bsmt = 10000.0
    m.addConstr(TotalBsmtSF <= M_bsmt * (1 - BsmtExposure["NA"]),
                name="8.13.1_opc_NoBasementIfNA")
    # Y si hay sótano (>0), NA debe ser 0 (versión suave con epsilon)
    eps = 1.0
    m.addConstr(TotalBsmtSF >= eps * (1 - BsmtExposure["NA"]),
                name="8.13.1_opc_PositiveIfBasement")
    
# ============================================================
# 8.13.2  Dependencia con Existencia del Sótano
# ============================================================

# Binaria: 1 si la vivienda tiene sótano
HasBasement = m.addVar(vtype=GRB.BINARY, name="HasBasement")

# (57) BsmtExposure_NA = 1 − HasBasement
if "NA" in BsmtExposure.keys():
    m.addConstr(
        BsmtExposure["NA"] == 1 - HasBasement,
        name="8.13.2_(57)_NA_vs_HasBasement"
    )

# (58) BsmtExposure_Gd + Av + Mn + No = HasBasement
sum_exposure_exist = gp.quicksum(
    BsmtExposure[k] for k in BsmtExposure.keys() if k in ("Gd", "Av", "Mn", "No")
)
m.addConstr(
    sum_exposure_exist == HasBasement,
    name="8.13.2_(58)_ExposureSum_eq_HasBasement"
)

# (Opcional pero útil) ligar área de sótano a existencia
M_bsmt = 10000.0
m.addConstr(TotalBsmtSF <= M_bsmt * HasBasement, name="8.13.2_opc_BsmtArea_if_Has")

# ============================================================
# 8.15  Restricciones de Baños en Sótano
# ============================================================

# 8.15.1  Dependencia con Existencia del Sótano
# (60) BsmtFullBath ≤ TotalBsmtSF · HasBasement
m.addConstr(
    BsmtFullBath <= TotalBsmtSF * HasBasement,
    name="8.15.1_(60)_BsmtFullBath_if_Has"
)

# (61) BsmtHalfBath ≤ TotalBsmtSF · HasBasement
m.addConstr(
    BsmtHalfBath <= TotalBsmtSF * HasBasement,
    name="8.15.1_(61)_BsmtHalfBath_if_Has"
)

# (62) No negatividad (redundante si ya tienen lb=0)
m.addConstr(BsmtFullBath >= 0, name="8.15.1_(62a)_BsmtFullBath_ge0")
m.addConstr(BsmtHalfBath >= 0, name="8.15.1_(62b)_BsmtHalfBath_ge0")

# ============================================================
# 8.16  Restricciones de Habitaciones
# ============================================================

# 8.16.1  Mínimo de Habitaciones
# (63) TotRmsAbvGrd ≥ 1
m.addConstr(
    TotRmsAbvGrd >= 1,
    name="8.16.1_(63)_TotRmsAbvGrd_Min1"
)



# ======================
# Bloque de CÁLCULO DE COSTOS
# ======================

# Utilities
costo_utilities   = gp.quicksum(COSTOS["UTILITIES"][u]      * Utilities[u] for u in Utilities.keys())

# Ampliación / Demolición (áreas en ft²)
costo_ampliacion = COSTOS["AMPLIACION_PSF"] * AreaAmpliacionFt2
costo_demolicion = COSTOS["DEMOLICION_PSF"] * AreaDemolicionFt2

# Techo (material * área ≈ 1stFlrSF)
costo_roofmatl    = gp.quicksum(COSTOS["ROOFMATL_PSF"][m_]  * RoofMatl[m_]   * FirstFlrSF for m_ in RoofMatl.keys())
costo_roofstyle   = gp.quicksum(COSTOS.get("ROOFSTYLE", {}).get(s, 0.0) * RoofStyle[s] for s in RoofStyle.keys())

costo_total = costo_utilities + costo_ampliacion + costo_demolicion + costo_roofmatl

# Revestimiento exterior (1st y 2nd)
costo_exterior1 = gp.quicksum(COSTOS["EXTERIOR_PSF"][e] * Exterior1st[e] * FirstFlrSF
                              for e in COSTOS["EXTERIOR_PSF"])
costo_exterior2 = gp.quicksum(COSTOS["EXTERIOR_PSF"][e] * Exterior2nd[e] * FirstFlrSF
                              for e in COSTOS["EXTERIOR_PSF"])
costo_total += costo_exterior1 + costo_exterior2

# Mampostería exterior (MasVnrType)
costo_masvnr = gp.quicksum(COSTOS["MASVNRTYPE_PSF"][t] * MasVnrType[t] * FirstFlrSF
                           for t in COSTOS["MASVNRTYPE_PSF"])
costo_total += costo_masvnr

# Calefacción
costo_heating = costo_cat("HEATING", Heating)
costo_total += costo_heating

# Calidad calefacción, A/C, eléctrico
costo_heatingqc = gp.quicksum(COSTOS["HEATING_QC"][q] * HeatingQC[q]
                              for q in COSTOS["HEATING_QC"])
costo_centralair = costo_cat("CENTRALAIR", CentralAir)

costo_electrical  = costo_cat("ELECTRICAL", Electrical)
costo_total += costo_heatingqc + costo_centralair + costo_electrical

# Misceláneos
costo_misc = gp.quicksum(COSTOS["MISCFEATURE"][m_] * MiscFeature[m_]
                         for m_ in COSTOS["MISCFEATURE"])


# Entrada (proxy 1stFlrSF)
costo_paved = costo_cat("PAVED_DRIVE", PavedDrive)

# Sótano (proxy 1stFlrSF)
costo_basement = COSTOS["BASEMENT_PSF"]["Bsmt"] * FirstFlrSF

costo_total += costo_misc + costo_paved + costo_basement

# Calidad del sótano + Cocina (psf)
costo_bsmtcond = gp.quicksum(COSTOS["BASEMENTCOND"][c] * BasementCond[c]
                             for c in COSTOS["BASEMENTCOND"])
costo_kitchen = gp.quicksum(COSTOS["KITCHEN_PSF"][k] * Kitchen * FirstFlrSF
                            for k in COSTOS["KITCHEN_PSF"])
costo_total += costo_bsmtcond + costo_kitchen

# Remodel cocina + baños
costo_kitchen_remodel = gp.quicksum(COSTOS["KITCHEN_REMODEL"][r] * KitchenQual[r]
                                    for r in COSTOS["KITCHEN_REMODEL"])
costo_halfbath = gp.quicksum(COSTOS["HALFBATH_CONSTR"][h] * HalfBath
                             for h in COSTOS["HALFBATH_CONSTR"])
costo_fullbath = gp.quicksum(COSTOS["FULLBATH_CONSTR"][f] * FullBath
                             for f in COSTOS["FULLBATH_CONSTR"])
costo_total += costo_kitchen_remodel + costo_halfbath + costo_fullbath

# Remodel baño + chimeneas
costo_bath_remodel = gp.quicksum(COSTOS["BATH_REMODEL_PSF"][b] * FirstFlrSF
                                 for b in COSTOS["BATH_REMODEL_PSF"])
costo_fireplacequ = gp.quicksum(COSTOS["FIREPLACE_QU"][q] * FireplaceQu[q]
                                for q in COSTOS["FIREPLACE_QU"])
costo_total += costo_bath_remodel + costo_fireplacequ

# Dormitorios
costo_bedroom = gp.quicksum(COSTOS["BEDROOM_CONSTR"][b] * Bedroom
                            for b in COSTOS["BEDROOM_CONSTR"])

costo_bedrooms    = COSTOS.get("BEDROOM_PSF", 0.0) * AreaBedroom
costo_baths       = COSTOS.get("FULLBATH_PSF", 0.0) * AreaFullBath \
                    + COSTOS.get("HALFBATH_PSF", 0.0) * AreaHalfBath
costo_total += costo_bedroom

# Garaje: calidad + acabado
costo_garagequ = gp.quicksum(COSTOS["GARAGE_QU"][q] * GarageQu[q]
                             for q in COSTOS["GARAGE_QU"])
costo_garagefinish = gp.quicksum(
    COSTOS["GARAGE_FINISH"].get(f, 0) * GarageFinish[f]
    for f in GarageFinish.keys()
)

costo_total += costo_garagequ + costo_garagefinish

# Garaje: número de autos (categórico)
costo_garagecars = gp.quicksum(COSTOS["GARAGE_CARS_AREA"][g] * GarageCarsCat[g]
                               for g in COSTOS["GARAGE_CARS_AREA"])
costo_total += costo_garagecars

Garage = m.addVar(vtype=GRB.BINARY, name="Garage")
M_gar = 2000.0; eps_gar = 1.0
m.addConstr(GarageArea <= M_gar * Garage, name="8.9_Garage_link_ub")
m.addConstr(GarageArea >= eps_gar * Garage, name="8.9_Garage_link_lb")
costo_garage_area   = COSTOS.get("GARAGE_PSF", 0.0) * GarageArea
costo_garage_finish = gp.quicksum(COSTOS.get("GARAGE_FINISH", {}).get(f, 0.0) * GarageFinish[f] for f in GarageFinish.keys())

# Piscina: área + calidad
costo_pool        = COSTOS.get("POOL_PSF", 0.0) * AreaPool
costo_pool_base = COSTOS["POOL_COSTS"]["PoolArea"] * PoolArea
costo_pool_calidad = gp.quicksum(COSTOS["POOL_COSTS"][p] * PoolQC[p]
                                 for p in ["Ex", "Gd", "TA", "Fa"])
costo_total += costo_pool_base + costo_pool_calidad

# Cercado (pies lineales)
costo_fence = gp.quicksum(COSTOS["FENCE_COSTS"][f] * Fence[f] * FenceLength
                          for f in COSTOS["FENCE_COSTS"])

costo_reja = COSTOS.get("FENCE_LF", 0.0) * FenceLength \
                    + COSTOS.get("FENCE_FIXED", 0.0) * Reja

costo_total += costo_fence
costo_total += costo_roofstyle
costo_total += costo_reja
costo_total += costo_bedrooms + costo_baths
costo_total += costo_garage_area

# ======================
# Presupuesto y Objetivo
# ======================
m.addConstr(costo_total <= presupuesto, name="presupuesto")
m.setObjective(costo_total, GRB.MINIMIZE)

###################################################################################







m.Params.InfUnbdInfo = 1

m, vars_dict, expr_costo = construir_modelo_con_restricciones(presupuesto=presupuesto)
fijar_campos_en_modelo(m, vars_dict, FIXED)
m.optimize()


# === RESUMEN FINAL DEL MODELO ===
# ==================== MANEJO DE ESTADO + REPORTE ====================
import os, datetime
from colorama import Fore, Style
from gurobipy import GRB

# Carpeta de salida: Modelo_creacion_casa/resultados_casas_creadas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # .../Modelo_creacion_casa
OUT_DIR = os.path.join(BASE_DIR, "resultados_casas_creadas")
os.makedirs(OUT_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def print_header():
    print("\n" + "="*80)
    print(Fore.CYAN + Style.BRIGHT + "🏡  RESUMEN FINAL DEL DISEÑO DE CASA OPTIMIZADO" + Style.RESET_ALL)
    print("="*80)

def print_groups(model):
    # Agrupa y lista TODO lo activo (binarios=1, continuas>0)
    piezas, cocina, banos, garage, ampliaciones = [], [], [], [], []
    calidades, materiales, otros = [], [], []
    for v in model.getVars():
        if abs(v.X) > 1e-6:
            n = v.VarName.lower()
            if "bed" in n or "rms" in n or "room" in n:
                piezas.append(v)
            elif "kitchen" in n:
                cocina.append(v)
            elif "bath" in n:
                banos.append(v)
            elif "garage" in n:
                garage.append(v)
            elif "area" in n or "ampliacion" in n or "porch" in n or "pool" in n:
                ampliaciones.append(v)
            elif "qual" in n or "finish" in n or "qc" in n or "cond" in n:
                calidades.append(v)
            elif "roof" in n or "foundation" in n or "exterior" in n:
                materiales.append(v)
            else:
                otros.append(v)

    def pgroup(title, items):
        if items:
            print(Fore.CYAN + f"\n🔹 {title}" + Style.RESET_ALL)
            for v in items:
                if v.VType == GRB.BINARY and abs(v.X-1) < 1e-9:
                    print(f"   ✅ {v.VarName}")
                else:
                    print(f"   🔹 {v.VarName}: {v.X:.2f}")

    pgroup("Dormitorios / Habitaciones", piezas)
    pgroup("Cocina", cocina)
    pgroup("Baños", banos)
    pgroup("Garage", garage)
    pgroup("Ampliaciones y Superficies", ampliaciones)
    pgroup("Calidades / Terminaciones", calidades)
    pgroup("Materiales estructurales", materiales)
    pgroup("Otros componentes", otros)

# ---- ESTADO DEL SOLVER ----
status = m.Status
print_header()

if status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) or (status == GRB.TIME_LIMIT and m.SolCount > 0):
    # Hay solución: imprimimos y guardamos
    presupuesto_inicial = presupuesto  # ajusta si usas otra variable/param
    costo_total_optimo = m.ObjVal
    presupuesto_restante = presupuesto_inicial - costo_total_optimo

    print(f"{Fore.YELLOW}💰 Presupuesto inicial :{Style.RESET_ALL} ${presupuesto_inicial:,.0f}")
    print(f"{Fore.RED}💸 Costo total usado   :{Style.RESET_ALL} ${costo_total_optimo:,.0f}")
    print(f"{Fore.GREEN}💵 Presupuesto restante:{Style.RESET_ALL} ${presupuesto_restante:,.0f}")
    print("-"*80)

    print_groups(m)

    # Guardado “bonito” en TXT
    out_txt = os.path.join(OUT_DIR, f"resultado_modelo_{timestamp}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("🏡 RESULTADOS DEL MODELO DE DISEÑO DE CASA\n")
        f.write(f"Presupuesto inicial: ${presupuesto_inicial:,.0f}\n")
        f.write(f"Costo total: ${costo_total_optimo:,.0f}\n")
        f.write(f"Presupuesto restante: ${presupuesto_restante:,.0f}\n\n")
        f.write("Variables activas:\n")
        for v in m.getVars():
            if abs(v.X) > 1e-6:
                f.write(f"{v.VarName}: {v.X}\n")

    # Guardamos la solución y el LP/QP del modelo
    m.write(os.path.join(OUT_DIR, f"solucion_{timestamp}.sol"))
    m.write(os.path.join(OUT_DIR, f"modelo_{timestamp}.lp"))  # útil para revisar

    print("\n" + "="*80)
    print(f"{Fore.MAGENTA}📊 Estado del modelo: {status}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}🗂 Guardados: {out_txt.split('/')[-1]}, solucion_{timestamp}.sol, modelo_{timestamp}.lp{Style.RESET_ALL}")
    print("="*80)

elif status == GRB.INFEASIBLE:
    print(Fore.RED + "❌ El modelo es INVIABLE." + Style.RESET_ALL)
    # IIS para diagnosticar
    m.computeIIS()
    iis_path = os.path.join(OUT_DIR, f"infeasible_{timestamp}.ilp")
    m.write(iis_path)   # contiene el IIS (restricciones/vars que causan inviabilidad)
    m.write(os.path.join(OUT_DIR, f"modelo_{timestamp}.lp"))
    print(Fore.YELLOW + f"🔎 IIS escrito en: {iis_path}" + Style.RESET_ALL)
    print(Fore.YELLOW + f"🔎 Modelo LP en: modelo_{timestamp}.lp" + Style.RESET_ALL)
    print(Style.RESET_ALL)

elif status == GRB.UNBOUNDED:
    print(Fore.RED + "⚠️ El modelo es NO ACOTADO." + Style.RESET_ALL)
    m.write(os.path.join(OUT_DIR, f"modelo_{timestamp}.lp"))
    print(Fore.YELLOW + f"🔎 Revisa costos negativos y variables sin cotas. Exportado modelo_{timestamp}.lp" + Style.RESET_ALL)

else:
    print(Fore.RED + f"⚠️ Estado del solver: {status}. No hay solución para imprimir." + Style.RESET_ALL)
    m.write(os.path.join(OUT_DIR, f"modelo_{timestamp}.lp"))


# =============================================
# PRUEBA BÁSICA
# =============================================
#if __name__ == "__main__":
    presupuesto = 12000000  # o el valor que quieras

    # Casa base (la misma que usaste)
    base = {
        "MS SubClass": 20,
        "MS Zoning": "RL",
        "Lot Frontage": 141.0,
        "Lot Area": 31770,
        "Street": "Pave",
        "Alley": "No aplica",
        "Lot Shape": "IR1",
        "Land Contour": "Lvl",
        "Utilities": "AllPub",
        "Lot Config": "Corner",
        "Land Slope": "Gtl",
        "Neighborhood": "No aplicames",
        "Condition 1": "Norm",
        "Condition 2": "Norm",
        "Bldg Type": "1Fam",
        "House Style": "1Story",
        "Overall Qual": 6,
        "Overall Cond": 5,
        "Year Built": 1960,
        "Year Remod/Add": 1960,
        "Roof Style": "Hip",
        "Roof Matl": "CompShg",
        "Exterior 1st": "BrkFace",
        "Exterior 2nd": "Plywood",
        "Mas Vnr Type": "Stone",
        "Mas Vnr Area": 112.0,
        "Exter Qual": "TA",
        "Exter Cond": "TA",
        "Foundation": "CBlock",
        "Bsmt Qual": "TA",
        "Bsmt Cond": "Gd",
        "Bsmt Exposure": "Gd",
        "BsmtFin Type 1": "BLQ",
        "BsmtFin SF 1": 639.0,
        "BsmtFin Type 2": "Unf",
        "BsmtFin SF 2": 0.0,
        "Bsmt Unf SF": 441.0,
        "Total Bsmt SF": 1080.0,
        "Heating": "GasA",
        "Heating QC": "Fa",
        "Central Air": "Y",
        "Electrical": "SBrkr",
        "1st Flr SF": 1656,
        "2nd Flr SF": 0,
        "Low Qual Fin SF": 0,
        "Gr Liv Area": 1656,
        "Bsmt Full Bath": 1.0,
        "Bsmt Half Bath": 0.0,
        "Full Bath": 1,
        "Half Bath": 0,
        "Bedroom AbvGr": 3,
        "Kitchen AbvGr": 1,
        "Kitchen Qual": "TA",
        "TotRms AbvGrd": 7,
        "FunctioNo aplical": "Typ",
        "Fireplaces": 2,
        "Fireplace Qu": "Gd",
        "Garage Type": "Attchd",
        "Garage Yr Blt": 1960.0,
        "Garage Finish": "Fin",
        "Garage Cars": 2.0,
        "Garage Area": 528.0,
        "Garage Qual": "TA",
        "Garage Cond": "TA",
        "Paved Drive": "P",
        "Wood Deck SF": 210,
        "Open Porch SF": 62,
        "Enclosed Porch": 0,
        "3Ssn Porch": 0,
        "Screen Porch": 0,
        "Pool Area": 0,
        "Pool QC": "No aplica",
        "Fence": "No aplica",
        "Misc Feature": "No aplica",
        "Misc Val": 0,
        "Mo Sold": 5,
        "Yr Sold": 2010,
        "Sale Type": "WD",
        "Sale Condition": "Normal"
    }

    mejor, todas = buscar_mejor_combinacion(presupuesto, base)


#if __name__ == "__main__":
    # 1) construye el modelo con TODAS TUS RESTRICCIONES + presupuesto
    m, vars_dict, expr_costo = construir_modelo_con_restricciones(presupuesto=presupuesto)

    # 2) optimiza buscando factible (objetivo = 0)
    m.optimize()

    if m.status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        print("⚠️  No se encontró solución factible con las restricciones actuales.")
    else:
        # 3) extrae la solución en el formato EXACTO que espera el XGB
        decision = extraer_decision(vars_dict)

        # 4) evalúa con tu pipeline (XGB + costo)
        precio, costo = evaluar_casa(decision)

        print("\n✅ Solución factible evaluada por XGB:")
        print(f"💰 Precio predicho: {precio:,.0f}")
        print(f"🏗️  Costo total  : {costo:,.0f}")
        print(f"📉 Gap presupuesto: {presupuesto - costo:,.0f}")


if __name__ == "__main__":
    m, vars_dict, expr_costo = construir_modelo_con_restricciones(presupuesto=presupuesto)

    # pool!
    m.setParam("PoolSearchMode", 2)
    m.setParam("PoolSolutions", 200)

    m.optimize()

    if m.SolCount > 0:
        mejor_pool, df_pool = evaluar_solution_pool(m, vars_dict, max_solutions=200)
    else:
        print("⚠️ El pool no devolvió soluciones.")
