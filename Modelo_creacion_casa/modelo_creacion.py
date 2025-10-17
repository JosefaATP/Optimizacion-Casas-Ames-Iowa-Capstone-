import gurobipy as gp
from gurobipy import GRB
from costos import COSTOS
from variables import definir_variables

# Parámetros del proyecto
presupuesto = 60000
limite_area = 80  # m2  (no usado aquí)

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

# ======================
# Bloque de CÁLCULO DE COSTOS
# ======================

# Utilities
costo_utilities = gp.quicksum(COSTOS["UTILITIES"][u] * Utilities[u]
                              for u in COSTOS["UTILITIES"])

# Ampliación / Demolición (áreas en ft²)
costo_ampliacion = COSTOS["AMPLIACION_PSF"] * AreaAmpliacionFt2
costo_demolicion = COSTOS["DEMOLICION_PSF"] * AreaDemolicionFt2

# Techo (material * área ≈ 1stFlrSF)
costo_roofmatl = gp.quicksum(COSTOS["ROOFMATL_PSF"][m_] * RoofMatl[m_] * FirstFlrSF
                             for m_ in COSTOS["ROOFMATL_PSF"])

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
costo_heating = gp.quicksum(COSTOS["HEATING"][h] * Heating[h]
                            for h in COSTOS["HEATING"])
costo_total += costo_heating

# Calidad calefacción, A/C, eléctrico
costo_heatingqc = gp.quicksum(COSTOS["HEATING_QC"][q] * HeatingQC[q]
                              for q in COSTOS["HEATING_QC"])
costo_centralair = gp.quicksum(COSTOS["CENTRALAIR"][a] * CentralAir[a]
                               for a in COSTOS["CENTRALAIR"])
costo_electrical = gp.quicksum(COSTOS["ELECTRICAL"][e] * Electrical[e]
                               for e in COSTOS["ELECTRICAL"])
costo_total += costo_heatingqc + costo_centralair + costo_electrical

# Misceláneos
costo_misc = gp.quicksum(COSTOS["MISCFEATURE"][m_] * MiscFeature[m_]
                         for m_ in COSTOS["MISCFEATURE"])


# Entrada (proxy 1stFlrSF)
costo_paved = gp.quicksum(COSTOS["PAVEDDRIVE_PSF"][p] * PavedDrive[p] * FirstFlrSF
                          for p in COSTOS["PAVEDDRIVE_PSF"])

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

# Piscina: área + calidad
costo_pool_base = COSTOS["POOL_COSTS"]["PoolArea"] * PoolArea
costo_pool_calidad = gp.quicksum(COSTOS["POOL_COSTS"][p] * PoolQC[p]
                                 for p in ["Ex", "Gd", "TA", "Fa"])
costo_total += costo_pool_base + costo_pool_calidad

# Cercado (pies lineales)
costo_fence = gp.quicksum(COSTOS["FENCE_COSTS"][f] * Fence[f] * FenceLength
                          for f in COSTOS["FENCE_COSTS"])
costo_total += costo_fence

# ======================
# Presupuesto y Objetivo
# ======================
m.addConstr(costo_total <= presupuesto, name="presupuesto")
m.setObjective(costo_total, GRB.MINIMIZE)

# (Opcional) Ejecutar y mostrar costo mínimo
# m.optimize()
# if m.Status == GRB.OPTIMAL:
#     print(f"Costo mínimo: ${m.ObjVal:,.2f}")
m.optimize()

# === RESUMEN FINAL DEL MODELO ===
from colorama import Fore, Style
import datetime

print("\n" + "="*80)
print(Fore.CYAN + Style.BRIGHT + "🏡  RESUMEN FINAL DEL DISEÑO DE CASA OPTIMIZADO" + Style.RESET_ALL)
print("="*80)

# 💰 Presupuesto
presupuesto_inicial = presupuesto  # ajusta si tu variable se llama distinto
costo_total_optimo = m.ObjVal
presupuesto_restante = presupuesto_inicial - costo_total_optimo

print(f"{Fore.YELLOW}💰 Presupuesto inicial :{Style.RESET_ALL} ${presupuesto_inicial:,.0f}")
print(f"{Fore.RED}💸 Costo total usado   :{Style.RESET_ALL} ${costo_total_optimo:,.0f}")
print(f"{Fore.GREEN}💵 Presupuesto restante:{Style.RESET_ALL} ${presupuesto_restante:,.0f}")
print("-"*80)

# Agrupar variables por tipo lógico
piezas = []
cocina = []
banos = []
garage = []
ampliaciones = []
calidades = []
materiales = []
otros = []

for v in m.getVars():
    if abs(v.X) > 1e-6:
        nombre = v.VarName.lower()
        if "bed" in nombre or "room" in nombre or "rms" in nombre:
            piezas.append(v)
        elif "kitchen" in nombre:
            cocina.append(v)
        elif "bath" in nombre:
            banos.append(v)
        elif "garage" in nombre:
            garage.append(v)
        elif "area" in nombre or "ampliacion" in nombre or "porch" in nombre:
            ampliaciones.append(v)
        elif "qual" in nombre or "finish" in nombre or "qc" in nombre or "cond" in nombre:
            calidades.append(v)
        elif "roof" in nombre or "foundation" in nombre or "exterior" in nombre:
            materiales.append(v)
        else:
            otros.append(v)

def print_group(title, vars_list):
    if vars_list:
        print(Fore.CYAN + f"\n🔹 {title}" + Style.RESET_ALL)
        for v in vars_list:
            if v.VType == 'B' and v.X == 1:
                print(f"   ✅ {v.VarName}")
            elif v.VType != 'B':
                print(f"   🔹 {v.VarName}: {v.X:.2f}")

# Mostrar por secciones
print_group("Dormitorios / Habitaciones", piezas)
print_group("Cocina", cocina)
print_group("Baños", banos)
print_group("Garage", garage)
print_group("Ampliaciones y Superficies", ampliaciones)
print_group("Calidades / Terminaciones", calidades)
print_group("Materiales estructurales", materiales)
print_group("Otros componentes", otros)

# Listado total
print(Fore.MAGENTA + "\n📋 LISTADO COMPLETO DE VARIABLES ACTIVAS" + Style.RESET_ALL)
for v in m.getVars():
    if abs(v.X) > 1e-6:
        tipo = "BIN" if v.VType == "B" else "CONT"
        print(f"   {v.VarName:<35} {v.X:>10.2f}   ({tipo})")

# Guardar resumen opcional
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
with open(f"resultado_modelo_{timestamp}.txt", "w", encoding="utf-8") as f:
    f.write("🏡 RESULTADOS DEL MODELO DE DISEÑO DE CASA\n")
    f.write(f"Presupuesto inicial: ${presupuesto_inicial:,.0f}\n")
    f.write(f"Costo total: ${costo_total_optimo:,.0f}\n")
    f.write(f"Presupuesto restante: ${presupuesto_restante:,.0f}\n\n")
    f.write("Variables activas:\n")
    for v in m.getVars():
        if abs(v.X) > 1e-6:
            f.write(f"{v.VarName}: {v.X}\n")

print("\n" + "="*80)
print(f"{Fore.MAGENTA}📊 Estado del modelo: {m.Status}{Style.RESET_ALL}")
print(f"{Fore.BLUE}🕒 Resultado guardado en archivo: resultado_modelo_{timestamp}.txt{Style.RESET_ALL}")
print("="*80)
print(Style.RESET_ALL)
