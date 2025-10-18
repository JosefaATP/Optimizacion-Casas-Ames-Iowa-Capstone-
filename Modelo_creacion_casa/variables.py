import gurobipy as gp
from gurobipy import GRB


# Presupuesto global por defecto (ajústalo a tu caso)
PRESUPUESTO = 12_000_000  # CLP

# DEFAULTS: fila base con TODOS los campos que tu XGB espera.
# Usa tu ejemplo real como base. Ajusta si alguna clave difiere.
DEFAULTS = {
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

# Ejemplo: valores que NO se pueden cambiar
FIXED = {
    "Neighborhood": "OldTown",
    "MS Zoning": "RL",
    "Lot Area": 8000,
    "Lot Frontage": 60.0,
    "Street": "Pave",
    # agrega lo que venga “dado por el lote”
}
VARIABLES = {
    # numéricas (rangos orientativos)
    "Bedroom AbvGr": (1, 6),
    "Full Bath": (1, 3),
    "Half Bath": (0, 2),
    "Gr Liv Area": (700, 2600),
    "1st Flr SF": (600, 2000),

    # categóricas (dominios; podrías tomarlo de DOMINIOS si ya lo tienes)
    "Roof Matl": ["CompShg", "Metal", "WdShngl", "TarGrv"],
    "Heating": ["GasA", "GasW", "Floor", "Grav"],
    "Central Air": ["Y", "N"],
    "Exterior 1st": ["VinylSd", "BrkFace", "MetalSd", "Wd Sdng"],
    "Bldg Type": ["1Fam", "TwnhsE", "TwnhsI"],
    "House Style": ["1Story", "2Story", "SLvl"],
}

def definir_variables(m):
    # Variables binarias para variables categóricas
    MSSubClass = m.addVars([20,30,40,45,50,60,70,75,80,85,90,120,150,160,180,190], vtype=GRB.BINARY, name="MSSubClass")
    Utilities  = m.addVars(["AllPub","NoSewr","NoSeWa","ELO"], vtype=GRB.BINARY, name="Utilities")
    BldgType   = m.addVars(["1Fam","2FmCon","Duplx","TwnhsE","TwnhsI"], vtype=GRB.BINARY, name="BldgType")
    HouseStyle = m.addVars(["1Story","1.5Fin","1.5Unf","2Story","2.5Fin","2.5Unf","SFoyer","SLvl"], vtype=GRB.BINARY, name="HouseStyle")
    RoofStyle  = m.addVars(["Flat","Gable","Gambrel","Hip","Mansard","Shed"], vtype=GRB.BINARY, name="RoofStyle")
    RoofMatl   = m.addVars(["ClyTile","CompShg","Membran","Metal","Roll","TarGrv","WdShake","WdShngl"], vtype=GRB.BINARY, name="RoofMatl")
    Exterior1st = m.addVars(["AsbShng","AsphShn","BrkComm","BrkFace","CBlock","CemntBd","HdBoard","ImStucc","MetalSd","Other","Plywood","PreCast","Stone","Stucco","VinylSd","WdSdng","WdShing"], vtype=GRB.BINARY, name="Exterior1st")
    Exterior2nd = m.addVars(["AsbShng","AsphShn","BrkComm","BrkFace","CBlock","CemntBd","HdBoard","ImStucc","MetalSd","Other","Plywood","PreCast","Stone","Stucco","VinylSd","WdSdng","WdShing"], vtype=GRB.BINARY, name="Exterior2nd")
    MasVnrType = m.addVars(["BrkCmn","BrkFace","CBlock","None","Stone"], vtype=GRB.BINARY, name="MasVnrType")
    MasVnrArea = m.addVar(vtype=GRB.INTEGER, lb=0, name="MasVnrArea")
    Foundation = m.addVars(["BrkTil","CBlock","PConc","Slab","Stone","Wood"], vtype=GRB.BINARY, name="Foundation")
    BsmtExposure = m.addVars(["Gd","Av","Mn","No","NA"], vtype=GRB.BINARY, name="BsmtExposure")
    BsmtFinType1 = m.addVars(["GLQ","ALQ","BLQ","Rec","LwQ","Unf","NA"], vtype=GRB.BINARY, name="BsmtFinType1")
    BsmtFinSF1   = m.addVar(vtype=GRB.INTEGER, lb=0, name="BsmtFinSF1")
    BsmtFinType2 = m.addVars(["GLQ","ALQ","BLQ","Rec","LwQ","Unf","NA"], vtype=GRB.BINARY, name="BsmtFinType2")
    BsmtFinSF2   = m.addVar(vtype=GRB.INTEGER, lb=0, name="BsmtFinSF2")
    BsmtUnfSF    = m.addVar(vtype=GRB.INTEGER, lb=0, name="BsmtUnfSF")
    TotalBsmtSF  = m.addVar(vtype=GRB.INTEGER, lb=0, name="TotalBsmtSF")
    Heating      = m.addVars(["Floor","GasA","GasW","Grav","OthW","Wall"], vtype=GRB.BINARY, name="Heating")
    CentralAir   = m.addVars(["Yes","No"], vtype=GRB.BINARY, name="CentralAir")
    Electrical   = m.addVars(["SBrkr","FuseA","FuseF","FuseP","Mix"], vtype=GRB.BINARY, name="Electrical")

    FirstFlrSF   = m.addVar(vtype=GRB.INTEGER, lb=0, name="1stFlrSF")
    SecondFlrSF  = m.addVar(vtype=GRB.INTEGER, lb=0, name="2ndFlrSF")
    LowQualFinSF = m.addVar(vtype=GRB.INTEGER, lb=0, name="LowQualFinSF")

    GrLivArea    = m.addVar(vtype=GRB.INTEGER, lb=0, name="GrLivArea")

    BsmtFullBath = m.addVar(vtype=GRB.INTEGER, lb=0, name="BsmtFullBath")
    BsmtHalfBath = m.addVar(vtype=GRB.INTEGER, lb=0, name="BsmtHalfBath")
    FullBath       = m.addVar(vtype=GRB.INTEGER, lb=0, name="FullBath")
    HalfBath       = m.addVar(vtype=GRB.INTEGER, lb=0, name="HalfBath")
    Bedroom        = m.addVar(vtype=GRB.INTEGER, lb=0, name="Bedroom")
    Kitchen        = m.addVar(vtype=GRB.INTEGER, lb=0, name="Kitchen")
    TotRmsAbvGrd   = m.addVar(vtype=GRB.INTEGER, lb=0, name="TotRmsAbvGrd")
    Fireplaces     = m.addVar(vtype=GRB.INTEGER, lb=0, name="Fireplaces")

    GarageType     = m.addVars(["2Types","Attchd","Bsmt","BuiltIn","CarPort","Detchd","Noaplica"],
                            vtype=GRB.BINARY, name="GarageType")
    GarageFinish   = m.addVars(["Fin","RFn","Unf","Noaplica"],
                            vtype=GRB.BINARY, name="GarageFinish")
    GarageCars     = m.addVar(vtype=GRB.INTEGER, lb=0, name="GarageCars")
    GarageArea     = m.addVar(vtype=GRB.INTEGER, lb=0, name="GarageArea")
    PavedDrive   = m.addVars(["Paved","PartialPavement","Dirt/Gravel"],vtype=GRB.BINARY, name="PavedDrive")
    
    WoodDeckSF   = m.addVar(vtype=GRB.INTEGER, lb=0, name="WoodDeckSF")
    OpenPorchSF  = m.addVar(vtype=GRB.INTEGER, lb=0, name="OpenPorchSF")
    EnclosedPorch = m.addVar(vtype=GRB.INTEGER, lb=0, name="EnclosedPorch")
    ThreeSsnPorch = m.addVar(vtype=GRB.INTEGER, lb=0, name="3SsnPorch")
    ScreenPorch   = m.addVar(vtype=GRB.INTEGER, lb=0, name="ScreenPorch")
    PoolArea      = m.addVar(vtype=GRB.INTEGER, lb=0, name="PoolArea")
    MiscFeature   = m.addVars(["Elev","Gar2","Othr","Shed","TenC","Noaplica"],
                          vtype=GRB.BINARY, name="MiscFeature")

    SaleType      = m.addVars(["WD","CWD","VWD","New","COD","Con","ConLw","ConLI","ConLD","Oth"],
                            vtype=GRB.BINARY, name="SaleType")

    SaleCondition = m.addVars(["Normal","Abnorml","AdjLand","Alloca","Family","Partial"],
                            vtype=GRB.BINARY, name="SaleCondition")
    

    FenceLength = m.addVar(vtype=GRB.INTEGER, lb=0, name="FenceLength")

    AreaAmpliacionFt2 = m.addVar(vtype=GRB.INTEGER, lb=0, name="AreaAmpliacionFt2")
    AreaDemolicionFt2 = m.addVar(vtype=GRB.INTEGER, lb=0, name="AreaDemolicionFt2")

    HeatingQC   = m.addVars(["Ex","Gd","TA","Fa","Po"], vtype=GRB.BINARY, name="HeatingQC")
    BasementCond= m.addVars(["Ex","Gd","TA","Fa","Po","NA"], vtype=GRB.BINARY, name="BasementCond")
    KitchenQual = m.addVars(["Ex","Gd","TA","Fa","Po"], vtype=GRB.BINARY, name="KitchenQual")
    FireplaceQu = m.addVars(["Ex","Gd","TA","Fa","Po","NA"], vtype=GRB.BINARY, name="FireplaceQu")
    PoolQC      = m.addVars(["Ex","Gd","TA","Fa"], vtype=GRB.BINARY, name="PoolQC")
    GarageQu    = m.addVars(["Ex","Gd","TA","Fa","Po","NA"], vtype=GRB.BINARY, name="GarageQu")

    GarageCarsCat = m.addVars(["1car","2car","3car","4car"], vtype=GRB.BINARY, name="GarageCarsCat")

    Fence = m.addVars(["Fence","GdPrv","MnPrv","GdWo","MnWw","NA"],vtype=GRB.BINARY, name="Fence")






    m.addConstr(gp.quicksum(MSSubClass[s] for s in [20,30,40,45,50,60,70,75,80,85,90,120,150,160,180,190]) == 1)
    m.addConstr(gp.quicksum(Utilities[u]  for u in ["AllPub","NoSewr","NoSeWa","ELO"]) == 1)
    m.addConstr(gp.quicksum(BldgType[b]  for b in ["1Fam","2FmCon","Duplx","TwnhsE","TwnhsI"]) == 1)
    m.addConstr(gp.quicksum(HouseStyle[hs] for hs in ["1Story","1.5Fin","1.5Unf","2Story","2.5Fin","2.5Unf","SFoyer","SLvl"]) == 1)
    m.addConstr(gp.quicksum(RoofStyle[r]  for r in ["Flat","Gable","Gambrel","Hip","Mansard","Shed"]) == 1)
    m.addConstr(gp.quicksum(RoofMatl[m_]  for m_ in ["ClyTile","CompShg","Membran","Metal","Roll","TarGrv","WdShake","WdShngl"]) == 1)
    m.addConstr(gp.quicksum(Exterior1st[e1] for e1 in ["AsbShng","AsphShn","BrkComm","BrkFace","CBlock","CemntBd","HdBoard","ImStucc","MetalSd","Other","Plywood","PreCast","Stone","Stucco","VinylSd","WdSdng","WdShing"]) == 1)
    m.addConstr(gp.quicksum(Exterior2nd[e2] for e2 in ["AsbShng","AsphShn","BrkComm","BrkFace","CBlock","CemntBd","HdBoard","ImStucc","MetalSd","Other","Plywood","PreCast","Stone","Stucco","VinylSd","WdSdng","WdShing"]) == 1)
    m.addConstr(gp.quicksum(MasVnrType[t] for t in ["BrkCmn","BrkFace","CBlock","None","Stone"]) == 1)
    m.addConstr(gp.quicksum(Foundation[f] for f in ["BrkTil","CBlock","PConc","Slab","Stone","Wood"]) == 1)
    m.addConstr(gp.quicksum(BsmtExposure[x] for x in ["Gd","Av","Mn","No","NA"]) == 1)
    m.addConstr(gp.quicksum(BsmtFinType1[b1] for b1 in ["GLQ","ALQ","BLQ","Rec","LwQ","Unf","NA"]) == 1)
    m.addConstr(gp.quicksum(BsmtFinType2[b2] for b2 in ["GLQ","ALQ","BLQ","Rec","LwQ","Unf","NA"]) == 1)
    m.addConstr(gp.quicksum(Heating[h] for h in ["Floor","GasA","GasW","Grav","OthW","Wall"]) == 1)
    m.addConstr(gp.quicksum(CentralAir[a] for a in ["Yes","No"]) == 1)
    m.addConstr(gp.quicksum(Electrical[e] for e in ["SBrkr","FuseA","FuseF","FuseP","Mix"]) == 1)

    m.addConstr(TotalBsmtSF == BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF)
    m.addConstr(GrLivArea == FirstFlrSF
            + SecondFlrSF
            + LowQualFinSF)
    m.addConstr(gp.quicksum(GarageType[g] for g in ["2Types","Attchd","Bsmt","BuiltIn","CarPort","Detchd","Noaplica"]) == 1)
    m.addConstr(gp.quicksum(GarageFinish[gf] for gf in ["Fin","RFn","Unf","Noaplica"]) == 1)
    M_g = 10000
    m.addConstr(GarageCars <= M_g * (1 - GarageType["Noaplica"]))  #Si no aplica GarageCars=0
    m.addConstr(GarageArea <= M_g * (1 - GarageType["Noaplica"])) #Si no aplica GarageArea=0
    m.addConstr(gp.quicksum(PavedDrive[p] for p in ["Paved","PartialPavement","Dirt/Gravel"]) == 1)
    m.addConstr(gp.quicksum(MiscFeature[misc] for misc in ["Elev","Gar2","Othr","Shed","TenC","Noaplica"]) == 1)
    m.addConstr(gp.quicksum(SaleType[st]     for st   in ["WD","CWD","VWD","New","COD","Con","ConLw","ConLI","ConLD","Oth"]) == 1)
    m.addConstr(gp.quicksum(SaleCondition[sc] for sc  in ["Normal","Abnorml","AdjLand","Alloca","Family","Partial"]) == 1)

    m.addConstr(gp.quicksum(GarageCarsCat[g] for g in ["1car","2car","3car","4car"]) == 1)

    m.addConstr(gp.quicksum(GarageCarsCat[g] for g in ["1car","2car","3car","4car"]) == 1)
    m.addConstr(GarageCars == 1*GarageCarsCat["1car"] + 2*GarageCarsCat["2car"] + 3*GarageCarsCat["3car"] + 4*GarageCarsCat["4car"])
    m.addConstr(gp.quicksum(Fence[f] for f in ["Fence","GdPrv","MnPrv","GdWo","MnWw","NA"]) == 1)

    M_f = 100000
    m.addConstr(FenceLength <= M_f * (1 - Fence["NA"]))

    ######################################################################################


    






















    #############################################################################################





    return MSSubClass, Utilities, BldgType, HouseStyle, RoofStyle, RoofMatl, Exterior1st, Exterior2nd, \
       MasVnrType, MasVnrArea, Foundation, BsmtExposure, BsmtFinType1, BsmtFinSF1, \
       BsmtFinType2, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, Heating, CentralAir, Electrical, \
       FirstFlrSF, SecondFlrSF, LowQualFinSF, GrLivArea, BsmtFullBath, BsmtHalfBath, \
       FullBath, HalfBath, Bedroom, Kitchen, TotRmsAbvGrd, Fireplaces, \
       GarageType, GarageFinish, GarageCars, GarageArea, \
       PavedDrive, WoodDeckSF, OpenPorchSF, EnclosedPorch, ThreeSsnPorch, ScreenPorch, PoolArea, \
       MiscFeature, SaleType, SaleCondition, FenceLength, AreaAmpliacionFt2, AreaDemolicionFt2, \
       HeatingQC, BasementCond, KitchenQual, FireplaceQu, PoolQC, GarageQu, GarageCarsCat, Fence


