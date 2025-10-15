# optimization/remodel/gurobi_model.py
from typing import Optional, Dict, Any, List, Tuple
from gurobipy import Model, GRB, quicksum

# -------------------------
# Helpers (calidades, alias)
# -------------------------
QUAL_ORDER = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}
def qscore(x): return QUAL_ORDER.get(str(x).strip(), 3)

def canon(col, v):
    if col == "Central Air":
        s = str(v).strip()
        return {"Y":"Yes","N":"No"}.get(s, s)
    if col in ("Mas Vnr Type", "Pool QC"):
        s = "" if v is None else str(v).strip()
        return "None" if s in ("", "NA", "No aplica", "None") else s
    return v

# ------------------------------------------------------
# Estructuras esperadas (siguen lo que ya venías usando)
# ------------------------------------------------------
class BaseHouse:
    def __init__(self, features: Dict[str, Any]):
        self.features = features

class CostTables:
    def __init__(self, **kwargs):
        # valores esperados en YAML (ajusta nombres si ocupas otros)
        self.utilities       = kwargs.get("utilities", {})
        self.roof_style      = kwargs.get("RoofStyle", kwargs.get("roof_style", {}))
        self.roof_matl       = kwargs.get("RoofMatl",  kwargs.get("roof_matl", {}))
        self.exterior1st     = kwargs.get("Exterior1st", {})
        self.exterior2nd     = kwargs.get("Exterior2nd", {})
        self.mas_vnr_type    = kwargs.get("MasVnrType", {})
        self.electrical      = kwargs.get("Electrical", {})
        self.heating         = kwargs.get("Heating", {})
        self.kitchen_qual    = kwargs.get("KitchenQual", {})
        self.central_air_install = kwargs.get("CentralAirInstall", 5362.0)  # default razonable
        # costos por m2 / ft2 de ampliaciones (si faltan en YAML uso defaults)
        self.cost_finish_bsmt_ft2 = kwargs.get("CBsmt", 15.0)         # Basement (ft2)
        self.cost_pool_ft2        = kwargs.get("PoolCostPerFt2", 70.) # Piscina (ft2)
        self.cost_addition_ft2    = kwargs.get("Cstr_floor", 106.49)  # Ampliación (ft2)
        self.cost_demolition_ft2  = kwargs.get("Cdemolition", 1.65)   # Demolición (ft2)
        # Kitchen remodel (por calidad, si falta algo interpolas en YAML)
        self.kitchen_remodel      = kwargs.get("KitchenRemodel", {"Ex":180000,"TA":42500,"Po":13000})

class PoolRules:
    def __init__(self, **kwargs):
        self.max_share = kwargs.get("max_share", 0.20)
        self.min_area_per_quality = kwargs.get("min_area_per_quality", 100)

class Compat:
    def __init__(self, roof_style_by_matl_forbidden: List[Tuple[str,str]]):
        self.roof_style_by_matl_forbidden = set(roof_style_by_matl_forbidden)

# ==============================
# MODELO MILP – REMODELACIÓN
# ==============================
class RemodelMILP:
    """
    Este modelo SOLO genera planes factibles con tus reglas.
    La rentabilidad (Valor_final - Valor_inicial - C_total) la maximizas fuera con XGBoost.
    """
    def __init__(self, base: BaseHouse, costs: CostTables, pool: PoolRules,
                 compat: Compat, budget: Optional[float] = None):
        self.base = base
        self.costs = costs
        self.pool = pool
        self.compat = compat
        self.budget = budget

        self.m = Model("remodel_generate")
        self.vars = {}
        self.total_cost = None

    def build(self):
        f = {k: canon(k, v) for k, v in self.base.features.items()}

        # ---------------------------------------
        # SETS (derivados de tablas de costos)
        # ---------------------------------------
        U  = list(self.costs.utilities.keys())
        RS = list(self.costs.roof_style.keys())
        RM = list(self.costs.roof_matl.keys())
        E1 = list(self.costs.exterior1st.keys())
        E2 = list(self.costs.exterior2nd.keys())
        MVT= list(self.costs.mas_vnr_type.keys())
        EL = list(self.costs.electrical.keys())
        H  = list(self.costs.heating.keys())
        KQ = list(self.costs.kitchen_qual.keys())

        # ===========================================================
        # (6) RESTR. REMODEL POR CATEGORÍA (mantener o subir costo)
        # ===========================================================

        # 6.1 Utilities: quedarse o cambiar a alternativa de costo >= costo base
        base_u = str(f.get("Utilities"))
        base_u_cost = float(self.costs.utilities.get(base_u, 0.0))
        self.vars["util"] = self.m.addVars(U, vtype=GRB.BINARY, name="Utilities")
        allowU = [u for u in U if self.costs.utilities.get(u, 0.0) >= base_u_cost]
        self.m.addConstr(quicksum(self.vars["util"][u] for u in allowU) == 1)  # selección única

        # 6.2 RoofStyle & RoofMatl: mantener o subir costo + compatibilidad
        base_rs = str(f.get("Roof Style"))
        base_rm = str(f.get("Roof Matl"))
        base_rs_cost = float(self.costs.roof_style.get(base_rs, 0.0))
        base_rm_cost = float(self.costs.roof_matl.get(base_rm, 0.0))
        self.vars["roof_s"] = self.m.addVars(RS, vtype=GRB.BINARY, name="RoofStyle")
        self.vars["roof_m"] = self.m.addVars(RM, vtype=GRB.BINARY, name="RoofMatl")
        allowRS = [s for s in RS if self.costs.roof_style.get(s,0.0) >= base_rs_cost]
        allowRM = [m for m in RM if self.costs.roof_matl.get(m,0.0)  >= base_rm_cost]
        self.m.addConstr(quicksum(self.vars["roof_s"][s] for s in allowRS) == 1)   # selección única
        self.m.addConstr(quicksum(self.vars["roof_m"][m] for m in allowRM) == 1)
        # compatibilidad (matriz A_{s,m}=0 ⇒ prohibido): xi,s + yi,m ≤ 1 (si incompatible)
        for (s,m) in self.compat.roof_style_by_matl_forbidden:
            if s in RS and m in RM:
                self.m.addConstr(self.vars["roof_s"][s] + self.vars["roof_m"][m] <= 1)

        # 6.3 Exterior1st/Exterior2nd + ExterQual/ExterCond
        # Si ExterQual o ExterCond <= TA ⇒ se activa Upg y se debe elegir material de costo >= base
        exq = qscore(f.get("Exter Qual", "TA"))
        exc = qscore(f.get("Exter Cond", "TA"))
        must_upg_ext = (exq <= 3) or (exc <= 3)

        base_e1 = str(f.get("Exterior 1st"))
        base_e2 = str(f.get("Exterior 2nd") or "")
        has_e2  = base_e2 != ""

        self.vars["ext1"] = self.m.addVars(E1, vtype=GRB.BINARY, name="Exterior1st")
        base_e1_cost = float(self.costs.exterior1st.get(base_e1, 0.0))
        if must_upg_ext:
            allowE1 = [e for e in E1 if self.costs.exterior1st.get(e,0.0) >= base_e1_cost + 1e-6]
        else:
            allowE1 = [base_e1]  # quedarse
        self.m.addConstr(quicksum(self.vars["ext1"][e] for e in allowE1) == 1)

        if has_e2:
            self.vars["ext2"] = self.m.addVars(E2, vtype=GRB.BINARY, name="Exterior2nd")
            base_e2_cost = float(self.costs.exterior2nd.get(base_e2, 0.0))
            if must_upg_ext:
                allowE2 = [e for e in E2 if self.costs.exterior2nd.get(e,0.0) >= base_e2_cost + 1e-6]
            else:
                allowE2 = [base_e2]
            self.m.addConstr(quicksum(self.vars["ext2"][e] for e in allowE2) == 1)



        # 6.4 MasVnrType: quedarse o subir costo
        base_mvt = str(canon("Mas Vnr Type", f.get("Mas Vnr Type","None")))
        base_mvt_cost = float(self.costs.mas_vnr_type.get(base_mvt, 0.0))
        self.vars["mvt"] = self.m.addVars(MVT, vtype=GRB.BINARY, name="MasVnrType")
        allowMVT = [t for t in MVT if self.costs.mas_vnr_type.get(t,0.0) >= base_mvt_cost]
        self.m.addConstr(quicksum(self.vars["mvt"][t] for t in allowMVT) == 1)

        # 6.5 Electrical: quedarse o subir costo (documento lo define así)
        base_el = str(f.get("Electrical"))
        base_el_cost = float(self.costs.electrical.get(base_el, 0.0))
        self.vars["el"] = self.m.addVars(EL, vtype=GRB.BINARY, name="Electrical")
        allowEL = [e for e in EL if self.costs.electrical.get(e,0.0) >= base_el_cost]
        self.m.addConstr(quicksum(self.vars["el"][e] for e in allowEL) == 1)

        # 6.6 CentralAir: si base=Yes ⇒ se mantiene; si base=No ⇒ {No,Yes} y si Yes cobra costo
        base_ca = str(canon("Central Air", f.get("Central Air","No")))
        self.vars["ca_yes"] = self.m.addVar(vtype=GRB.BINARY, name="CentralAirYes")
        if base_ca == "Yes":
            self.m.addConstr(self.vars["ca_yes"] == 1)
        # si era No, queda libre (0/1)

        # 6.7 Heating + HeatingQC: si QC ≤ TA ⇒ mantener o subir costo; si >TA ⇒ se mantiene base
        base_h  = str(f.get("Heating"))
        base_hc = float(self.costs.heating.get(base_h, 0.0))
        hqc = qscore(f.get("Heating QC","TA"))
        self.vars["heat"] = self.m.addVars(H, vtype=GRB.BINARY, name="Heating")
        if hqc <= 3:
            allowH = [h for h in H if self.costs.heating.get(h,0.0) >= base_hc + 1e-6]
        else:
            allowH = [base_h]
        self.m.addConstr(quicksum(self.vars["heat"][h] for h in allowH) == 1)

        # 6.8 KitchenQual: si ≤ TA ⇒ subir; si >TA ⇒ mantener
        base_kq = str(f.get("Kitchen Qual"))
        base_kq_cost = float(self.costs.kitchen_qual.get(base_kq, 0.0))
        self.vars["kq"] = self.m.addVars(KQ, vtype=GRB.BINARY, name="KitchenQual")
        if qscore(base_kq) <= 3:
            allowKQ = [k for k in KQ if self.costs.kitchen_qual.get(k,0.0) >= base_kq_cost + 1e-6]
        else:
            allowKQ = [base_kq]
        self.m.addConstr(quicksum(self.vars["kq"][k] for k in allowKQ) == 1)

        # 6.9 Basement finish (terminar completamente lo no terminado: todo o nada)
        bsmt_unf0 = float(f.get("Bsmt Unf SF", f.get("BsmtUnfSF", 0.0)) or 0.0)
        bsmt_fin1_0 = float(f.get("BsmtFin SF 1", f.get("BsmtFinSF1", 0.0)) or 0.0)
        bsmt_fin2_0 = float(f.get("BsmtFin SF 2", f.get("BsmtFinSF2", 0.0)) or 0.0)
        total_bsmt0 = float(f.get("Total Bsmt SF", f.get("TotalBsmtSF", bsmt_fin1_0+bsmt_fin2_0+bsmt_unf0)))
        self.vars["finish_bsmt"] = self.m.addVar(vtype=GRB.BINARY, name="FinishBSMT")
        self.vars["x_b1"] = self.m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="x_to_BsmtFin1")
        self.vars["x_b2"] = self.m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="x_to_BsmtFin2")
        # Conservación TotalBsmtSF (34)
        # (para remodel no cambiamos total_bsmt; solo reasignamos Unf → Fin1/Fin2)
        self.m.addConstr(self.vars["x_b1"] + self.vars["x_b2"] <= bsmt_unf0 * self.vars["finish_bsmt"])
        # Si termino, transfiero TODO lo no terminado (x1+x2 = BsmtUnf0) – versión fuerte:
        self.m.addConstr(self.vars["x_b1"] + self.vars["x_b2"] >= bsmt_unf0 * self.vars["finish_bsmt"])

        # ===========================
        # PISCINA (PoolArea, PoolQC)
        # ===========================
        lot_area = float(f.get("Lot Area", f.get("LotArea", 0.0)) or 0.0)
        self.vars["add_pool"]  = self.m.addVar(vtype=GRB.BINARY, name="AddPool")
        self.vars["pool_area"] = self.m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="PoolArea_new")
        self.vars["pool_qc"]   = self.m.addVar(vtype=GRB.INTEGER, lb=0, ub=4, name="PoolQC_new")
        # (3) área mínima por calidad
        self.m.addConstr(self.vars["pool_area"] >= self.pool.min_area_per_quality * self.vars["pool_qc"])
        # (4) calidad sólo si hay piscina
        self.m.addConstr(self.vars["pool_qc"] <= 4 * self.vars["add_pool"])
        # cota por tamaño de sitio
        self.m.addConstr(self.vars["pool_area"] <= self.pool.max_share * lot_area * self.vars["add_pool"])

        # ==================================================
        # (5) CONSISTENCIAS Y CAPACIDADES (ideas del doc)
        # ==================================================
        # (1) 1stFlrSF ≥ 2ndFlrSF
        first0 = float(f.get("1st Flr SF", f.get("1stFlrSF", 0.0)) or 0.0)
        second0= float(f.get("2nd Flr SF", f.get("2ndFlrSF", 0.0)) or 0.0)
        lowq0  = float(f.get("Low Qual Fin SF", f.get("LowQualFinSF", 0.0)) or 0.0)
        grliv0 = float(f.get("Gr Liv Area", f.get("GrLivArea", first0+second0+lowq0)) or 0.0)
        self.vars["first_sf"]  = self.m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="1stFlrSF_new")
        self.vars["second_sf"] = self.m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="2ndFlrSF_new")
        self.vars["lowq_sf"]   = self.m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="LowQualFinSF_new")
        self.vars["grliv"]     = self.m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="GrLivArea_new")

        # Mantener consistencia de base (sumas) y reglas:
        # (7)/(23) GrLivArea = 1st + 2nd + LowQual
        self.m.addConstr(self.vars["grliv"] == self.vars["first_sf"] + self.vars["second_sf"] + self.vars["lowq_sf"])
        # (1) 1stFlrSF ≥ 2ndFlrSF
        self.m.addConstr(self.vars["first_sf"] >= self.vars["second_sf"])
        # (2) GrLivArea ≤ LotArea
        self.m.addConstr(self.vars["grliv"] <= lot_area)

        # Para remodel: por defecto fijamos al valor actual (sin ampliación estructural).
        # Si luego activan ampliaciones, estos lb pueden liberarse o agregamos variables de delta.
        self.m.addConstr(self.vars["first_sf"]  == first0)
        self.m.addConstr(self.vars["second_sf"] == second0)
        self.m.addConstr(self.vars["lowq_sf"]   == lowq0)
        self.m.addConstr(self.vars["grliv"]     == grliv0)

        # Baños/dormitorios mínimos y consistencias (5), (4)
        full0 = int(f.get("Full Bath", f.get("FullBath", 1)) or 1)
        half0 = int(f.get("Half Bath", f.get("HalfBath", 0)) or 0)
        bed0  = int(f.get("Bedroom AbvGr", f.get("Bedroom", 1)) or 1)
        kit0  = int(f.get("Kitchen AbvGr", f.get("Kitchen", 1)) or 1)

        self.vars["full"]  = self.m.addVar(vtype=GRB.INTEGER, lb=0, name="FullBath_new")
        self.vars["half"]  = self.m.addVar(vtype=GRB.INTEGER, lb=0, name="HalfBath_new")
        self.vars["bed"]   = self.m.addVar(vtype=GRB.INTEGER, lb=0, name="Bedroom_new")
        self.vars["kit"]   = self.m.addVar(vtype=GRB.INTEGER, lb=0, name="Kitchen_new")

        # mantener valores base (hasta que metamos capa de ampliaciones de ambientes)
        self.m.addConstr(self.vars["full"] == full0)
        self.m.addConstr(self.vars["half"] == half0)
        self.m.addConstr(self.vars["bed"]  == bed0)
        self.m.addConstr(self.vars["kit"]  == kit0)

        # (4) FullBath + HalfBath ≤ Bedroom (sin bsmt) – del doc
        self.m.addConstr(self.vars["full"] + self.vars["half"] <= self.vars["bed"])
        # (5) mínimos básicos
        self.m.addConstr(self.vars["full"] >= 1)
        self.m.addConstr(self.vars["bed"]  >= 1)
        self.m.addConstr(self.vars["kit"]  >= 1)

        # ===========================
        # COSTO TOTAL (C_total)
        # ===========================
        cost_terms = []

        # Utilities (cobra cuando no eliges la base)
        for u in allowU:
            if u != base_u:
                cost_terms.append(self.costs.utilities[u] * self.vars["util"][u])

        # Roof style/material (cobra si cambia respecto a base)
        for s in allowRS:
            if s != base_rs:
                cost_terms.append(self.costs.roof_style[s] * self.vars["roof_s"][s])
        for m in allowRM:
            if m != base_rm:
                cost_terms.append(self.costs.roof_matl[m] * self.vars["roof_m"][m])

        # Exterior 1st/2nd (cobra si cambia)
        for e in allowE1:
            if e != base_e1:
                cost_terms.append(self.costs.exterior1st[e] * self.vars["ext1"][e])
        if has_e2:
            for e in allowE2:
                if e != base_e2:
                    cost_terms.append(self.costs.exterior2nd[e] * self.vars["ext2"][e])

        # MasVnrType
        for t in allowMVT:
            if t != base_mvt:
                cost_terms.append(self.costs.mas_vnr_type[t] * self.vars["mvt"][t])

        # Electrical
        for e in allowEL:
            if e != base_el:
                cost_terms.append(self.costs.electrical[e] * self.vars["el"][e])

        # Central Air (si antes era No y ahora Yes)
        if base_ca == "No":
            cost_terms.append(self.costs.central_air_install * self.vars["ca_yes"])

        # Heating
        for h in allowH:
            if h != base_h:
                cost_terms.append(self.costs.heating[h] * self.vars["heat"][h])

        # KitchenQual (si ≤TA obligamos upgrade; acá se cobra si cambió)
        for k in allowKQ:
            if k != base_kq:
                # si existe tabla de costos de remodel de cocina por calidad, úsala
                kcost = self.costs.kitchen_remodel.get(k, self.costs.kitchen_qual.get(k, 0.0))
                cost_terms.append(kcost * self.vars["kq"][k])

        # Basement – terminar lo no terminado (x_b1 + x_b2) * CBsmt
        if bsmt_unf0 > 0:
            cost_terms.append(self.costs.cost_finish_bsmt_ft2 * (self.vars["x_b1"] + self.vars["x_b2"]))

        # Piscina – costo por ft2
        cost_terms.append(self.costs.cost_pool_ft2 * self.vars["pool_area"])

        self.total_cost = quicksum(cost_terms)


        # -----------------------------------------
        # FUNCIÓN OBJETIVO = 0 (solo factibilidad)
        # -----------------------------------------
        self.m.setObjective(0.0, GRB.MINIMIZE)


    # enumeración de k planes factibles
    def solve_pool(self, k: int = 20, time_limit: int = 60):
        self.m.Params.PoolSearchMode = 2
        self.m.Params.PoolSolutions  = k
        self.m.Params.TimeLimit      = time_limit
        self.m.optimize()

        sols = []
        solcount = min(k, self.m.SolCount)
        for i in range(solcount):
            self.m.Params.SolutionNumber = i
            plan = {
                "Utilities":   [u for u,v in self.vars["util"].items()   if v.Xn > 0.5],
                "RoofStyle":   [s for s,v in self.vars["roof_s"].items() if v.Xn > 0.5],
                "RoofMatl":    [m for m,v in self.vars["roof_m"].items() if v.Xn > 0.5],
                "Exterior1st": [e for e,v in self.vars["ext1"].items()   if v.Xn > 0.5],
                "MasVnrType":  [t for t,v in self.vars["mvt"].items()    if v.Xn > 0.5],
                "Electrical":  [e for e,v in self.vars["el"].items()     if v.Xn > 0.5],
                "CentralAir":  "Yes" if self.vars["ca_yes"].Xn > 0.5 else "No",
                "Heating":     [h for h,v in self.vars["heat"].items()   if v.Xn > 0.5],
                "KitchenQual": [k for k,v in self.vars["kq"].items()     if v.Xn > 0.5],
                "AddPool":     int(self.vars["add_pool"].Xn > 0.5),
                "PoolQC":      int(self.vars["pool_qc"].Xn + 1e-6),
                "PoolArea":    float(self.vars["pool_area"].Xn),
                "Cost":        float(self.total_cost.getValue()) if self.total_cost is not None else 0.0,
            }
            if "ext2" in self.vars:
                plan["Exterior2nd"] = [e for e,v in self.vars["ext2"].items() if v.Xn > 0.5]
            else:
                plan["Exterior2nd"] = []
            sols.append(plan)
        return sols
