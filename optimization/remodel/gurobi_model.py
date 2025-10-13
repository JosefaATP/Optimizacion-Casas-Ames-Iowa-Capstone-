from gurobipy import Model, GRB, quicksum
from typing import Dict, Any, Tuple, List, Optional
from .model_spec import BaseHouse, CostTables, PoolRules, Compat

class RemodelMILP:
    def __init__(self, base: BaseHouse, costs: CostTables, pool: PoolRules, compat: Compat, budget: Optional[float] = None):
        self.base = base
        self.costs = costs
        self.pool = pool
        self.compat = compat
        self.budget = budget
        self.m = Model("remodel_generate")
        self.vars = {}

    def build(self):
        f = self.base.features

        # --- decision vars (ejemplos representativos del PDF) ---
        # Utilities (quedarse o subir costo)
        U = list(self.costs.utilities.keys())
        self.vars["util"] = self.m.addVars(U, vtype=GRB.BINARY, name="util")
        # seleccion unica
        self.m.addConstr(sum(self.vars["util"][u] for u in U) == 1)
        # si no hay mejora, forzamos la base si quieres bloquear cambios al no mejorar
        # o permitimos cualquier u más caro que la base (controlaremos en el costo)

        # Roof style / material
        RS = list(self.costs.roof_style.keys())
        RM = list(self.costs.roof_matl.keys())
        self.vars["roof_s"] = self.m.addVars(RS, vtype=GRB.BINARY, name="roof_s")
        self.vars["roof_m"] = self.m.addVars(RM, vtype=GRB.BINARY, name="roof_m")
        self.m.addConstr(self.vars["roof_s"].sum() == 1)
        self.m.addConstr(self.vars["roof_m"].sum() == 1)
        # compatibilidad: (s,m) prohibidos -> no pueden estar juntos
        for (s, m) in self.compat.roof_style_by_matl_forbidden:
            self.m.addConstr(self.vars["roof_s"][s] + self.vars["roof_m"][m] <= 1)

        # Exterior1st / Exterior2nd con regla "solo si ExterQual o ExterCond <= Average"
# creación de variables Exterior
        E1 = list(self.costs.exterior1st.keys())
        self.vars["ext1"] = self.m.addVars(E1, vtype=GRB.BINARY, name="ext1")
        self.m.addConstr(self.vars["ext1"].sum() == 1)

        # detectar si la casa tiene exterior 2
        base_e2_val = str(f.get("Exterior 2nd", "")).strip()
        has_ext2 = base_e2_val != "" and base_e2_val.upper() != "NA"

        if has_ext2:
            E2 = list(self.costs.exterior2nd.keys())
            self.vars["ext2"] = self.m.addVars(E2, vtype=GRB.BINARY, name="ext2")
            self.m.addConstr(self.vars["ext2"].sum() == 1)

        # MasVnrType
        MVT = list(self.costs.mas_vnr_type.keys())
        self.vars["mvt"] = self.m.addVars(MVT, vtype=GRB.BINARY, name="mvt")
        self.m.addConstr(self.vars["mvt"].sum() == 1)

        # Electrical
        EL = list(self.costs.electrical.keys())
        self.vars["el"] = self.m.addVars(EL, vtype=GRB.BINARY, name="el")
        self.m.addConstr(self.vars["el"].sum() == 1)

        # CentralAir (si base es No, permite agregar)
        base_ca = f.get("Central Air", "No")
        self.vars["ca_yes"] = self.m.addVar(vtype=GRB.BINARY, name="central_air_yes")
        if base_ca == "Yes":
            self.m.addConstr(self.vars["ca_yes"] == 1)  # se mantiene
        # si es No, se decide 0/1

        # Heating (si HeatingQC <= Average, se permite mejorar)
        H = list(self.costs.heating.keys())
        self.vars["heat"] = self.m.addVars(H, vtype=GRB.BINARY, name="heat")
        self.m.addConstr(self.vars["heat"].sum() == 1)

        # KitchenQual (si <= Average, se puede subir)
        KQ = list(self.costs.kitchen_qual.keys())
        self.vars["kq"] = self.m.addVars(KQ, vtype=GRB.BINARY, name="kq")
        self.m.addConstr(self.vars["kq"].sum() == 1)

        # Pool (add y area)
        self.vars["add_pool"] = self.m.addVar(vtype=GRB.BINARY, name="add_pool")
        self.vars["pool_qc"]  = self.m.addVar(vtype=GRB.INTEGER, lb=0, ub=4, name="pool_qc")
        self.vars["pool_area"] = self.m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="pool_area")

        lot_area = float(f["LotArea"])
        self.m.addConstr(self.vars["pool_area"] <= self.pool.max_share * lot_area * self.vars["add_pool"])
        self.m.addConstr(self.vars["pool_area"] >= self.pool.min_area_per_quality * self.vars["pool_qc"])
        self.m.addConstr(self.vars["pool_qc"] <= 4 * self.vars["add_pool"])

        # --- costos (lineales) ---
        # regla “subir o quedarse” respecto a la base se implementa via costo: si eliges algo más caro, paga diferencia.
        cost = 0.0

        # utilities
        base_u = f["Utilities"]
        cost += quicksum(self.costs.utilities[u] * self.vars["util"][u] for u in U
                         if self.costs.utilities[u] >= self.costs.utilities[base_u])

        # roof
        base_rs, base_rm = f["Roof Style"], f["Roof Matl"]
        base_e1 = f["Exterior 1st"]
        cost += quicksum(self.costs.roof_style[s]*self.vars["roof_s"][s] for s in RS
                        if self.costs.roof_style[s] >= self.costs.roof_style[base_rs])
        cost += quicksum(self.costs.roof_matl[m]*self.vars["roof_m"][m] for m in RM
                        if self.costs.roof_matl[m] >= self.costs.roof_matl[base_rm])

        # exterior 1st
        cost += quicksum(self.costs.exterior1st[e]*self.vars["ext1"][e] for e in E1
                        if self.costs.exterior1st[e] >= self.costs.exterior1st[base_e1])

        # exterior 2nd (si existe)
        if has_ext2:
            base_e2 = f["Exterior 2nd"]
            E2 = list(self.costs.exterior2nd.keys())
            cost += quicksum(self.costs.exterior2nd[e]*self.vars["ext2"][e] for e in E2
                            if self.costs.exterior2nd[e] >= self.costs.exterior2nd[base_e2])

        # mvt
        base_mvt = f["Mas Vnr Type"]
        cost += quicksum(self.costs.mas_vnr_type[t]*self.vars["mvt"][t] for t in MVT
                         if self.costs.mas_vnr_type[t] >= self.costs.mas_vnr_type[base_mvt])

        # electrical
        base_el = f["Electrical"]
        cost += quicksum(self.costs.electrical[e]*self.vars["el"][e] for e in EL
                         if self.costs.electrical[e] >= self.costs.electrical[base_el])

        # central air
        if base_ca == "No":
            cost += self.costs.central_air_install * self.vars["ca_yes"]

        # heating
        base_h = f["Heating"]
        cost += quicksum(self.costs.heating[h]*self.vars["heat"][h] for h in H
                         if self.costs.heating[h] >= self.costs.heating[base_h])

        # kitchen qual
        base_kq = f["Kitchen Qual"]
        cost += quicksum(self.costs.kitchen_qual[k]*self.vars["kq"][k] for k in KQ
                         if self.costs.kitchen_qual[k] >= self.costs.kitchen_qual[base_kq])

        # pool: podrías agregar costo variable por ft2 y por calidad si tienes tabla
        # cost += c_pool_area * self.vars["pool_area"] + c_pool_qc[self.vars["pool_qc"]]

        # guardar costo total
        self.total_cost = cost

        # consistencias basicas (del PDF) – agrega las que ya anotaron:
        # GrLivArea <= LotArea, baños vs dormitorios, etc. (dejadas como TODOs segun datos reales)

        if self.budget is not None:
            self.m.addConstr(cost <= self.budget)

        # objetivo del generador: por ahora, minimiza costo para diversificar con pool
        # (tambien puedes maximizar “calidad media” si defines pesos)
        self.m.setObjective(cost, GRB.MINIMIZE)

    def solve_pool(self, k: int = 20, time_limit: int = 60):
        self.m.Params.PoolSearchMode = 2   # buscar soluciones diversas
        self.m.Params.PoolSolutions = k
        self.m.Params.TimeLimit = time_limit
        self.m.optimize()

        sols = []
        for i in range(min(k, self.m.SolCount)):
            self.m.Params.SolutionNumber = i
            plan = {
                "Utilities": [v.VarName.split('[')[1][:-1] for v in self.vars["util"].values() if v.Xn > 0.5],
                "Roof Style": [s for s,v in self.vars["roof_s"].items() if v.Xn > 0.5],
                "Roof Matl":  [m for m,v in self.vars["roof_m"].items() if v.Xn > 0.5],
                "Exterior 1st":[e for e,v in self.vars["ext1"].items() if v.Xn > 0.5],
                "Mas Vnr Type":[t for t,v in self.vars["mvt"].items() if v.Xn > 0.5],
                "Electrical":[e for e,v in self.vars["el"].items() if v.Xn > 0.5],
                "Central Air": "Yes" if self.vars["ca_yes"].Xn > 0.5 else self.base.features.get("CentralAir","No"),
                "Heating":   [h for h,v in self.vars["heat"].items() if v.Xn > 0.5],
                "Kitchen Qual":[k for k,v in self.vars["kq"].items() if v.Xn > 0.5],
                "AddPool":   int(self.vars["add_pool"].Xn > 0.5),
                "PoolQC":    int(self.vars["pool_qc"].Xn + 1e-6),
                "Pool Area":  float(self.vars["pool_area"].Xn),
                "Cost":      float(self.total_cost.getValue())
            }
            sols.append(plan)
        return sols
