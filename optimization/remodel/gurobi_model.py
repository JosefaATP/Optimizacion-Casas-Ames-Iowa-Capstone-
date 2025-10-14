# optimization/remodel/gurobi_model.py
from gurobipy import Model, GRB, quicksum
from typing import Optional

# NOTA DE DISEÑO:
# - Este MILP NO optimiza costo ni "uso de presupuesto".
# - Solo genera combinaciones factibles bajo tus reglas.
# - La FO es constante (0) para permitir enumerar soluciones con Solution Pool.
# - El costo total (self.total_cost) se calcula y se usa solo como atributo para que
#   evaluate.py compute profit = pred_final - base_pred - cost.

# ---- utilidades de normalización y orden de calidades ----
QUAL_ORDER = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}  # "Average" ≈ TA (3)

def qual_score(val) -> int:
    if val is None:
        return 3
    v = str(val).strip()
    return QUAL_ORDER.get(v, 3)

ALIASES = {
    # normalización de "No aplica"/NA/etc.
    "Mas Vnr Type": {
        "No aplica": "None",
        "NA": "None",
        "": "None",
        None: "None",
    },
    "Exterior 2nd": {
        "NA": "",
        None: "",
    },
    "Central Air": {
        "Y": "Yes",
        "N": "No",
    },
}

def canon(colname: str, val):
    s = val
    if isinstance(s, str):
        s = s.strip()
    return ALIASES.get(colname, {}).get(s, s)

class RemodelMILP:
    def __init__(self, base, costs, pool_rules, compat, budget: Optional[float] = None):
        """
        base.features: dict con columnas crudas (Ames con espacios)
        costs: CostTables (con exterior1st/exterior2nd separados)
        pool_rules: PoolRules (max_share, min_area_per_quality)
        compat: Compat (set de (style, matl) prohibidos)
        budget: tope de costo (solo restricción; no es objetivo)
        """
        self.base = base
        self.costs = costs
        self.pool = pool_rules
        self.compat = compat
        self.budget = budget

        self.m = Model("remodel_generate")
        self.vars = {}
        self.total_cost = None

    def build(self):
        f = {k: (v if not isinstance(v, str) else v.strip()) for k, v in self.base.features.items()}

        # =========================
        # 1) VARIABLES DE DECISIÓN
        # =========================

        # Utilities: quedarse o mejorar (por costo proxy). No obliga cambio.
        U = list(self.costs.utilities.keys())
        base_u = str(f["Utilities"])
        base_u_cost = float(self.costs.utilities.get(base_u, 0.0))
        self.m.addConstr(
            quicksum(self.vars["util"][u] for u in self.costs.utilities
                    if self.costs.utilities[u] >= base_u_cost) == 1
)

        # Roof style / material (selección única) + compatibilidad
        RS = list(self.costs.roof_style.keys())
        RM = list(self.costs.roof_matl.keys())
        base_rs = str(f["Roof Style"])
        base_rm = str(f["Roof Matl"])

        self.vars["roof_s"] = self.m.addVars(RS, vtype=GRB.BINARY, name="roof_s")
        self.vars["roof_m"] = self.m.addVars(RM, vtype=GRB.BINARY, name="roof_m")
        self.m.addConstr(self.vars["roof_s"].sum() == 1)
        self.m.addConstr(self.vars["roof_m"].sum() == 1)

        for (s, m) in self.compat.roof_style_by_matl_forbidden:
            if s in RS and m in RM:
                self.m.addConstr(self.vars["roof_s"][s] + self.vars["roof_m"][m] <= 1)

        # Exterior 1st / 2nd (selección única)
        E1 = list(self.costs.exterior1st.keys())
        base_e1 = str(f["Exterior 1st"])
        self.vars["ext1"] = self.m.addVars(E1, vtype=GRB.BINARY, name="ext1")
        self.m.addConstr(self.vars["ext1"].sum() == 1)

        base_e2_raw = canon("Exterior 2nd", f.get("Exterior 2nd", ""))
        has_ext2 = (base_e2_raw != "")
        if has_ext2:
            E2 = list(self.costs.exterior2nd.keys())
            base_e2 = str(base_e2_raw)
            self.vars["ext2"] = self.m.addVars(E2, vtype=GRB.BINARY, name="ext2")
            self.m.addConstr(self.vars["ext2"].sum() == 1)

        # MasVnrType (selección única)
        MVT = list(self.costs.mas_vnr_type.keys())
        base_mvt = canon("Mas Vnr Type", f.get("Mas Vnr Type", "None"))
        self.vars["mvt"] = self.m.addVars(MVT, vtype=GRB.BINARY, name="mvt")
        self.m.addConstr(self.vars["mvt"].sum() == 1)

        # Electrical (selección única)
        EL = list(self.costs.electrical.keys())
        base_el = str(f["Electrical"])
        self.vars["el"] = self.m.addVars(EL, vtype=GRB.BINARY, name="el")
        self.m.addConstr(self.vars["el"].sum() == 1)

        # Central Air (binaria: si base=Yes, se fija; si No, se puede instalar)
        base_ca = canon("Central Air", f.get("Central Air", "No"))
        self.vars["ca_yes"] = self.m.addVar(vtype=GRB.BINARY, name="central_air_yes")
        if base_ca == "Yes":
            self.m.addConstr(self.vars["ca_yes"] == 1)
        # si es No, queda libre 0/1

        # Heating (selección única)
        H = list(self.costs.heating.keys())
        base_h = str(f["Heating"])
        self.vars["heat"] = self.m.addVars(H, vtype=GRB.BINARY, name="heat")
        self.m.addConstr(self.vars["heat"].sum() == 1)

        # Kitchen Qual (selección única)
        KQ = list(self.costs.kitchen_qual.keys())
        base_kq = str(f["Kitchen Qual"])
        self.vars["kq"] = self.m.addVars(KQ, vtype=GRB.BINARY, name="kq")
        self.m.addConstr(self.vars["kq"].sum() == 1)

        # Pool (add / qc / area)
        self.vars["add_pool"]  = self.m.addVar(vtype=GRB.BINARY, name="add_pool")
        self.vars["pool_qc"]   = self.m.addVar(vtype=GRB.INTEGER, lb=0, ub=4, name="pool_qc")  # 0=NA,1=Fa,2=TA,3=Gd,4=Ex
        self.vars["pool_area"] = self.m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="pool_area")

        lot_area = float(f.get("Lot Area", 0.0))
        self.m.addConstr(self.vars["pool_area"] <= self.pool.max_share * lot_area * self.vars["add_pool"])
        # si agrego piscina, qc >= 1 y area mínima por qc
        self.m.addConstr(self.vars["pool_qc"] >= 1 * self.vars["add_pool"])
        self.m.addConstr(self.vars["pool_area"] >= self.pool.min_area_per_quality * self.vars["pool_qc"])
        # si no agrego, qc y area deben ser 0
        self.m.addConstr(self.vars["pool_qc"] <= 4 * self.vars["add_pool"])

        # ===============================
        # 2) REGLAS "QUEDARSE O MEJORAR"
        # ===============================
        # La lógica general que nos diste: si la calidad/condición es <= Average (TA),
        # se DEBE cambiar a una opción "mejor". Como proxy de “mejor”, usamos costo mayor.
        eps = 1e-6

        # --- Utilities: quedarse o subir (no obliga cambio) ---
        base_u_cost = float(self.costs.utilities.get(base_u, 0.0))
        # nada que forzar; se permite quedarse o elegir alternativa de mayor costo (se paga en C)

        # --- Roof: quedarse o subir + compatibilidad ya puesta ---
        base_rs_cost = float(self.costs.roof_style.get(base_rs, 0.0))
        base_rm_cost = float(self.costs.roof_matl.get(base_rm, 0.0))
        # sin obligación de cambio (no hay "calidad" explícita acá)

        # --- Exterior 1st/2nd: si ExterQual o ExterCond <= TA, debe mejorar (costo mayor) ---
        exter_qual = qual_score(f.get("Exter Qual", "TA"))
        exter_cond = qual_score(f.get("Exter Cond", "TA"))
        must_improve_exterior = (exter_qual <= 3) or (exter_cond <= 3)

        base_e1_cost = float(self.costs.exterior1st.get(base_e1, 0.0))
        if must_improve_exterior:
            # obliga a NO elegir la base
            if base_e1 in E1:
                self.m.addConstr(self.vars["ext1"][base_e1] == 0)
            # y obliga a elegir una opción con costo >= costo base + eps
            self.m.addConstr(
                quicksum(self.vars["ext1"][e] for e in E1 if self.costs.exterior1st[e] >= base_e1_cost + eps) == 1
            )

        if has_ext2:
            base_e2_cost = float(self.costs.exterior2nd.get(base_e2, 0.0))
            if must_improve_exterior:
                if base_e2 in self.costs.exterior2nd:
                    self.m.addConstr(self.vars["ext2"][base_e2] == 0)
                self.m.addConstr(
                    quicksum(self.vars["ext2"][e] for e in self.costs.exterior2nd.keys()
                            if self.costs.exterior2nd[e] >= base_e2_cost + eps) == 1
                )

        # --- MasVnrType: quedarse o subir (no obliga cambio) ---
        base_mvt_cost = float(self.costs.mas_vnr_type.get(base_mvt, 0.0))
        # sin obligación de cambio

        # --- Electrical: quedarse o subir (no obliga cambio) ---
        base_el_cost = float(self.costs.electrical.get(base_el, 0.0))
        # sin obligación de cambio

        # --- Central Air: si base=No, se puede instalar ---
        # ya modelado con ca_yes libre si base=No

        # --- Heating: si HeatingQC <= Average, debe mejorar (costo mayor) ---
        heat_qc = qual_score(f.get("Heating QC", "TA"))
        base_h_cost = float(self.costs.heating.get(base_h, 0.0))
        if heat_qc <= 3:
            # no permite la base
            if base_h in H:
                self.m.addConstr(self.vars["heat"][base_h] == 0)
            # obliga a elegir opción con costo >= base + eps
            self.m.addConstr(
                quicksum(self.vars["heat"][h] for h in H if self.costs.heating[h] >= base_h_cost + eps) == 1
            )

        # --- KitchenQual: si <= Average, debe mejorar (costo mayor) ---
        kq_score = qual_score(base_kq)
        base_kq_cost = float(self.costs.kitchen_qual.get(base_kq, 0.0))
        if kq_score <= 3:
            if base_kq in KQ:
                self.m.addConstr(self.vars["kq"][base_kq] == 0)
            self.m.addConstr(
                quicksum(self.vars["kq"][k] for k in KQ if self.costs.kitchen_qual[k] >= base_kq_cost + eps) == 1
            )

        # ====================================
        # 3) COSTO TOTAL (solo para evaluación)
        # ====================================
        cost_terms = []

        # Utilities (solo pagamos si elegimos algo con costo >= base; si base=AllPub=0, upgrades pagan su premium)
        cost_terms.append(quicksum(self.costs.utilities[u] * self.vars["util"][u] for u in U
                                   if self.costs.utilities[u] >= base_u_cost))

        # Roof
        cost_terms.append(quicksum(self.costs.roof_style[s] * self.vars["roof_s"][s] for s in RS
                                   if self.costs.roof_style[s] >= base_rs_cost))
        cost_terms.append(quicksum(self.costs.roof_matl[m] * self.vars["roof_m"][m] for m in RM
                                   if self.costs.roof_matl[m] >= base_rm_cost))

        # Exterior
        cost_terms.append(quicksum(self.costs.exterior1st[e] * self.vars["ext1"][e] for e in E1
                                   if self.costs.exterior1st[e] >= base_e1_cost))
        if has_ext2:
            cost_terms.append(quicksum(self.costs.exterior2nd[e] * self.vars["ext2"][e] for e in self.costs.exterior2nd.keys()
                                       if self.costs.exterior2nd[e] >= (0.0 if not must_improve_exterior else base_e2_cost)))

        # Mas Vnr Type
        cost_terms.append(quicksum(self.costs.mas_vnr_type[t] * self.vars["mvt"][t] for t in MVT
                                   if self.costs.mas_vnr_type[t] >= base_mvt_cost))

        # Electrical
        cost_terms.append(quicksum(self.costs.electrical[e] * self.vars["el"][e] for e in EL
                                   if self.costs.electrical[e] >= base_el_cost))

        # Central Air (instalar si base=No)
        if base_ca == "No":
            cost_terms.append(self.costs.central_air_install * self.vars["ca_yes"])

        # Heating
        cost_terms.append(quicksum(self.costs.heating[h] * self.vars["heat"][h] for h in H
                                   if self.costs.heating[h] >= base_h_cost))

        # Kitchen
        cost_terms.append(quicksum(self.costs.kitchen_qual[k] * self.vars["kq"][k] for k in KQ
                                   if self.costs.kitchen_qual[k] >= base_kq_cost))

        # Pool (opcional: costo por ft2 y por calidad si dispones de tabla)
        # Por ahora no sumamos costo de piscina salvo que tengas unitarios.
        # Ejemplo (si quisieras):
        # cost_terms.append(self.pool.cost_per_ft2 * self.vars["pool_area"])

        total_cost_expr = quicksum(cost_terms)
        self.total_cost = total_cost_expr

        # Presupuesto (solo como restricción, NO es objetivo)
        if self.budget is not None:
            self.m.addConstr(self.total_cost <= self.budget)

        # ===================================
        # 4) FUNCIÓN OBJETIVO: SOLO FACTIBILIDAD
        # ===================================
        # FO constante = 0  -> queremos enumerar soluciones factibles con Pool.
        self.m.setObjective(0.0, GRB.MINIMIZE)

    def solve_pool(self, k: int = 20, time_limit: int = 60):
        # Buscar soluciones diversas sin optimizar nada en particular (FO constante).
        self.m.Params.PoolSearchMode = 2
        self.m.Params.PoolSolutions = k
        self.m.Params.TimeLimit = time_limit
        self.m.optimize()

        sols = []
        solcount = min(k, self.m.SolCount)
        for i in range(solcount):
            self.m.Params.SolutionNumber = i

            plan = {
                "Utilities":   [u for u, v in self.vars["util"].items() if v.Xn > 0.5],
                "RoofStyle":   [s for s, v in self.vars["roof_s"].items() if v.Xn > 0.5],
                "RoofMatl":    [m for m, v in self.vars["roof_m"].items() if v.Xn > 0.5],
                "Exterior1st": [e for e, v in self.vars["ext1"].items() if v.Xn > 0.5],
                "MasVnrType":  [t for t, v in self.vars["mvt"].items() if v.Xn > 0.5],
                "Electrical":  [e for e, v in self.vars["el"].items() if v.Xn > 0.5],
                "CentralAir":  "Yes" if self.vars["ca_yes"].Xn > 0.5 else "No",
                "Heating":     [h for h, v in self.vars["heat"].items() if v.Xn > 0.5],
                "KitchenQual": [k for k, v in self.vars["kq"].items() if v.Xn > 0.5],
                "AddPool":     int(self.vars["add_pool"].Xn > 0.5),
                "PoolQC":      int(self.vars["pool_qc"].Xn + 1e-6),
                "PoolArea":    float(self.vars["pool_area"].Xn),
                "Cost":        float(self.total_cost.getValue()) if self.total_cost is not None else 0.0,
            }
            if "ext2" in self.vars:
                plan["Exterior2nd"] = [e for e, v in self.vars["ext2"].items() if v.Xn > 0.5]
            else:
                plan["Exterior2nd"] = []
            sols.append(plan)
        return sols
