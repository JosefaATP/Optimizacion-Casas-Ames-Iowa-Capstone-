# optimization/remodel/gurobi_model.py
from typing import Optional, Dict, Any, List, Tuple
from gurobipy import Model, GRB, quicksum

# Reuse canonicalization and modelspec dataclasses from shared modules
from .canon import canon, qscore
from .model_spec import BaseHouse, CostTables, PoolRules, Compat


# ==============================
# MODELO MILP – REMODELACIÓN COMPLETO
# ==============================
class RemodelMILP:
    """
    Modelo completo de remodelación con TODAS las restricciones del PDF.
    Genera planes factibles. La rentabilidad se maximiza fuera con XGBoost.
    """
    def __init__(self, base: BaseHouse, costs: CostTables, pool: PoolRules,
                 compat: Compat, budget: Optional[float] = None, predictor=None):
        self.base = base
        self.costs = costs
        self.pool = pool
        self.compat = compat
        self.budget = budget
        # optional price predictor (XGBPricePredictor-like with predict_price(feats)->float)
        self.predictor = predictor

        self.m = Model("remodel_complete")
        self.vars = {}
        self.total_cost = None
        # list of callables that return (label, cost) when evaluated after a solution is set
        self._cost_items = []

    def build(self):
        """Construye el modelo MILP completo con todas las restricciones del PDF."""
        f = {k: canon(k, v) for k, v in self.base.features.items()}
        # Almacenar features base para calcular deltas en solve_pool
        self._base_f = f

        # ---------------------------------------
        # SETS (derivados de tablas de costos)
        # ---------------------------------------
        U   = list(self.costs.utilities.keys())
        RS  = list(self.costs.roof_style.keys())
        RM  = list(self.costs.roof_matl.keys())
        E1  = list(self.costs.exterior1st.keys())
        E2  = list(self.costs.exterior2nd.keys())
        MVT = list(self.costs.mas_vnr_type.keys())
        EL  = list(self.costs.electrical.keys())
        H   = list(self.costs.heating.keys())
        KQ  = list(self.costs.kitchen_qual.keys())
        BC  = ["Ex", "Gd", "TA", "Fa", "Po"]  # BsmtCond
        BT  = ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"]  # BsmtFinType
        FQ  = ["Ex", "Gd", "TA", "Fa", "Po", "NA"]  # FireplaceQu
        FE  = ["GdPrv", "MnPrv", "GdWo", "MnWw", "NA"]  # Fence
        PD  = ["Y", "P", "N"]  # PavedDrive
        GQ  = ["Ex", "Gd", "TA", "Fa", "Po", "NA"]  # GarageQual/Cond
        GF  = ["Fin", "RFn", "Unf", "NA"]  # GarageFinish
        PQC = ["Ex", "Gd", "TA", "Fa", "Po", "NA"]  # PoolQC

        # ===========================================================
        # (6) RESTRICCIONES DE REMODELACIÓN POR CATEGORÍA
        # ===========================================================

        # 6.1 Utilities: quedarse o subir costo
        base_u = str(f.get("Utilities"))
        base_u_cost = float(self.costs.utilities.get(base_u, 0.0))
        self.vars["util"] = self.m.addVars(U, vtype=GRB.BINARY, name="Utilities")
        allowU = [u for u in U if self.costs.utilities.get(u, 0.0) >= base_u_cost - 1e-6]
        self.m.addConstr(quicksum(self.vars["util"][u] for u in allowU) == 1, name="Util_Select")

        # 6.2 RoofStyle & RoofMatl: mantener o subir + compatibilidad
        base_rs = str(f.get("Roof Style"))
        base_rm = str(f.get("Roof Matl"))
        base_rs_cost = float(self.costs.roof_style.get(base_rs, 0.0))
        base_rm_cost = float(self.costs.roof_matl.get(base_rm, 0.0))
        
        self.vars["roof_s"] = self.m.addVars(RS, vtype=GRB.BINARY, name="RoofStyle")
        self.vars["roof_m"] = self.m.addVars(RM, vtype=GRB.BINARY, name="RoofMatl")
        
        allowRS = [s for s in RS if self.costs.roof_style.get(s,0.0) >= base_rs_cost - 1e-6]
        allowRM = [m for m in RM if self.costs.roof_matl.get(m,0.0) >= base_rm_cost - 1e-6]

        self.m.addConstr(quicksum(self.vars["roof_s"][s] for s in allowRS) == 1, name="RoofStyle_Select")
        self.m.addConstr(quicksum(self.vars["roof_m"][m] for m in allowRM) == 1, name="RoofMatl_Select")
        
        # Compatibilidad (matriz A_{s,m}=0 → prohibido)
        for (s, m) in self.compat.roof_style_by_matl_forbidden:
            if s in RS and m in RM:
                self.m.addConstr(self.vars["roof_s"][s] + self.vars["roof_m"][m] <= 1, 
                               name=f"RoofCompat_{s}_{m}")

        # 6.3 Exterior1st/2nd + ExterQual/Cond
        exq = qscore(f.get("Exter Qual", "TA"))
        exc = qscore(f.get("Exter Cond", "TA"))
        must_upg_ext = (exq <= 3) or (exc <= 3)

        base_e1 = str(f.get("Exterior 1st"))
        base_e2 = str(f.get("Exterior 2nd") or "")
        has_e2 = base_e2 != ""

        self.vars["ext1"] = self.m.addVars(E1, vtype=GRB.BINARY, name="Exterior1st")
        base_e1_cost = float(self.costs.exterior1st.get(base_e1, 0.0))
        
        if must_upg_ext:
            allowE1 = [e for e in E1 if self.costs.exterior1st.get(e,0.0) >= base_e1_cost - 1e-6]
        else:
            allowE1 = [base_e1]
        self.m.addConstr(quicksum(self.vars["ext1"][e] for e in allowE1) == 1, name="Ext1_Select")

        if has_e2:
            self.vars["ext2"] = self.m.addVars(E2, vtype=GRB.BINARY, name="Exterior2nd")
            base_e2_cost = float(self.costs.exterior2nd.get(base_e2, 0.0))
            if must_upg_ext:
                allowE2 = [e for e in E2 if self.costs.exterior2nd.get(e,0.0) >= base_e2_cost - 1e-6]
            else:
                allowE2 = [base_e2]
            self.m.addConstr(quicksum(self.vars["ext2"][e] for e in allowE2) == 1, name="Ext2_Select")

        # 6.4 MasVnrType: quedarse o subir
        base_mvt = str(canon("Mas Vnr Type", f.get("Mas Vnr Type","None")))
        base_mvt_cost = float(self.costs.mas_vnr_type.get(base_mvt, 0.0))
        self.vars["mvt"] = self.m.addVars(MVT, vtype=GRB.BINARY, name="MasVnrType")
        allowMVT = [t for t in MVT if self.costs.mas_vnr_type.get(t,0.0) >= base_mvt_cost - 1e-6]
        self.m.addConstr(quicksum(self.vars["mvt"][t] for t in allowMVT) == 1, name="MVT_Select")

        # 6.5 Electrical: quedarse o subir
        base_el = str(f.get("Electrical"))
        base_el_cost = float(self.costs.electrical.get(base_el, 0.0))
        self.vars["el"] = self.m.addVars(EL, vtype=GRB.BINARY, name="Electrical")
        allowEL = [e for e in EL if self.costs.electrical.get(e,0.0) >= base_el_cost - 1e-6]
        self.m.addConstr(quicksum(self.vars["el"][e] for e in allowEL) == 1, name="Elec_Select")

        # 6.6 CentralAir: si base=Yes → mantener; si base=No → {No,Yes}
        base_ca = str(canon("Central Air", f.get("Central Air","No")))
        self.vars["ca_yes"] = self.m.addVar(vtype=GRB.BINARY, name="CentralAirYes")
        if base_ca == "Yes":
            self.m.addConstr(self.vars["ca_yes"] == 1, name="CA_MustStayYes")

        # 6.7 Heating + HeatingQC: si QC ≤ TA → mantener o subir
        base_h = str(f.get("Heating"))
        base_hc = float(self.costs.heating.get(base_h, 0.0))
        hqc = qscore(f.get("Heating QC","TA"))
        self.vars["heat"] = self.m.addVars(H, vtype=GRB.BINARY, name="Heating")
        
        if hqc <= 3:
            allowH = [h for h in H if self.costs.heating.get(h,0.0) >= base_hc - 1e-6]
        else:
            allowH = [base_h]
        self.m.addConstr(quicksum(self.vars["heat"][h] for h in allowH) == 1, name="Heat_Select")

        # 6.8 KitchenQual: si ≤ TA → subir; si >TA → mantener
        base_kq = str(f.get("Kitchen Qual"))
        base_kq_cost = float(self.costs.kitchen_qual.get(base_kq, 0.0))
        self.vars["kq"] = self.m.addVars(KQ, vtype=GRB.BINARY, name="KitchenQual")

        if qscore(base_kq) <= 3:
            allowKQ = [k for k in KQ if self.costs.kitchen_qual.get(k,0.0) >= base_kq_cost - 1e-6]
        else:
            allowKQ = [base_kq]
        self.m.addConstr(quicksum(self.vars["kq"][k] for k in allowKQ) == 1, name="Kitchen_Select")

        # ===========================================================
        # BASEMENT (finish, cond, type1, type2)
        # ===========================================================
        bsmt_unf0 = float(f.get("Bsmt Unf SF", f.get("BsmtUnfSF", 0.0)) or 0.0)
        bsmt_fin1_0 = float(f.get("BsmtFin SF 1", f.get("BsmtFinSF1", 0.0)) or 0.0)
        bsmt_fin2_0 = float(f.get("BsmtFin SF 2", f.get("BsmtFinSF2", 0.0)) or 0.0)
        total_bsmt0 = float(f.get("Total Bsmt SF", f.get("TotalBsmtSF", bsmt_fin1_0+bsmt_fin2_0+bsmt_unf0)))
        
        # 6.9 Basement finish (terminar lo no terminado: todo o nada)
        self.vars["finish_bsmt"] = self.m.addVar(vtype=GRB.BINARY, name="FinishBSMT")
        self.vars["x_b1"] = self.m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="x_to_BsmtFin1")
        self.vars["x_b2"] = self.m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="x_to_BsmtFin2")
        
        # Transferir TODO si se termina
        self.m.addConstr(self.vars["x_b1"] + self.vars["x_b2"] <= bsmt_unf0 * self.vars["finish_bsmt"], 
                        name="BsmtFinish_Upper")
        self.m.addConstr(self.vars["x_b1"] + self.vars["x_b2"] >= bsmt_unf0 * self.vars["finish_bsmt"], 
                        name="BsmtFinish_Lower")

        # 6.10 BsmtCond: si ≤ TA → puede subir a Gd/Ex
        base_bc = str(f.get("Bsmt Cond", "TA"))
        self.vars["bsmt_cond"] = self.m.addVars(BC, vtype=GRB.BINARY, name="BsmtCond")
        
        if qscore(base_bc) <= 3:
            allowBC = [b for b in BC if self.costs.bsmt_cond.get(b,0) >= self.costs.bsmt_cond.get(base_bc,0) - 1e-6]
        else:
            allowBC = [base_bc]
        self.m.addConstr(quicksum(self.vars["bsmt_cond"][b] for b in allowBC) == 1, name="BsmtCond_Select")

        # 6.11 BsmtFinType1 y BsmtFinType2: si ≤ Rec → puede subir
        bt_score = {"GLQ":5, "ALQ":4, "BLQ":3, "Rec":2, "LwQ":1, "Unf":0, "NA":0}
        base_bt1 = str(f.get("BsmtFin Type 1", "NA"))
        base_bt2 = str(f.get("BsmtFin Type 2", "NA"))
        
        self.vars["bsmt_type1"] = self.m.addVars(BT, vtype=GRB.BINARY, name="BsmtFinType1")
        self.vars["bsmt_type2"] = self.m.addVars(BT, vtype=GRB.BINARY, name="BsmtFinType2")

        if base_bt1 != "NA" and bt_score.get(base_bt1,0) <= 2:
            allowBT1 = [t for t in BT if self.costs.bsmt_fin_type.get(t,0) >= self.costs.bsmt_fin_type.get(base_bt1,0) - 1e-6]
        else:
            allowBT1 = [base_bt1]

        if base_bt2 != "NA" and bt_score.get(base_bt2,0) <= 2:
            allowBT2 = [t for t in BT if self.costs.bsmt_fin_type.get(t,0) >= self.costs.bsmt_fin_type.get(base_bt2,0) - 1e-6]
        else:
            allowBT2 = [base_bt2]

        self.m.addConstr(quicksum(self.vars["bsmt_type1"][t] for t in allowBT1) == 1, name="BsmtType1_Select")
        self.m.addConstr(quicksum(self.vars["bsmt_type2"][t] for t in allowBT2) == 1, name="BsmtType2_Select")

        # ===========================================================
        # FIREPLACE, FENCE, PAVED DRIVE
        # ===========================================================

        # 6.12 FireplaceQu: Si TA → {TA,Gd,Ex}; Si Po → {Po,Fa}; Si NA → mantener
        # Normalizamos valores base potencialmente en español (ej. 'No aplica')
        raw_base_fq = f.get("Fireplace Qu", "NA")
        base_fq = (str(raw_base_fq) if raw_base_fq is not None else "NA").strip()
        # Map common Spanish/variant values to canonical 'NA'
        if base_fq.lower() in ("no aplica", "noaplica", "no aplica.", "no", "n/a", "na", "none"):
            base_fq = "NA"

        self.vars["fire_qu"] = self.m.addVars(FQ, vtype=GRB.BINARY, name="FireplaceQu")

        # Build allowed set but only keep keys that exist in the variable dictionary (guard KeyError)
        if base_fq == "NA":
            allowFQ = [k for k in ["NA"] if k in self.vars["fire_qu"]]
        elif base_fq == "TA":
            allowFQ = [k for k in ["TA", "Gd", "Ex"] if k in self.vars["fire_qu"]]
        elif base_fq == "Po":
            allowFQ = [k for k in ["Po", "Fa"] if k in self.vars["fire_qu"]]
        else:
            allowFQ = [base_fq] if base_fq in self.vars["fire_qu"] else [k for k in FQ if k in self.vars["fire_qu"]][:1]

        # Ensure there's at least one allowed choice (fallback to first available)
        if not allowFQ:
            allowFQ = [k for k in FQ if k in self.vars["fire_qu"]][:1]

        self.m.addConstr(quicksum(self.vars["fire_qu"][fq] for fq in allowFQ) == 1, name="FireQu_Select")

        # 6.13 Fence: Si GdWo/MnWw → puede subir; Si NA → puede construir
        raw_base_fence = f.get("Fence", "NA")
        base_fence = (str(raw_base_fence) if raw_base_fence is not None else "NA").strip()
        # Normalize common variants to canonical NA
        if base_fence.lower() in ("no aplica", "noaplica", "no aplica.", "no", "n/a", "na", "none"):
            base_fence = "NA"

        self.vars["fence"] = self.m.addVars(FE, vtype=GRB.BINARY, name="Fence")
        lot_area = float(f.get("Lot Area", f.get("LotArea", 0.0)) or 0.0)

        # Build candidate list but only keep keys that exist in variables to avoid KeyError
        if base_fence == "NA":
            cand = ["NA", "MnPrv", "GdPrv"]
        elif base_fence in ["GdWo", "MnWw"]:
            cand = [base_fence, "MnPrv", "GdPrv"]
        else:
            cand = [base_fence]

        allowFE = [k for k in cand if k in self.vars["fence"]]
        # Fallback to first available variable key if nothing matched
        if not allowFE:
            allowFE = [k for k in FE if k in self.vars["fence"]][:1]

        self.m.addConstr(quicksum(self.vars["fence"][kv] for kv in allowFE) == 1, name="Fence_Select")

        # 6.14 PavedDrive: Si P → {P,Y}; Si N → {N,P,Y}; Si Y → mantener
        base_pd = str(f.get("Paved Drive", "N"))
        self.vars["paved"] = self.m.addVars(PD, vtype=GRB.BINARY, name="PavedDrive")

        if base_pd == "Y":
            allowPD = ["Y"]
        elif base_pd == "P":
            allowPD = ["P", "Y"]
        else:
            allowPD = ["N", "P", "Y"]
        self.m.addConstr(quicksum(self.vars["paved"][p] for p in allowPD) == 1, name="Paved_Select")

        # ===========================================================
        # GARAGE (Qual, Cond, Finish)
        # ===========================================================

        # 6.15 GarageQual y GarageCond: Si ALGUNO ≤ TA → AMBOS pueden subir
        base_gq = str(f.get("Garage Qual", "TA"))
        base_gc = str(f.get("Garage Cond", "TA"))
        
        self.vars["gar_qual"] = self.m.addVars(GQ, vtype=GRB.BINARY, name="GarageQual")
        self.vars["gar_cond"] = self.m.addVars(GQ, vtype=GRB.BINARY, name="GarageCond")

        must_upg_gar = (qscore(base_gq) <= 3) or (qscore(base_gc) <= 3)

        if must_upg_gar:
            allowGQ = [g for g in GQ if self.costs.garage_qual.get(g,0) >= self.costs.garage_qual.get(base_gq,0) - 1e-6]
            allowGC = [g for g in GQ if self.costs.garage_cond.get(g,0) >= self.costs.garage_cond.get(base_gc,0) - 1e-6]
        else:
            allowGQ = [base_gq]
            allowGC = [base_gc]

        self.m.addConstr(quicksum(self.vars["gar_qual"][g] for g in allowGQ) == 1, name="GarQual_Select")
        self.m.addConstr(quicksum(self.vars["gar_cond"][g] for g in allowGC) == 1, name="GarCond_Select")

        # 6.16 GarageFinish: Si RFn o Unf → puede subir a Fin
        base_gf = str(f.get("Garage Finish", "Unf"))
        self.vars["gar_finish"] = self.m.addVars(GF, vtype=GRB.BINARY, name="GarageFinish")

        if base_gf in ["RFn", "Unf"]:
            allowGF = [base_gf, "Fin"]
        else:
            allowGF = [base_gf]
        self.m.addConstr(quicksum(self.vars["gar_finish"][g] for g in allowGF) == 1, name="GarFinish_Select")

        # ===========================================================
        # PISCINA (PoolArea, PoolQC)
        # ===========================================================
        base_pqc = str(f.get("Pool QC", "NA"))
        pool_area0 = float(f.get("Pool Area", 0.0) or 0.0)
        
        self.vars["pool_qc_cat"] = self.m.addVars(PQC, vtype=GRB.BINARY, name="PoolQC_cat")
        self.vars["pool_area"] = self.m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="PoolArea_new")

        if base_pqc == "NA":
            # Puede construir nueva piscina
            allowPQC = ["NA", "Fa", "TA", "Gd", "Ex"]
        elif qscore(base_pqc) <= 3:
            allowPQC = [p for p in PQC if self.costs.pool_qc.get(p,0) >= self.costs.pool_qc.get(base_pqc,0) - 1e-6]
        else:
            allowPQC = [base_pqc]

        self.m.addConstr(quicksum(self.vars["pool_qc_cat"][p] for p in allowPQC) == 1, name="PoolQC_Select")

        # Vincular área con calidad (min_area_per_quality por nivel)
        for p in PQC:
            if p != "NA":
                self.m.addConstr(
                    self.vars["pool_area"] >= self.pool.min_area_per_quality * qscore(p) * self.vars["pool_qc_cat"][p],
                    name=f"PoolArea_MinQual_{p}"
                )

        # Cota por tamaño de sitio (si construye piscina)
        has_pool = self.m.addVar(vtype=GRB.BINARY, name="HasPool")
        for p in PQC:
            if p != "NA":
                self.m.addConstr(has_pool >= self.vars["pool_qc_cat"][p], name=f"HasPool_From_{p}")
        self.m.addConstr(self.vars["pool_area"] <= self.pool.max_share * lot_area * has_pool, name="PoolArea_MaxShare")

        # ===========================================================
        # AMPLIACIONES (Garage, Decks, Porches) - 10%, 20%, 30%
        # ===========================================================
        garage0 = float(f.get("Garage Area", 0.0) or 0.0)
        wood0   = float(f.get("Wood Deck SF", 0.0) or 0.0)
        oporch0 = float(f.get("Open Porch SF", 0.0) or 0.0)
        eporch0 = float(f.get("Enclosed Porch", 0.0) or 0.0)
        ssn0    = float(f.get("3Ssn Porch", 0.0) or 0.0)
        screen0 = float(f.get("Screen Porch", 0.0) or 0.0)

        AMPL = ["Garage", "WoodDeck", "OpenPorch", "Enclosed", "ThreeSsn", "Screen"]
        base_areas = {
            "Garage": garage0, "WoodDeck": wood0, "OpenPorch": oporch0,
            "Enclosed": eporch0, "ThreeSsn": ssn0, "Screen": screen0
        }

        # Variables de expansión (a lo más una por componente)
        self.vars["expand_10"] = self.m.addVars(AMPL, vtype=GRB.BINARY, name="Expand10pct")
        self.vars["expand_20"] = self.m.addVars(AMPL, vtype=GRB.BINARY, name="Expand20pct")
        self.vars["expand_30"] = self.m.addVars(AMPL, vtype=GRB.BINARY, name="Expand30pct")

        for a in AMPL:
            self.m.addConstr(
                self.vars["expand_10"][a] + self.vars["expand_20"][a] + self.vars["expand_30"][a] <= 1,
                name=f"Expand_{a}_OneOnly"
            )

        # Variables de áreas finales
        self.vars["garage_sf"]   = self.m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="GarageArea_new")
        self.vars["wood_sf"]     = self.m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="WoodDeckSF_new")
        self.vars["oporch_sf"]   = self.m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="OpenPorchSF_new")
        self.vars["eporch_sf"]   = self.m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="EnclosedPorch_new")
        self.vars["ssn_sf"]      = self.m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="ThreeSsnPorch_new")
        self.vars["screen_sf"]   = self.m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="ScreenPorch_new")

        # Vincular expansión con área final
        area_vars = {
            "Garage": self.vars["garage_sf"],
            "WoodDeck": self.vars["wood_sf"],
            "OpenPorch": self.vars["oporch_sf"],
            "Enclosed": self.vars["eporch_sf"],
            "ThreeSsn": self.vars["ssn_sf"],
            "Screen": self.vars["screen_sf"]
        }

        for a in AMPL:
            base_val = base_areas[a]
            delta_10 = int(0.10 * base_val)
            delta_20 = int(0.20 * base_val)
            delta_30 = int(0.30 * base_val)
            
            self.m.addConstr(
                area_vars[a] == base_val 
                + delta_10 * self.vars["expand_10"][a]
                + delta_20 * self.vars["expand_20"][a]
                + delta_30 * self.vars["expand_30"][a],
                name=f"Area_{a}_Expansion"
            )

        # ===========================================================
        # CONSISTENCIAS Y CAPACIDADES (del PDF)
        # ===========================================================
        first0  = float(f.get("1st Flr SF", f.get("1stFlrSF", 0.0)) or 0.0)
        second0 = float(f.get("2nd Flr SF", f.get("2ndFlrSF", 0.0)) or 0.0)
        lowq0   = float(f.get("Low Qual Fin SF", f.get("LowQualFinSF", 0.0)) or 0.0)
        grliv0  = float(f.get("Gr Liv Area", f.get("GrLivArea", first0+second0+lowq0)) or 0.0)

        # En remodelación, las áreas de piso NO cambian (solo se remodelan interiores)
        # Por lo tanto usamos directamente los parámetros
        
        # (1) 1stFlrSF ≥ 2ndFlrSF (ya cumplido en base)
        self.m.addConstr(first0 >= second0, name="First_vs_Second")

        # (3) TotalBsmtSF ≤ 1stFlrSF
        self.m.addConstr(total_bsmt0 <= first0, name="Bsmt_vs_1stFlr")

        # (2) Restricción de terreno total (GrLivArea + áreas exteriores ≤ LotArea)
        used_area = (grliv0 
                    + self.vars["garage_sf"] 
                    + self.vars["wood_sf"] 
                    + self.vars["oporch_sf"] 
                    + self.vars["eporch_sf"] 
                    + self.vars["ssn_sf"] 
                    + self.vars["screen_sf"] 
                    + self.vars["pool_area"])
        
        self.m.addConstr(used_area <= lot_area, name="TotalArea_vs_Lot")

        # Baños/dormitorios (usan valores base - no cambian en remodelación)
        full0 = int(f.get("Full Bath", f.get("FullBath", 1)) or 1)
        half0 = int(f.get("Half Bath", f.get("HalfBath", 0)) or 0)
        bed0  = int(f.get("Bedroom AbvGr", f.get("Bedroom", 1)) or 1)
        kit0  = int(f.get("Kitchen AbvGr", f.get("Kitchen", 1)) or 1)

        # (4) FullBath + HalfBath ≤ Bedroom (sin bsmt)
        self.m.addConstr(full0 + half0 <= bed0, name="Baths_vs_Beds")
        
        # (5) Mínimos básicos
        self.m.addConstr(full0 >= 1, name="MinFullBath")
        self.m.addConstr(bed0 >= 1, name="MinBedroom")
        self.m.addConstr(kit0 >= 1, name="MinKitchen")

        # ===========================================================
        # COSTO TOTAL (C_total)
        # ===========================================================
        cost_terms = []
        # reset cost items (callables returning (label, cost) using .Xn on vars)
        self._cost_items = []

        def _append_simple(label: str, scalar: float, var):
            # add model expression
            cost_terms.append(scalar * var)
            # add evaluator
            def _eval(label=label, scalar=scalar, var=var):
                try:
                    val = float(var.Xn)
                except Exception:
                    val = 0.0
                cost = scalar * val
                if cost <= 1e-6:
                    return None
                return (label, float(cost))
            self._cost_items.append(_eval)

        def _append_sum(label: str, scalar: float, var_list: List):
            # sum of continuous/vars times scalar (e.g., x_b1 + x_b2)
            expr = quicksum(var_list)
            cost_terms.append(scalar * expr)
            def _eval(label=label, scalar=scalar, var_list=var_list):
                s = 0.0
                for v in var_list:
                    try:
                        s += float(v.Xn)
                    except Exception:
                        s += 0.0
                cost = scalar * s
                if cost <= 1e-6:
                    return None
                return (label, float(cost))
            self._cost_items.append(_eval)

        # --- Utilities ---
        for u in allowU:
            if u != base_u:
                _append_simple(f"Utilities → {u}", float(self.costs.utilities.get(u, 0.0)), self.vars["util"][u])

        # --- Roof Style/Material ---
        # Use first floor area as a proxy for roof/exterior area (USD/ft2 tables)
        roof_area_proxy = float(first0)
        for s in allowRS:
            if s != base_rs:
                _append_simple(f"RoofStyle → {s}", float(self.costs.roof_style.get(s, 0.0)) * roof_area_proxy, self.vars["roof_s"][s])
        for m in allowRM:
            if m != base_rm:
                _append_simple(f"RoofMatl → {m}", float(self.costs.roof_matl.get(m, 0.0)) * roof_area_proxy, self.vars["roof_m"][m])

        # --- Exterior 1st/2nd ---
        # Exterior costs are per ft2: approximate wall area with first floor area
        exterior_area_proxy = float(first0)
        for e in allowE1:
            if e != base_e1:
                _append_simple(f"Exterior1st → {e}", float(self.costs.exterior1st.get(e, 0.0)) * exterior_area_proxy, self.vars["ext1"][e])
        if has_e2:
            for e in allowE2:
                if e != base_e2:
                    _append_simple(f"Exterior2nd → {e}", float(self.costs.exterior2nd.get(e, 0.0)) * exterior_area_proxy, self.vars["ext2"][e])

        # --- MasVnrType ---
        # Masonry veneer costs are per ft2; scale by first floor area proxy
        for t in allowMVT:
            if t != base_mvt:
                _append_simple(f"MasVnrType → {t}", float(self.costs.mas_vnr_type.get(t, 0.0)) * float(first0), self.vars["mvt"][t])

        # --- Electrical ---
        for e in allowEL:
            if e != base_el:
                _append_simple(f"Electrical → {e}", float(self.costs.electrical.get(e, 0.0)), self.vars["el"][e])

        # --- Central Air (si antes era No y ahora Yes) ---
        if base_ca in ("N", "No"):
            _append_simple("CentralAir → Yes", float(self.costs.central_air_install), self.vars["ca_yes"])

        # --- Heating ---
        for h in allowH:
            if h != base_h:
                _append_simple(f"Heating → {h}", float(self.costs.heating.get(h, 0.0)), self.vars["heat"][h])

        # --- KitchenQual ---
        for k in allowKQ:
            if k != base_kq:
                # Usa costo de remodel si existe, sino usa el de calidad estándar
                kcost = float(self.costs.kitchen_remodel.get(k, self.costs.kitchen_qual.get(k, 0.0)))
                _append_simple(f"KitchenQual → {k}", kcost, self.vars["kq"][k])

        # --- Basement Finish (ÚNICA VEZ - corregido) ---
        if bsmt_unf0 > 0:
            _append_sum(f"Bsmt finish (ft²)", float(self.costs.cost_finish_bsmt_ft2), [self.vars["x_b1"], self.vars["x_b2"]])

        # --- BsmtCond ---
        for b in allowBC:
            if b != base_bc:
                _append_simple(f"BsmtCond → {b}", float(self.costs.bsmt_cond.get(b, 0.0)), self.vars["bsmt_cond"][b])

        # --- BsmtFinType1/2 ---
        for t in allowBT1:
            if t != base_bt1:
                _append_simple(f"BsmtFinType1 → {t}", float(self.costs.bsmt_fin_type.get(t, 0.0)), self.vars["bsmt_type1"][t])
        for t in allowBT2:
            if t != base_bt2:
                _append_simple(f"BsmtFinType2 → {t}", float(self.costs.bsmt_fin_type.get(t, 0.0)), self.vars["bsmt_type2"][t])

        # --- FireplaceQu ---
        for f in allowFQ:
            if f != base_fq:
                _append_simple(f"FireplaceQu → {f}", float(self.costs.fireplace_qu.get(f, 0.0)), self.vars["fire_qu"][f])

        # --- Fence (construcción desde NA o mejora de calidad) ---
        for fe in allowFE:
            if fe != base_fence:
                if base_fence == "NA" and fe in ["MnPrv", "GdPrv"]:
                    # Costo de construcción por LotArea
                    _append_simple(f"Fence build {fe}", float(self.costs.fence_build_psf * lot_area), self.vars["fence"][fe])
                else:
                    # Solo mejora de calidad
                    _append_simple(f"Fence → {fe}", float(self.costs.fence_cat.get(fe, 0.0)), self.vars["fence"][fe])

        # --- PavedDrive ---
        for p in allowPD:
            if p != base_pd:
                _append_simple(f"PavedDrive → {p}", float(self.costs.paved_drive.get(p, 0.0)), self.vars["paved"][p])

        # --- GarageQual/Cond ---
        for g in allowGQ:
            if g != base_gq:
                _append_simple(f"GarageQual → {g}", float(self.costs.garage_qual.get(g, 0.0)), self.vars["gar_qual"][g])
        for g in allowGC:
            if g != base_gc:
                _append_simple(f"GarageCond → {g}", float(self.costs.garage_cond.get(g, 0.0)), self.vars["gar_cond"][g])

        # --- GarageFinish ---
        for g in allowGF:
            if g != base_gf:
                _append_simple(f"GarageFinish → {g}", float(self.costs.garage_finish.get(g, 0.0)), self.vars["gar_finish"][g])

        # --- PoolQC (mejora de calidad) ---
        for p in allowPQC:
            if p != base_pqc and p != "NA":
                _append_simple(f"PoolQC → {p}", float(self.costs.pool_qc.get(p, 0.0)), self.vars["pool_qc_cat"][p])

        # --- Pool Area (costo por ft2 si hay cambio en área) ---
        # Solo cobra por el área NUEVA (no por la que ya existía)
        pool_area_delta = self.vars["pool_area"] - pool_area0
        if pool_area0 > 0:
            # Si ya tenía piscina, cobra solo por el incremento
            # We model this as cost_per_ft2 * (pool_area - pool_area0) so use a helper that sums vars
            _append_sum("Pool area delta (ft²)", float(self.costs.cost_pool_ft2), [pool_area_delta])
        else:
            # Si no tenía, cobra por toda el área nueva
            _append_simple("Pool area new (ft²)", float(self.costs.cost_pool_ft2), self.vars["pool_area"])

        # --- Ampliaciones (10%, 20%, 30%) ---
        for a in AMPL:
            base_val = base_areas[a]
            cost_per_ft2 = float(self.costs.expansion.get(a, 100.0))  # costo por ft2 de ampliación
            
            delta_10 = int(0.10 * base_val)
            delta_20 = int(0.20 * base_val)
            delta_30 = int(0.30 * base_val)
            
            if delta_10 > 0:
                _append_simple(f"{a} expansion 10% (+{delta_10} ft2)", cost_per_ft2 * delta_10, self.vars["expand_10"][a])
            if delta_20 > 0:
                _append_simple(f"{a} expansion 20% (+{delta_20} ft2)", cost_per_ft2 * delta_20, self.vars["expand_20"][a])
            if delta_30 > 0:
                _append_simple(f"{a} expansion 30% (+{delta_30} ft2)", cost_per_ft2 * delta_30, self.vars["expand_30"][a])

        # --- COSTO TOTAL ---
        self.total_cost = quicksum(cost_terms)

        # -----------------------------------------
        # RESTRICCIÓN DE PRESUPUESTO
        # -----------------------------------------
        if self.budget is not None and self.budget > 0:
            self.m.addConstr(self.total_cost <= self.budget, name="Budget_Constraint")

        # -----------------------------------------
        # FUNCIÓN OBJETIVO: si tenemos predictor, maximizamos
        # (predicted_delta) - cost; en otro caso, mantenemos objetivo 0
        # -----------------------------------------
        if self.predictor is not None:
            # compute base prediction
            try:
                base_pred = float(self.predictor.predict_price(self.base.features))
            except Exception:
                base_pred = 0.0

            value_terms = []

            # helper to compute pred delta for a categorical choice
            def pred_delta_for(key_in_feats, val):
                feats = dict(self.base.features)
                feats[key_in_feats] = val
                try:
                    p = float(self.predictor.predict_price(feats))
                    return p - base_pred
                except Exception:
                    return 0.0

            # Utilities
            for u in allowU:
                if u != base_u:
                    coeff = pred_delta_for('Utilities', u)
                    if abs(coeff) > 1e-9:
                        value_terms.append(coeff * self.vars['util'][u])

            # Roof style/material
            for s in allowRS:
                if s != base_rs:
                    coeff = pred_delta_for('Roof Style', s)
                    if abs(coeff) > 1e-9:
                        value_terms.append(coeff * self.vars['roof_s'][s])
            for m in allowRM:
                if m != base_rm:
                    coeff = pred_delta_for('Roof Matl', m)
                    if abs(coeff) > 1e-9:
                        value_terms.append(coeff * self.vars['roof_m'][m])

            # Exterior
            for e in allowE1:
                if e != base_e1:
                    coeff = pred_delta_for('Exterior 1st', e)
                    if abs(coeff) > 1e-9:
                        value_terms.append(coeff * self.vars['ext1'][e])
            if has_e2:
                for e in allowE2:
                    if e != base_e2:
                        coeff = pred_delta_for('Exterior 2nd', e)
                        if abs(coeff) > 1e-9:
                            value_terms.append(coeff * self.vars['ext2'][e])

            # MasVnrType
            for t in allowMVT:
                if t != base_mvt:
                    coeff = pred_delta_for('Mas Vnr Type', t)
                    if abs(coeff) > 1e-9:
                        value_terms.append(coeff * self.vars['mvt'][t])

            # Electrical
            for e in allowEL:
                if e != base_el:
                    coeff = pred_delta_for('Electrical', e)
                    if abs(coeff) > 1e-9:
                        value_terms.append(coeff * self.vars['el'][e])

            # Heating
            for h in allowH:
                if h != base_h:
                    coeff = pred_delta_for('Heating', h)
                    if abs(coeff) > 1e-9:
                        value_terms.append(coeff * self.vars['heat'][h])

            # KitchenQual
            for k in allowKQ:
                if k != base_kq:
                    coeff = pred_delta_for('Kitchen Qual', k)
                    if abs(coeff) > 1e-9:
                        value_terms.append(coeff * self.vars['kq'][k])

            # BsmtCond
            for b in allowBC:
                if b != base_bc:
                    coeff = pred_delta_for('Bsmt Cond', b)
                    if abs(coeff) > 1e-9:
                        value_terms.append(coeff * self.vars['bsmt_cond'][b])

            # other categorical variables follow same pattern (fireplace, fence, paved, garage, poolqc)
            for fkey, varname, featname in [
                (allowFQ, 'fire_qu', 'Fireplace Qu'),
                (allowFE, 'fence', 'Fence'),
                (allowPD, 'paved', 'Paved Drive'),
            ]:
                for opt in fkey:
                    if opt:
                        v = self.vars.get(varname)
                        if v and opt in v:
                            coeff = pred_delta_for(featname, opt)
                            if abs(coeff) > 1e-9:
                                value_terms.append(coeff * v[opt])

            # PoolQC
            for p in allowPQC:
                if p != base_pqc and p != 'NA':
                    coeff = pred_delta_for('Pool QC', p)
                    if abs(coeff) > 1e-9:
                        value_terms.append(coeff * self.vars['pool_qc_cat'][p])

            # Build final objective: maximize predicted delta - total_cost
            obj_expr = quicksum(value_terms) - self.total_cost
            self.m.setObjective(obj_expr, GRB.MAXIMIZE)
            # keep the linearized objective expression for later evaluation/diagnostics
            self._linearized_obj_expr = obj_expr
        else:
            # default: feasibility search
            self.m.setObjective(0.0, GRB.MINIMIZE)


    def solve_pool(self, k: int = 20, time_limit: int = 60):
        """
        Enumera hasta k planes factibles usando PoolSearchMode.
        Retorna lista de diccionarios con las decisiones y costos.
        """
        self.m.Params.PoolSearchMode = 2
        self.m.Params.PoolSolutions = k
        self.m.Params.TimeLimit = time_limit
        self.m.optimize()

        sols = []
        seen = set()
        solcount = min(k, self.m.SolCount)

        for i in range(solcount):
            self.m.Params.SolutionNumber = i
            plan = self._extract_solution()
            # deduplicate by decision fingerprint (tuple of sorted key/values)
            key = tuple(sorted((k, str(v)) for k, v in plan.items() if k not in ("Cost_breakdown", "Cost")))
            if key in seen:
                continue
            seen.add(key)
            sols.append(plan)

        return sols


    def _extract_solution(self) -> Dict[str, Any]:
        """Extrae una solución del pool y calcula desglose de costos."""
        
        # Helper para extraer valores binarios
        def get_selected(var_dict):
            return [k for k, v in var_dict.items() if v.Xn > 0.5]
        
        def get_first(var_dict):
            selected = get_selected(var_dict)
            return selected[0] if selected else None

        # Extraer decisiones
        plan = {
            # Categorías principales
            "Utilities": get_first(self.vars["util"]),
            "RoofStyle": get_first(self.vars["roof_s"]),
            "RoofMatl": get_first(self.vars["roof_m"]),
            "Exterior1st": get_first(self.vars["ext1"]),
            "MasVnrType": get_first(self.vars["mvt"]),
            "Electrical": get_first(self.vars["el"]),
            "CentralAir": "Yes" if self.vars["ca_yes"].Xn > 0.5 else "No",
            "Heating": get_first(self.vars["heat"]),
            "KitchenQual": get_first(self.vars["kq"]),
            
            # Basement
            "BsmtCond": get_first(self.vars["bsmt_cond"]),
            "BsmtFinType1": get_first(self.vars["bsmt_type1"]),
            "BsmtFinType2": get_first(self.vars["bsmt_type2"]),
            "FinishBSMT": int(self.vars["finish_bsmt"].Xn > 0.5),
            "x_to_BsmtFin1": float(self.vars["x_b1"].Xn),
            "x_to_BsmtFin2": float(self.vars["x_b2"].Xn),
            
            # Extras
            "FireplaceQu": get_first(self.vars["fire_qu"]),
            "Fence": get_first(self.vars["fence"]),
            "PavedDrive": get_first(self.vars["paved"]),
            
            # Garage
            "GarageQual": get_first(self.vars["gar_qual"]),
            "GarageCond": get_first(self.vars["gar_cond"]),
            "GarageFinish": get_first(self.vars["gar_finish"]),
            
            # Piscina
            "PoolQC": get_first(self.vars["pool_qc_cat"]),
            "PoolArea": float(self.vars["pool_area"].Xn),
            
            # Ampliaciones
            "GarageArea": float(self.vars["garage_sf"].Xn),
            "WoodDeckSF": float(self.vars["wood_sf"].Xn),
            "OpenPorchSF": float(self.vars["oporch_sf"].Xn),
            "EnclosedPorch": float(self.vars["eporch_sf"].Xn),
            "3SsnPorch": float(self.vars["ssn_sf"].Xn),
            "ScreenPorch": float(self.vars["screen_sf"].Xn),
        }

        # Exterior2nd (si existe)
        if "ext2" in self.vars:
            plan["Exterior2nd"] = get_first(self.vars["ext2"])

        # Desglose de expansiones (para debug)
        AMPL = ["Garage", "WoodDeck", "OpenPorch", "Enclosed", "ThreeSsn", "Screen"]
        plan["Expansions"] = {}
        for a in AMPL:
            if self.vars["expand_10"][a].Xn > 0.5:
                plan["Expansions"][a] = "10%"
            elif self.vars["expand_20"][a].Xn > 0.5:
                plan["Expansions"][a] = "20%"
            elif self.vars["expand_30"][a].Xn > 0.5:
                plan["Expansions"][a] = "30%"

        # Add explicit expansion deltas (ft2) for diagnostics
        plan["_expansion_deltas"] = {}
        for a in AMPL:
            base_key = {
                "Garage": "Garage Area", "WoodDeck": "Wood Deck SF", "OpenPorch": "Open Porch SF",
                "Enclosed": "Enclosed Porch", "ThreeSsn": "3Ssn Porch", "Screen": "Screen Porch"
            }[a]
            base_area = float(self._base_f.get(base_key, 0.0) or 0.0)
            plan_key = {
                "Garage": "GarageArea", "WoodDeck": "WoodDeckSF", "OpenPorch": "OpenPorchSF",
                "Enclosed": "EnclosedPorch", "ThreeSsn": "3SsnPorch", "Screen": "ScreenPorch"
            }[a]
            new_area = float(plan.get(plan_key, base_area))
            plan["_expansion_deltas"][a] = new_area - base_area

        # --- CALCULAR DESGLOSE DE COSTOS usando evaluadores registrados ---
        breakdown = []
        for ev in self._cost_items:
            try:
                item = ev()
            except Exception:
                item = None
            if item:
                label, cost = item
                # skip trivial/NA labels or zero-cost
                if cost > 1e-6:
                    breakdown.append((label, cost))

        # sort breakdown by descending cost for readability
        breakdown.sort(key=lambda x: -x[1])
        plan["Cost_breakdown"] = breakdown
        plan["Cost"] = sum(v for _, v in breakdown)

        # Debug: validar presupuesto
        if self.budget is not None and self.budget > 0:
            if plan["Cost"] > float(self.budget) + 1e-6:
                plan["_violates_budget"] = True
                plan["_budget_value"] = float(self.budget)

        return plan


    def _compute_cost_breakdown(self, plan: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Calcula el desglose detallado de costos basado en las decisiones."""
        breakdown = []
        bf = self._base_f

        # Helper para comparar
        def changed(key, plan_key=None):
            if plan_key is None:
                plan_key = key
            base_val = str(bf.get(key, ""))
            new_val = plan.get(plan_key)
            return new_val and str(new_val) != base_val

        # Utilities
        if changed("Utilities"):
            new_val = plan["Utilities"]
            cost = float(self.costs.utilities.get(new_val, 0.0))
            breakdown.append((f"Utilities → {new_val}", cost))

        # Roof Style
        if changed("Roof Style", "RoofStyle"):
            new_val = plan["RoofStyle"]
            # scale per-ft2 cost by first floor area proxy (same as model)
            first0 = float(bf.get("1st Flr SF", bf.get("1stFlrSF", 0.0)) or 0.0)
            cost = float(self.costs.roof_style.get(new_val, 0.0)) * first0
            breakdown.append((f"RoofStyle → {new_val}", cost))

        # Roof Material
        if changed("Roof Matl", "RoofMatl"):
            new_val = plan["RoofMatl"]
            first0 = float(bf.get("1st Flr SF", bf.get("1stFlrSF", 0.0)) or 0.0)
            cost = float(self.costs.roof_matl.get(new_val, 0.0)) * first0
            breakdown.append((f"RoofMatl → {new_val}", cost))

        # Exterior 1st
        if changed("Exterior 1st", "Exterior1st"):
            new_val = plan["Exterior1st"]
            first0 = float(bf.get("1st Flr SF", bf.get("1stFlrSF", 0.0)) or 0.0)
            cost = float(self.costs.exterior1st.get(new_val, 0.0)) * first0
            breakdown.append((f"Exterior1st → {new_val}", cost))

        # Exterior 2nd
        if "Exterior2nd" in plan and changed("Exterior 2nd", "Exterior2nd"):
            new_val = plan["Exterior2nd"]
            first0 = float(bf.get("1st Flr SF", bf.get("1stFlrSF", 0.0)) or 0.0)
            cost = float(self.costs.exterior2nd.get(new_val, 0.0)) * first0
            breakdown.append((f"Exterior2nd → {new_val}", cost))

        # MasVnrType
        if changed("Mas Vnr Type", "MasVnrType"):
            new_val = plan["MasVnrType"]
            first0 = float(bf.get("1st Flr SF", bf.get("1stFlrSF", 0.0)) or 0.0)
            cost = float(self.costs.mas_vnr_type.get(new_val, 0.0)) * first0
            breakdown.append((f"MasVnrType → {new_val}", cost))

        # Electrical
        if changed("Electrical"):
            new_val = plan["Electrical"]
            cost = float(self.costs.electrical.get(new_val, 0.0))
            breakdown.append((f"Electrical → {new_val}", cost))

        # Central Air
        base_ca = str(canon("Central Air", bf.get("Central Air", "No")))
        if base_ca in ("N", "No") and plan["CentralAir"] == "Yes":
            cost = float(self.costs.central_air_install)
            breakdown.append(("CentralAir → Yes", cost))

        # Heating
        if changed("Heating"):
            new_val = plan["Heating"]
            cost = float(self.costs.heating.get(new_val, 0.0))
            breakdown.append((f"Heating → {new_val}", cost))

        # KitchenQual
        if changed("Kitchen Qual", "KitchenQual"):
            new_val = plan["KitchenQual"]
            cost = float(self.costs.kitchen_remodel.get(new_val, 
                        self.costs.kitchen_qual.get(new_val, 0.0)))
            breakdown.append((f"KitchenQual → {new_val}", cost))

        # Basement finish
        xb1 = float(plan.get("x_to_BsmtFin1", 0.0))
        xb2 = float(plan.get("x_to_BsmtFin2", 0.0))
        if (xb1 + xb2) > 1e-6:
            cost = float(self.costs.cost_finish_bsmt_ft2) * (xb1 + xb2)
            breakdown.append((f"Bsmt finish ({xb1+xb2:.0f} ft²)", cost))

        # BsmtCond
        if changed("Bsmt Cond", "BsmtCond"):
            new_val = plan["BsmtCond"]
            cost = float(self.costs.bsmt_cond.get(new_val, 0.0))
            breakdown.append((f"BsmtCond → {new_val}", cost))

        # BsmtFinType1/2
        if changed("BsmtFin Type 1", "BsmtFinType1"):
            new_val = plan["BsmtFinType1"]
            cost = float(self.costs.bsmt_fin_type.get(new_val, 0.0))
            breakdown.append((f"BsmtFinType1 → {new_val}", cost))
        if changed("BsmtFin Type 2", "BsmtFinType2"):
            new_val = plan["BsmtFinType2"]
            cost = float(self.costs.bsmt_fin_type.get(new_val, 0.0))
            breakdown.append((f"BsmtFinType2 → {new_val}", cost))

        # FireplaceQu
        if changed("Fireplace Qu", "FireplaceQu"):
            new_val = plan["FireplaceQu"]
            cost = float(self.costs.fireplace_qu.get(new_val, 0.0))
            breakdown.append((f"FireplaceQu → {new_val}", cost))

        # Fence
        base_fence = str(bf.get("Fence", "NA"))
        new_fence = plan.get("Fence")
        if new_fence and new_fence != base_fence:
            if base_fence == "NA" and new_fence in ["MnPrv", "GdPrv"]:
                lot_area = float(bf.get("Lot Area", bf.get("LotArea", 0.0)) or 0.0)
                cost = float(self.costs.fence_build_psf * lot_area)
                breakdown.append((f"Fence build {new_fence}", cost))
            else:
                cost = float(self.costs.fence_cat.get(new_fence, 0.0))
                breakdown.append((f"Fence → {new_fence}", cost))

        # PavedDrive
        if changed("Paved Drive", "PavedDrive"):
            new_val = plan["PavedDrive"]
            cost = float(self.costs.paved_drive.get(new_val, 0.0))
            breakdown.append((f"PavedDrive → {new_val}", cost))

        # GarageQual/Cond
        if changed("Garage Qual", "GarageQual"):
            new_val = plan["GarageQual"]
            cost = float(self.costs.garage_qual.get(new_val, 0.0))
            breakdown.append((f"GarageQual → {new_val}", cost))
        if changed("Garage Cond", "GarageCond"):
            new_val = plan["GarageCond"]
            cost = float(self.costs.garage_cond.get(new_val, 0.0))
            breakdown.append((f"GarageCond → {new_val}", cost))

        # GarageFinish
        if changed("Garage Finish", "GarageFinish"):
            new_val = plan["GarageFinish"]
            cost = float(self.costs.garage_finish.get(new_val, 0.0))
            breakdown.append((f"GarageFinish → {new_val}", cost))

        # PoolQC
        base_pqc = str(bf.get("Pool QC", "NA"))
        new_pqc = plan.get("PoolQC")
        if new_pqc and new_pqc != base_pqc and new_pqc != "NA":
            cost = float(self.costs.pool_qc.get(new_pqc, 0.0))
            breakdown.append((f"PoolQC → {new_pqc}", cost))

        # Pool Area (solo delta si ya existía)
        pool_area0 = float(bf.get("Pool Area", 0.0) or 0.0)
        pool_area_new = float(plan.get("PoolArea", 0.0))
        if pool_area_new > pool_area0 + 1e-6:
            delta = pool_area_new - pool_area0
            cost = float(self.costs.cost_pool_ft2) * delta
            breakdown.append((f"Pool area +{delta:.0f} ft²", cost))

        # Ampliaciones
        AMPL = ["Garage", "WoodDeck", "OpenPorch", "Enclosed", "ThreeSsn", "Screen"]
        area_map = {
            "Garage": ("Garage Area", "GarageArea"),
            "WoodDeck": ("Wood Deck SF", "WoodDeckSF"),
            "OpenPorch": ("Open Porch SF", "OpenPorchSF"),
            "Enclosed": ("Enclosed Porch", "EnclosedPorch"),
            "ThreeSsn": ("3Ssn Porch", "3SsnPorch"),
            "Screen": ("Screen Porch", "ScreenPorch")
        }

        for a in AMPL:
            base_key, plan_key = area_map[a]
            base_area = float(bf.get(base_key, 0.0) or 0.0)
            new_area = float(plan.get(plan_key, base_area))
            
            if new_area > base_area + 1e-6:
                delta = new_area - base_area
                cost_per_ft2 = float(self.costs.expansion.get(a, 100.0))
                cost = cost_per_ft2 * delta
                pct = plan["Expansions"].get(a, "?")
                breakdown.append((f"{a} expansion {pct} (+{delta:.0f} ft²)", cost))

        return breakdown