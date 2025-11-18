# === FILE: optimization/construction/summary_and_costs_hooks.py ===

from .xgb_predictor import ROOF_STYLE_TO_ORD, ROOF_MATL_TO_ORD

"""Lightweight hooks you can import from run_opt.py or gurobi_model.py
   - build_cost_expr(m, x, ct, params)
   - summarize_solution(m, x, base_row, ct, params)
"""
from typing import Dict
import gurobipy as gp

def build_cost_expr(m: gp.Model, x: Dict[str, gp.Var], ct, params: Dict) -> gp.LinExpr:
    cost = gp.LinExpr(0.0)
    # estructura de costos, respeta tu costs.py (rellena valores allí)
    # 1) área construida
    c_build = float(getattr(ct, "construction_cost", 0.0))
    cost += c_build * (x["1st Flr SF"] + x["2nd Flr SF"] + x["BsmtFin SF 1"] + x["BsmtFin SF 2"])  
    # 2) techo por material y área real
    for mat, unit in getattr(ct, "roof_cost_per_sf", {}).items():
        v = m._x.get(f"roof_matl_is_{mat}")
        if v is not None and unit:
            cost += unit * x["ActualRoofArea"] * v
    # 3) garage (área)
    cost += float(getattr(ct, "garage_cost_per_sf", 0.0)) * x["Garage Area"]
    # 4) porches / pool
    for nm, unit in {
        "Total Porch SF": getattr(ct, "porch_cost_per_sf", 0.0),
        "Pool Area": getattr(ct, "pool_cost_per_sf", 0.0),
    }.items():
        if unit:
            cost += unit * x[nm]
    return cost


def summarize_solution(m, x=None, base_row=None, ct=None, params=None, top_cost_terms=12):
    try:
        print("\n" + "="*50)
        print("               HOUSE SUMMARY")
        print("="*50)
        rv = getattr(m, "_report_vars", {})
        def val(name):
            v = rv.get(name)
            return None if v is None else (v.X if hasattr(v, "X") else float(v))
        def val_by_name(col):
            for nm in (f"x_{col}", col):
                v = m.getVarByName(nm)
                if v is not None:
                    return float(getattr(v, "X", v))
            return None
        def fmt(x):
            if x is None:
                return "-"
            try:
                return f"{float(x):,.1f}" if abs(float(x)) >= 10 else f"{float(x):,.2f}"
            except Exception:
                return str(x)

        # seleccion de categorias
        def chosen(prefix, opts):
            for o in opts:
                v = m.getVarByName(f"{prefix}__{o}")
                if v is not None and getattr(v, 'X', 0.0) > 0.5:
                    return o
            return None

        rows = []
        # pisos y areas bases
        rows.append(("Floor1", fmt(val("Floor1")), "-"))
        rows.append(("Floor2", fmt(val("Floor2")), "-"))
        rows.append(("1st Flr SF", fmt(val("1st Flr SF")), "-"))
        rows.append(("2nd Flr SF", fmt(val("2nd Flr SF")), "-"))
        rows.append(("Total Bsmt SF", fmt(val("Total Bsmt SF")), "-"))
        rows.append(("Garage Area", fmt(val("Garage Area")), "-"))
        rows.append(("Gr Liv Area", fmt(val("Gr Liv Area")), "-"))

        # habitaciones / baños / cocina
        rows.append(("Bedrooms", fmt(val("Bedrooms")), "-"))
        rows.append(("FullBath", fmt(val("FullBath")), "-"))
        rows.append(("HalfBath", fmt(val("HalfBath")), "-"))
        rows.append(("Kitchen", fmt(val("Kitchen")), "-"))
        rows.append(("Fireplaces", fmt(val("Fireplaces")), "-"))

        # techo elegido
        roof = []
        for s in ['Flat','Gable','Gambrel','Hip','Mansard','Shed']:
            for mm in ['ClyTile','CompShg','Membran','Metal','Roll','Tar&Grv','WdShake','WdShngl']:
                y = m.getVarByName(f"Y__{s}__{mm}")
                if y and y.X > 0.5:
                    roof.append(f"{s}-{mm}")
        rows.append(("Roof", "-", roof[0] if roof else "-"))

        # categorias clave
        heating = chosen('Heating', ['Floor','GasA','GasW','Grav','OthW','Wall'])
        electrical = chosen('Electrical', ['SBrkr','FuseA','FuseF','FuseP','Mix'])
        paved = chosen('PavedDrive', ['Y','P','N'])
        house_style = chosen('HouseStyle', ['1Story','1.5Fin','1.5Unf','2Story','2.5Fin','2.5Unf','SFoyer','SLvl'])
        ext1 = chosen('Exterior1st', ['VinylSd','MetalSd','Wd Sdng','HdBoard','Stucco','Plywood','CemntBd','BrkFace','BrkComm','WdShngl','AsbShng','Stone','ImStucc','AsphShn','CBlock'])
        ext2 = chosen('Exterior2nd', ['VinylSd','MetalSd','Wd Sdng','HdBoard','Stucco','Plywood','CemntBd','BrkFace','BrkComm','WdShngl','AsbShng','Stone','ImStucc','AsphShn','CBlock'])
        fnd = chosen('Foundation', ['BrkTil','CBlock','PConc','Slab','Stone','Wood'])
        try:
            area_fnd = m.getVarByName(f"FA__{fnd}")
            area_fnd = float(getattr(area_fnd, 'X', 0.0)) if area_fnd is not None else 0.0
        except Exception:
            area_fnd = 0.0
        if heating: rows.append(("Heating", "-", heating))
        if electrical: rows.append(("Electrical", "-", electrical))
        if paved: rows.append(("PavedDrive", "-", paved))
        if house_style: rows.append(("HouseStyle", "-", house_style))
        if ext1: rows.append(("Exterior1st", "-", ext1))
        if ext2: rows.append(("Exterior2nd", "-", ext2))
        if fnd: rows.append(("Foundation", fmt(area_fnd)+" ft2", fnd))

        # calidades con costo (si aplica)
        qual_map = { -1:"NA", 0:"Po", 1:"Fa", 2:"TA", 3:"Gd", 4:"Ex" }
        if hasattr(ct, "heating_qc_costs"):
            hq = val_by_name("Heating QC")
            if hq is not None:
                rows.append(("Heating QC", "-", qual_map.get(int(round(hq)), hq)))
        if hasattr(ct, "fireplace_costs"):
            fq = val_by_name("Fireplace Qu")
            if fq is not None:
                fp_cnt = val("Fireplaces") or 0
                rows.append(("Fireplace Qu", fmt(fp_cnt), qual_map.get(int(round(fq)), fq)))
        if hasattr(ct, "poolqc_costs"):
            pq = val_by_name("Pool QC")
            if pq is not None:
                pool_area = val("Pool Area") or 0
                rows.append(("Pool QC", f"{pool_area:.0f} ft2", qual_map.get(int(round(pq)), pq)))
        if hasattr(ct, "bsmt_cond_upgrade_costs"):
            bc = val_by_name("Bsmt Cond")
            if bc is not None:
                tbsmt = val("Total Bsmt SF") or 0
                rows.append(("Bsmt Cond", f"{tbsmt:.0f} ft2", qual_map.get(int(round(bc)), bc)))

        # áres por ambiente (totales)
        for n1, n2, ntot in [
            ("AreaKitchen1","AreaKitchen2","AreaKitchen"),
            ("AreaFullBath1","AreaFullBath2","AreaFullBath"),
            ("AreaHalfBath1","AreaHalfBath2","AreaHalfBath"),
            ("AreaBedroom1","AreaBedroom2","AreaBedroom"),
            ("AreaOther1","AreaOther2",None),
        ]:
            v1 = m.getVarByName(n1); v2 = m.getVarByName(n2); vt = m.getVarByName(ntot) if ntot else None
            a1 = float(getattr(v1,'X',0.0)) if v1 is not None else 0.0
            a2 = float(getattr(v2,'X',0.0)) if v2 is not None else 0.0
            at = float(getattr(vt,'X',a1+a2)) if vt is not None else a1+a2
            label = (ntot or n1.replace('1','')).replace('Area','')
            rows.append((f"{label} (ft2)", fmt(at), "-"))

        # imprime tabla
        print(f"{'atributo':<16s} {'cantidad':>12s} {'calidad':>12s}")
        for attr, qty, qual in rows:
            print(f"{attr:<16s} {str(qty):>12s} {str(qual):>12s}")

        # economia
        yp = getattr(m, "_y_price_var", None)
        yl = getattr(m, "_y_log_var", None)
        if yp is not None:
            print(f"[econ] precio_predicho = {yp.X:,.0f}")
        if yl is not None:
            print(f"[econ] log_precio = {yl.X:.4f}")

        # detalle de áreas por piso y por ambiente (si existen)
        try:
            names = [
                ("AreaKitchen1","AreaKitchen2","AreaKitchen"),
                ("AreaFullBath1","AreaFullBath2","AreaFullBath"),
                ("AreaHalfBath1","AreaHalfBath2","AreaHalfBath"),
                ("AreaBedroom1","AreaBedroom2","AreaBedroom"),
                ("AreaOther1","AreaOther2",None),
            ]
            print("\n-- áreas por ambiente (ft2) --")
            print(f"{'ambiente':12s} {'1st':>8s} {'2nd':>8s} {'total':>10s}")
            for n1,n2,ntot in names:
                v1 = m.getVarByName(n1); v2 = m.getVarByName(n2); vt = m.getVarByName(ntot) if ntot else None
                a1 = float(getattr(v1,'X',0.0)) if v1 is not None else 0.0
                a2 = float(getattr(v2,'X',0.0)) if v2 is not None else 0.0
                at = float(getattr(vt,'X',a1+a2)) if vt is not None else a1+a2
                label = (ntot or n1.replace('1','')).replace('Area','')
                print(f"{label:<12s} {a1:8.1f} {a2:8.1f} {at:10.1f}")
        except Exception:
            pass

        # costos top
        if hasattr(m, "_cost_terms"):
            terms = []
            for label, coef, var in m._cost_terms:
                v = var.X if hasattr(var, "X") else float(var)
                amt = coef * v
                if abs(amt) > 1e-6:
                    terms.append((amt, label, v, coef))
            terms.sort(reverse=True)
            print("\n-- top costos --")
            for amt, label, v, coef in terms[:top_cost_terms]:
                print(f"{label:28s} = {amt:12,.0f}  (coef {coef:,.1f} * {v:,.1f})")
        else:
            print("[COST-BREAKDOWN] no _cost_terms en el modelo")

        # auditoria XGB
        xin = getattr(m, "_X_input", None)
        if xin:
            print("[AUDIT] hay _X_input disponible para comparar con el XGB fuera del MIP")
        else:
            print("[AUDIT] no hay _X_input para auditar")

        
        xin = getattr(m, "_X_input", None)
        if xin:
            order = xin["order"]; xdict = xin["x"]
            rows = []
            for i, col in enumerate(order):
                v = xdict.get(col)
                val = float(v.X) if v is not None else float("nan")
                lb  = float(getattr(v, "LB", float("nan"))) if v is not None else float("nan")
                ub  = float(getattr(v, "UB", float("nan"))) if v is not None else float("nan")
                rows.append((i, col, val, lb, ub))
            print("\n-- XGB INPUT (primeras 30) --")
            for i, col, val, lb, ub in rows[:70]:
                print(f"{i:3d} {col:25s} = {val:8.3f}  [{lb:,.1f},{ub:,.1f}]")
            # escribe CSV
            try:
                import csv
                with open("X_input_after_opt.csv", "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["idx","feature","value","LB","UB"])
                    w.writerows(rows)
                print("[AUDIT] guardado X_input_after_opt.csv")
            except Exception as e:
                print("[AUDIT] no se pudo escribir CSV:", e)

    except Exception as e:
        print(f"[HOUSE SUMMARY] error: {e}")
    
    return {"ok": True}
