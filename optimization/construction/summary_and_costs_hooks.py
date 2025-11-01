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
        print("\n==== HOUSE SUMMARY ====")
        rv = getattr(m, "_report_vars", {})
        def val(name):
            v = rv.get(name)
            return None if v is None else (v.X if hasattr(v, "X") else float(v))
        print(f"pisos: floor1={val('Floor1')}, floor2={val('Floor2')}")
        print(f"areas: 1st={val('1st Flr SF')}, 2nd={val('2nd Flr SF')}, bsmt={val('Total Bsmt SF')}, garage={val('Garage Area')}")
        print(f"ambientes: beds={val('Bedrooms')}, bathsF={val('FullBath')}, bathsH={val('HalfBath')}, kitchen={val('Kitchen')}")

        # techo elegido
        roof = []
        for s in ['Flat','Gable','Gambrel','Hip','Mansard','Shed']:
            for mm in ['ClyTile','CompShg','Membran','Metal','Roll','Tar&Grv','WdShake','WdShngl']:
                y = m.getVarByName(f"Y__{s}__{mm}")
                if y and y.X > 0.5:
                    roof.append(f"{s}-{mm}")
        if roof:
            print("roof:", ", ".join(roof))

        # categorias elegidas clave (Heating/Electrical/PavedDrive/Exterior/Foundation)
        def chosen(prefix, opts):
            for o in opts:
                v = m.getVarByName(f"{prefix}__{o}")
                if v is not None and getattr(v, 'X', 0.0) > 0.5:
                    return o
            return None
        heating = chosen('Heating', ['Floor','GasA','GasW','Grav','OthW','Wall'])
        electrical = chosen('Electrical', ['SBrkr','FuseA','FuseF','FuseP','Mix'])
        paved = chosen('PavedDrive', ['Y','P','N'])
        ext1 = chosen('Exterior1st', ['VinylSd','MetalSd','Wd Sdng','HdBoard','Stucco','Plywood','CemntBd','BrkFace','BrkComm','WdShngl','AsbShng','Stone','ImStucc','AsphShn','CBlock'])
        ext2 = chosen('Exterior2nd', ['VinylSd','MetalSd','Wd Sdng','HdBoard','Stucco','Plywood','CemntBd','BrkFace','BrkComm','WdShngl','AsbShng','Stone','ImStucc','AsphShn','CBlock'])
        fnd = chosen('Foundation', ['BrkTil','CBlock','PConc','Slab','Stone','Wood'])
        try:
            area_fnd = m.getVarByName(f"FA__{fnd}")
            area_fnd = float(getattr(area_fnd, 'X', 0.0)) if area_fnd is not None else 0.0
        except Exception:
            area_fnd = 0.0
        feats = []
        if heating: feats.append(f"heating={heating}")
        if electrical: feats.append(f"electrical={electrical}")
        if paved: feats.append(f"paved={paved}")
        if ext1: feats.append(f"ext1={ext1}")
        if ext2: feats.append(f"ext2={ext2}")
        if fnd: feats.append(f"foundation={fnd} ({area_fnd:.0f} ft2)")
        if feats:
            print("features:", ", ".join(feats))

        # economia
        yp = getattr(m, "_y_price_var", None)
        yl = getattr(m, "_y_log_var", None)
        if yp is not None:
            print(f"precio_predicho = {yp.X:,.0f}")
        if yl is not None:
            print(f"log_precio = {yl.X:.4f}")

        # detalle de áreas por piso y por ambiente (si existen)
        try:
            names = [
                ("AreaKitchen1","AreaKitchen2","AreaKitchen"),
                ("AreaFullBath1","AreaFullBath2","AreaFullBath"),
                ("AreaHalfBath1","AreaHalfBath2","AreaHalfBath"),
                ("AreaBedroom1","AreaBedroom2","AreaBedroom"),
                ("AreaOther1","AreaOther2",None),
            ]
            print("\n-- areas por ambiente (ft2) --")
            for n1,n2,ntot in names:
                v1 = m.getVarByName(n1); v2 = m.getVarByName(n2); vt = m.getVarByName(ntot) if ntot else None
                a1 = float(getattr(v1,'X',0.0)) if v1 is not None else 0.0
                a2 = float(getattr(v2,'X',0.0)) if v2 is not None else 0.0
                at = float(getattr(vt,'X',a1+a2)) if vt is not None else a1+a2
                label = (ntot or n1.replace('1','')).replace('Area','')
                print(f"{label:<12s}: 1st={a1:6.1f}  2nd={a2:6.1f}  total={at:6.1f}")
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
