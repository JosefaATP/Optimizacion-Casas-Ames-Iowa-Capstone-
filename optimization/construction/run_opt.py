"""
CLI para correr el modelo de CONSTRUCCIÃ“N (MIP + XGB) y auditar costos.
ASCII solo (sin emojis) para evitar problemas de consola en Windows.
"""

import argparse
import pandas as pd
import numpy as np
import gurobipy as gp

from .config import PARAMS
from .io import get_base_house
from . import costs
from .xgb_predictor import XGBBundle
from .gurobi_model import build_mip_embed, summarize_solution


def money(v: float) -> str:
    try:
        f = float(pd.to_numeric(v, errors="coerce"))
    except Exception:
        return str(v)
    if pd.isna(f):
        return "-"
    return f"${f:,.0f}"


def audit_cost_breakdown_vars(m: gp.Model, top: int = 30):
    expr = getattr(m, "_lin_cost_expr", None)
    if expr is None:
        print("[COST-BREAKDOWN] no _lin_cost_expr en el modelo")
        return
    try:
        vs = expr.getVars(); cs = expr.getCoeffs()
        X = m.getAttr("X", vs)
        rows = []
        for v, c, x in zip(vs, cs, X):
            try:
                contrib = float(c) * float(x)
            except Exception:
                continue
            rows.append((v.VarName, float(c), float(x), contrib))
        rows.sort(key=lambda t: abs(t[3]), reverse=True)
        print("\n[COST-BREAKDOWN] top terminos de costo:")
        for name, c, x, contr in rows[:top]:
            print(f"  {name:<35s} coef={c:>10.4f} * X={x:>10.4f}  => {contr:>10.2f}")
        total = sum(contr for _, _, _, contr in rows)
        cvar = m.getVarByName("cost_model")
        cval = float(getattr(cvar, 'X', float('nan'))) if cvar is not None else float('nan')
        print(f"[COST-CHECK] suma_terms={total:,.2f}  | cost_model={cval:,.2f}  | diff={cval-total:,.2f}")
    except Exception:
        # Fallback: _cost_terms
        terms = getattr(m, "_cost_terms", [])
        rows = []
        for label, coef, var in terms:
            try:
                xv = var.X if hasattr(var, "X") else float(var)
                contr = float(coef) * float(xv)
            except Exception:
                continue
            rows.append((label, float(coef), float(xv), contr))
        rows.sort(key=lambda t: abs(t[3]), reverse=True)
        print("\n[COST-BREAKDOWN] top terminos de costo (fallback):")
        for name, c, x, contr in rows[:top]:
            print(f"  {name:<35s} coef={c:>10.4f} * X={x:>10.4f}  => {contr:>10.2f}")
        try:
            total = sum(contr for _, _, _, contr in rows)
            cvar = m.getVarByName("cost_model")
            cval = float(getattr(cvar, 'X', float('nan'))) if cvar is not None else float('nan')
            print(f"[COST-CHECK] suma_terms={total:,.2f}  | cost_model={cval:,.2f}  | diff={cval-total:,.2f}")
        except Exception:
            pass

        # categorÃ­as seleccionadas (auditorÃ­a completa)
        ct = getattr(m, "_ct", None)
        if ct is not None:
            def pick(prefix, opts):
                for o in opts:
                    v = m.getVarByName(f"{prefix}__{o}")
                    if v is not None and getattr(v, 'X', 0.0) > 0.5:
                        return o, v
                return None, None

            groups = [
                ("Heating",   ["Floor","GasA","GasW","Grav","OthW","Wall"]),
                ("Electrical",["SBrkr","FuseA","FuseF","FuseP","Mix"]),
                ("PavedDrive",["Y","P","N"]),
                ("RoofStyle", ["Flat","Gable","Gambrel","Hip","Mansard","Shed"]),
                ("RoofMatl",  ["ClyTile","CompShg","Membran","Metal","Roll","Tar&Grv","WdShake","WdShngl"]),
                ("Exterior1st",["VinylSd","MetalSd","Wd Sdng","HdBoard","Stucco","Plywood","CemntBd","BrkFace","BrkComm","WdShngl","AsbShng","Stone","ImStucc","AsphShn","CBlock"]),
                ("Exterior2nd",["VinylSd","MetalSd","Wd Sdng","HdBoard","Stucco","Plywood","CemntBd","BrkFace","BrkComm","WdShngl","AsbShng","Stone","ImStucc","AsphShn","CBlock"]),
                ("Foundation",["BrkTil","CBlock","PConc","Slab","Stone","Wood"]),
                ("GarageFinish",["NA","Fin","RFn","Unf"]),
                ("Fence",     ["GdPrv","MnPrv","GdWo","MnWw","No aplica"]),
                ("MiscFeature",["Elev","Gar2","Othr","Shed","TenC","No aplica"]),
            ]
            print("\n[COST-CATEGORIES] seleccion y costo unitario (contrib puede ser 0):")
            for gname, opts in groups:
                opt, var = pick(gname, opts)
                if opt is None:
                    continue
                xv = float(getattr(var, 'X', 0.0)) if var is not None else 0.0
                coef = 0.0; contr = 0.0
                try:
                    if gname == 'Heating':
                        coef = float(ct.heating_type_costs.get(opt, 0.0)); contr = coef * xv
                    elif gname == 'Electrical':
                        coef = float(ct.electrical_type_costs.get(opt, 0.0)); contr = coef * xv
                    elif gname == 'PavedDrive':
                        coef = float(ct.paved_drive_costs.get(opt, 0.0)); contr = coef * xv
                    elif gname == 'RoofMatl':
                        coef = float(ct.roof_matl_fixed.get(opt, 0.0)); contr = coef * xv
                    elif gname == 'RoofStyle':
                        coef = float(getattr(ct, 'roof_style_costs', {}).get(opt, 0.0)); contr = coef * xv
                    elif gname in ('Exterior1st','Exterior2nd'):
                        coef = float(ct.exterior_matl_lumpsum.get(opt, 0.0)); contr = coef * xv
                    elif gname == 'GarageFinish':
                        key = opt if opt != 'NA' else 'No aplica'
                        coef = float(ct.garage_finish_costs_sqft.get(key, 0.0)); contr = coef * xv
                    elif gname == 'Fence':
                        coef = float(ct.fence_category_costs.get(opt, 0.0)); contr = coef * xv
                    elif gname == 'MiscFeature':
                        coef = float(ct.misc_feature_costs.get(opt, 0.0)); contr = coef * xv
                    elif gname == 'Foundation':
                        z = m.getVarByName(f"FA__{opt}")
                        area = float(getattr(z, 'X', 0.0)) if z is not None else 0.0
                        coef = float(ct.foundation_cost_per_sf.get(opt, 0.0)); contr = coef * area
                        print(f"  {gname:<12s} -> {opt:<12s}  coef={coef:>10.4f} * X={area:>6.1f} => {contr:>10.2f}")
                        continue
                except Exception:
                    coef = 0.0; contr = 0.0
                print(f"  {gname:<12s} -> {opt:<12s}  coef={coef:>10.4f} * X={xv:>4.1f} => {contr:>10.2f}")


def audit_predict_outside(m: gp.Model, bundle: XGBBundle):
    X_in = getattr(m, "_X_input", None)
    if X_in is None:
        print("[AUDIT] no hay _X_input para predecir fuera")
        return
    if isinstance(X_in, dict) and "order" in X_in and "x" in X_in:
        order = list(X_in["order"])  
        xvars = X_in["x"]            
        Z = pd.DataFrame([[0.0]*len(order)], columns=order)
        for c in order:
            v = xvars.get(c)
            try:
                Z.loc[0, c] = float(v.X) if hasattr(v, "X") else float(v)
            except Exception:
                Z.loc[0, c] = 0.0
    else:
        Z = X_in.copy()
        if getattr(Z, "empty", True):
            print("[AUDIT] _X_input vacio")
            return
    def _first_scalar(obj):
        try:
            import numpy as _np
            import pandas as _pd
            if isinstance(obj, _pd.Series):
                return float(obj.iloc[0])
            if isinstance(obj, (list, tuple)):
                return float(obj[0])
            if hasattr(obj, 'shape') and getattr(obj, 'shape', None):
                return float(_np.ravel(obj)[0])
            return float(obj)
        except Exception:
            return float('nan')

    try:
        y_pred = bundle.predict(Z)
        y_out = _first_scalar(y_pred)
        if not (y_out == y_out):  # NaN check
            # Fallback: usa el pipeline completo (sin early stopping)
            try:
                y_full = bundle.pipe_full.predict(Z)
                y_out = _first_scalar(y_full)
                print("[AUDIT] fallback pipe_full para predict fuera (iter_range dio NaN)")
            except Exception:
                pass
        print(f"[AUDIT] predict fuera = {y_out:,.2f}")
        # Diagnóstico: comparar y_log interno vs margin del XGB afuera
        try:
            ylog_raw = bundle.predict_log_raw(Z)
            ylog_out = _first_scalar(ylog_raw)
        except Exception:
            ylog_out = float('nan')
        ylog_in = float(getattr(m, '_y_log_var', None).X) if getattr(m, '_y_log_var', None) is not None else float('nan')
        print(f"[AUDIT] y_log_in={ylog_in:.4f} | y_log_out={ylog_out:.4f} | delta={ylog_in - ylog_out:+.4f}")
        yprice_in = float(getattr(m, '_y_price_var', None).X) if getattr(m, '_y_price_var', None) is not None else float('nan')
        print(f"[AUDIT] y_price_in={yprice_in:,.2f} | y_price_out={y_out:,.2f} | delta={yprice_in - y_out:,.2f}")
        # Verificar orden de features (Booster vs Z)
        try:
            bo = bundle.booster_feature_order()
            if list(Z.columns) != list(bo):
                # imprime primeras diferencias
                mism = [i for i,(a,b) in enumerate(zip(Z.columns, bo)) if a!=b]
                print(f"[AUDIT] booster_feature_order difiere en {len(mism)} posiciones (muestra hasta 10):")
                for i in mism[:10]:
                    print(f"  idx {i}: Z={Z.columns[i]} | booster={bo[i]}")
        except Exception as _e:
            pass
        # Guardar para CSV
        try:
            m._diag_y_price_out = float(y_out)
            m._diag_y_log_out = float(ylog_out)
        except Exception:
            pass
    except Exception as e:
        print(f"[AUDIT] fallo predict fuera: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, default=None)
    ap.add_argument("--neigh", type=str, default=None)
    ap.add_argument("--lot", type=float, default=None)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--xgbdir", type=str, default=None, help="carpeta con model_xgb.joblib/meta.json a usar")
    ap.add_argument("--basecsv", type=str, default=None)
    ap.add_argument("--debug-xgb", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--profile", type=str, default="balanced", choices=["balanced","feasible","bound"], help="perfil de solver: balanced/feasible/bound")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--outcsv", type=str, default=None, help="ruta CSV para append de resultados por corrida")
    ap.add_argument("--bldg", type=str, default=None, help="tipo de edificio (por ejemplo: 1Fam, TwnhsE, TwnhsI, Duplex, 2FmCon)")
    args = ap.parse_args()

    def _make_seed_row(neigh: str, lot: float) -> pd.Series:
        base_defaults = {
            "Neighborhood": neigh or "NAmes",
            "LotArea": float(lot or 7000.0),
            "MS SubClass": 20,
            "MSZoning": "RL",
            "Street": "Pave",
            "Lot Shape": "Reg",
            "LandContour": "Lvl",
            "LotConfig": "Inside",
            "LandSlope": "Gtl",
            "BldgType": "1Fam",
            "HouseStyle": "1Story",
            "YearBuilt": 2025,
            "YearRemodAdd": 2025,
            "Foundation": "PConc",
            "Condition 1": "Norm",
            "Condition 2": "Norm",
            "Roof Style": "Gable",
            "Garage Cars": 0,
            "Low Qual Fin SF": 0,
            "Bsmt Qual": "TA",
            "Bsmt Full Bath": 0,
            "Bsmt Half Bath": 0,
            "Month Sold": 6,
            "Year Sold": 2025,
            "Sale Type": "WD",
            "Sale Condition": "Normal",
            "1st Flr SF": 0.0,
            "2nd Flr SF": 0.0,
            "Gr Liv Area": 0.0,
            "Total Bsmt SF": 0.0,
            "Bsmt Unf SF": 0.0,
            "BsmtFin SF 1": 0.0,
            "BsmtFin SF 2": 0.0,
            "Garage Area": 0.0,
            "Wood Deck SF": 0.0,
            "Open Porch SF": 0.0,
            "Enclosed Porch": 0.0,
            "3Ssn Porch": 0.0,
            "Screen Porch": 0.0,
            "Pool Area": 0.0,
            "Bedroom AbvGr": 0,
            "Full Bath": 0,
            "Half Bath": 0,
            "Kitchen AbvGr": 0,
        }
        base_defaults.update({
            "OverallQual": 10,
            "OverallCond": 10,
            "Exter Qual": 4,
            "ExterCond": 4,
            "Bsmt Qual": 4,
            "Heating QC": 4,
            "Kitchen Qual": 4,
            "Utilities": 3,
            "Low Qual Fin SF": 0.0,
        })
        return pd.Series(base_defaults)

    if args.pid is not None:
        base = get_base_house(args.pid, base_csv=args.basecsv)
        try:
            base_row = base.row
        except AttributeError:
            base_row = base if isinstance(base, pd.Series) else pd.Series(base)
    else:
        base_row = _make_seed_row(args.neigh, args.lot)
    # Override de tipo de edificio si se pide por CLI
    if args.bldg:
        try:
            base_row["BldgType"] = str(args.bldg)
        except Exception:
            pass

    ct = costs.CostTables()
    if args.xgbdir:
        from pathlib import Path
        model_path = Path(args.xgbdir) / "model_xgb.joblib"
        bundle = XGBBundle(model_path=model_path)
    else:
        bundle = XGBBundle()

    m: gp.Model = build_mip_embed(base_row=base_row, budget=args.budget, ct=ct, bundle=bundle)

    time_limit = PARAMS.time_limit
    if args.fast:
        time_limit = 60
    if args.deep:
        time_limit = 900

    # Perfil de solver
    prof = args.profile
    m.Params.MIPGap = PARAMS.mip_gap
    m.Params.TimeLimit = time_limit
    m.Params.LogToConsole = PARAMS.log_to_console
    m.Params.FeasibilityTol = 1e-7
    m.Params.IntFeasTol = 1e-7
    m.Params.OptimalityTol = 1e-7
    if prof == "feasible":
        m.Params.NumericFocus = 2
        m.Params.MIPFocus = 1         # prioriza factibilidad
        m.Params.Heuristics = 0.3
        m.Params.Cuts = 1
        m.Params.Presolve = 2
        m.Params.Symmetry = 2
    elif prof == "bound":
        m.Params.NumericFocus = 1
        m.Params.MIPFocus = 3         # prioriza bound (gap)
        m.Params.Heuristics = 0.05
        m.Params.Cuts = 2
        m.Params.Presolve = 2
        m.Params.Symmetry = 2
        m.Params.Method = 2           # barrier para la relajaciÃ³n
    else:  # balanced
        m.Params.NumericFocus = 2
        m.Params.MIPFocus = 1
        m.Params.Heuristics = 0.2
        m.Params.Cuts = 1
        m.Params.Presolve = 2

    m.optimize()
    print(f"[STATUS] gurobi Status = {int(getattr(m, 'Status', -1))}")

    try:
        summarize_solution(m)
    except Exception as e:
        print(f"[HOUSE SUMMARY] no disponible: {e}")

    st = m.Status
    if st in (gp.GRB.INF_OR_UNBD, gp.GRB.INFEASIBLE, gp.GRB.UNBOUNDED):
        try:
            print("[DEBUG] infeasible/unbounded; re-ejecutando con DualReductions=0 y computeIIS()")
            m.Params.DualReductions = 0
            m.optimize()
            if m.Status in (gp.GRB.INFEASIBLE,):
                m.computeIIS()
                tag = f"construction_neigh_{args.neigh}_lot_{args.lot}_budget_{args.budget}"
                m.write(f"{tag}_conflict.ilp")
                m.write(f"{tag}_model.lp")
                print(f"[DEBUG] escrito IIS en {tag}_conflict.ilp y modelo en {tag}_model.lp")
        except Exception as e:
            print("[DEBUG] fallo computeIIS:", e)
        print("[ERR] modelo infeasible/unbounded. usa los archivos *_conflict.ilp y *_model.lp")
        return
    if st not in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT) or m.SolCount == 0:
        print("[WARN] no hay solucion valida")
        return

    y_var = getattr(m, "_y_price_var", None)
    c_var = m.getVarByName("cost_model") or getattr(m, "_cost_var", None)
    y_price = float(y_var.X) if y_var is not None else float("nan")
    total_cost = float(c_var.X) if c_var is not None else float("nan")
    obj = float(getattr(m, "objVal", float("nan")))

    print("\n" + "="*60)
    print("             RESULTADOS OPTIMIZACION CONSTRUCCION")
    print("="*60 + "\n")

    print(f"PID: {base_row.get('PID', 'N/A')} - {base_row.get('Neighborhood', 'N/A')} | Presupuesto: {money(args.budget)}")
    print(f"Tiempo: {getattr(m, 'Runtime', 0.0):.2f}s | Gap: {getattr(m, 'MIPGap', 0.0)*100:.4f}%\n")

    print("resumen economico")
    print(f"  Precio predicho (post):  {money(y_price)}")
    print(f"  Costos totales modelo:   {money(total_cost)}")
    print(f"  Objetivo (utilidad):     {money(obj)}  (= precio - costo)")
    try:
        budget = float(getattr(m, "_budget_usd", args.budget))
        print(f"  Slack presupuesto:       {money(budget - total_cost)}")
    except Exception:
        pass

    m._print_categories = bool(args.audit)
    top_n = 15 if args.quiet else 50
    audit_cost_breakdown_vars(m, top=top_n)
    audit_predict_outside(m, bundle)

    print("\n" + "="*60)
    print("                     FIN RESULTADOS")
    print("="*60 + "\n")

    # ================= CSV append (si se pidió) =================
    if args.outcsv:
        try:
            import csv, os
            os.makedirs(os.path.dirname(args.outcsv), exist_ok=True)
            rv = getattr(m, "_report_vars", {})
            def vnum(name):
                try:
                    v = rv.get(name)
                    return float(v.X) if hasattr(v, 'X') else (float(v) if v is not None else float('nan'))
                except Exception:
                    return float('nan')
            row = {
                'neigh': str(base_row.get('Neighborhood', args.neigh or 'N/A')),
                'lot': float(base_row.get('LotArea', args.lot or 0.0)),
                'budget': float(args.budget),
                'status': int(getattr(m, 'Status', -1)),
                'runtime_s': float(getattr(m, 'Runtime', 0.0)),
                'gap': float(getattr(m, 'MIPGap', float('nan'))),
                'y_price': y_price,
                'y_price_out': float(getattr(m, '_diag_y_price_out', float('nan'))),
                'y_log_in': float(getattr(m, '_y_log_var', None).X) if getattr(m, '_y_log_var', None) is not None else float('nan'),
                'y_log_out': float(getattr(m, '_diag_y_log_out', float('nan'))),
                'cost': total_cost,
                'obj': obj,
                'floor1': vnum('Floor1'),
                'floor2': vnum('Floor2'),
                'area_1st': vnum('1st Flr SF'),
                'area_2nd': vnum('2nd Flr SF'),
                'bsmt': vnum('Total Bsmt SF'),
                'beds': vnum('Bedrooms'),
                'fullbath': vnum('FullBath'),
                'halfbath': vnum('HalfBath'),
                'kitchen': vnum('Kitchen'),
            }
            write_header = not os.path.exists(args.outcsv)
            with open(args.outcsv, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    w.writeheader()
                w.writerow(row)
        except Exception as e:
            print(f"[WARN] No se pudo escribir outcsv: {e}")


if __name__ == "__main__":
    main()


