"""
CLI para correr el modelo de CONSTRUCCIÃ“N (MIP + XGB) y auditar costos.
ASCII solo (sin emojis) para evitar problemas de consola en Windows.
"""

import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import gurobipy as gp
import joblib
import re

from optimization.construction.config import PARAMS
from optimization.construction.io import get_base_house
from optimization.construction import costs
from optimization.construction.xgb_predictor import XGBBundle
from optimization.construction.gurobi_model import build_mip_embed
from .preprocess_regresion import load_regression_reference_df, prepare_regression_input


def _var_from_x(m: gp.Model, name: str):
    x = getattr(m, "_x", {})
    var = x.get(name)
    if var is None:
        var = x.get(name.replace(" ", "_"))
    return var


def _add_bounds(model: gp.Model, var: gp.Var | None, lb, ub, label: str):
    if var is None:
        return
    if lb is not None:
        model.addConstr(var >= lb, name=f"user_lb_{label}")
    if ub is not None:
        model.addConstr(var <= ub, name=f"user_ub_{label}")


def _apply_user_constraints(model: gp.Model, args) -> None:
    rv = getattr(model, "_report_vars", {})
    lb_ub_pairs = [
        (rv.get("Bedrooms"), args.min_beds, args.max_beds, "beds"),
        (rv.get("FullBath"), args.min_fullbath, args.max_fullbath, "fullbath"),
        (rv.get("HalfBath"), args.min_halfbath, args.max_halfbath, "halfbath"),
        (rv.get("Kitchen"), args.min_kitchen, args.max_kitchen, "kitchen"),
        (rv.get("Garage Area") or _var_from_x(model, "Garage Area"), args.min_garage_area, args.max_garage_area, "garage_area"),
        (rv.get("Total Bsmt SF") or _var_from_x(model, "Total Bsmt SF"), args.min_totalbsmt, args.max_totalbsmt, "total_bsmt"),
        (_var_from_x(model, "Overall Qual"), args.min_overallqual, args.max_overallqual, "overallqual"),
        (_var_from_x(model, "Gr Liv Area"), args.min_grliv, args.max_grliv, "grliv"),
    ]
    for var, lb, ub, label in lb_ub_pairs:
        _add_bounds(model, var, lb, ub, label)


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


def compute_neigh_means(neigh: str, base_csv_path=None, quantile: float = 0.10) -> dict[str, float]:
    df = load_base_df(base_csv_path)
    sub = df[df["Neighborhood"] == neigh]
    if sub.empty:
        return {}
    cols = [
        "1st Flr SF", "2nd Flr SF", "Total Bsmt SF", "Garage Area",
        "Bedroom AbvGr", "Full Bath", "Half Bath", "Kitchen AbvGr",
        "Fireplaces", "Mas Vnr Area", "Gr Liv Area",
    ]
    means: dict[str, float] = {}
    for c in cols:
        if c in sub.columns:
            try:
                mv = float(sub[c].quantile(quantile))
                # refuerza con el mínimo observado (por si la serie es muy corta)
                mv = min(mv, float(sub[c].min()))
            except Exception:
                mv = float('nan')
            if pd.notna(mv):
                means[c] = mv
    return means


def enforce_neigh_means(m: gp.Model, means: dict[str, float], percentile: float = 0.10):
    """Agrega var >= piso_neigh (cuantil bajo o mínimo) para variables clave si existen en el modelo."""
    for col, mv in means.items():
        if pd.isna(mv):
            continue
        # Usa el cuantil bajo del vecindario (percentile), si está disponible en self._neigh_df
        v = m.getVarByName(f"x_{col}") or m.getVarByName(col)
        if v is None:
            continue
        try:
            m.addConstr(v >= float(mv), name=f"neigh_floor__{col.replace(' ', '_')}")
        except Exception:
            pass


def compute_neigh_modes(neigh: str, base_csv_path=None) -> dict[str, str]:
    """Moda por barrio para variables categóricas accionables."""
    df = load_base_df(base_csv_path)
    sub = df[df["Neighborhood"] == neigh]
    if sub.empty:
        return {}
    cols = [
        "Heating", "Electrical", "PavedDrive",
        "Exterior1st", "Exterior2nd", "Foundation",
        "Roof Style", "Roof Matl", "Garage Finish",
    ]
    modes: dict[str, str] = {}
    for c in cols:
        if c in sub.columns:
            try:
                mode_val = sub[c].mode(dropna=True)
                if not mode_val.empty:
                    modes[c] = str(mode_val.iloc[0])
            except Exception:
                continue
    return modes


def canonicalize_category(value: str | None, column: str, base_csv_path=None) -> str | None:
    if value is None:
        return None
    try:
        df = load_base_df(base_csv_path)
    except Exception:
        return value
    if column not in df.columns:
        return value
    uniq = df[column].dropna().unique()
    norm_map = {_norm_key(v): v for v in uniq}
    token = _norm_key(value)
    if token in norm_map:
        canon = norm_map[token]
        if canon != value:
            print(f"[WARN] {column} '{value}' no encontrado exactamente; usando '{canon}'")
        return canon
    print(f"[WARN] {column} '{value}' no existe en la base; se usa tal cual")
    return value


def enforce_neigh_modes(m: gp.Model, modes: dict[str, str]):
    """Fija categorías al valor modal del barrio si existe la OHE correspondiente."""
    for col, val in modes.items():
        v = m.getVarByName(f"{col}__{val}")
        if v is None:
            # intenta con reemplazo de espacios por guion bajo
            v = m.getVarByName(f"{col.replace(' ', '_')}__{val}")
        if v is not None:
            try:
                m.addConstr(v == 1.0, name=f"neigh_mode__{col.replace(' ', '_')}")
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
        # Usa el mismo predictor (con early stopping) que en el embed
        y_full = bundle.predict(Z)
        y_out = _first_scalar(y_full)
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


def _norm_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def materialize_optimal_input_df(m: gp.Model) -> pd.DataFrame | None:
    Xi = getattr(m, "_X_input", None)
    if Xi is None:
        return None
    if isinstance(Xi, dict) and "order" in Xi and "x" in Xi:
        order = list(Xi["order"])
        xvars = Xi["x"]
        df = pd.DataFrame([[0.0] * len(order)], columns=order)
        for c in order:
            v = xvars.get(c)
            try:
                df.loc[0, c] = float(v.X) if hasattr(v, "X") else float(v)
            except Exception:
                df.loc[0, c] = 0.0
        return df
    if hasattr(Xi, "copy"):
        df = Xi.copy()
        for c in getattr(df, "columns", []):
            try:
                val = df.loc[0, c]
                df.loc[0, c] = float(val.X) if hasattr(val, "X") else float(val)
            except Exception:
                continue
        return df
    return None


def report_regression_comparison(m: gp.Model, reg_model, *, reg_df=None, feature_names=None, base_row=None):
    if reg_model is None:
        m._reg_price = float("nan")
        return
    X_df = materialize_optimal_input_df(m)
    if X_df is None:
        print("[REG] no se pudo reconstruir X_input para la regresión")
        m._reg_price = float("nan")
        return
    feat_order = feature_names or list(getattr(reg_model, "feature_names_in_", []))
    try:
        X_reg = prepare_regression_input(X_df, base_row, feat_order, reg_df)
        precio_reg = float(reg_model.predict(X_reg)[0])
        y_price = getattr(m, "_y_price_var", None)
        precio_xgb = float(y_price.X) if y_price is not None else float("nan")
        delta = precio_xgb - precio_reg if pd.notna(precio_xgb) else float("nan")
        pct = (delta / precio_reg * 100) if precio_reg not in (0.0, None) else float("nan")
        print(f"[REG] precio_reg_lineal = ${precio_reg:,.0f} | delta_vs_XGB = ${delta:,.0f} ({pct:.2f}%)")
        m._reg_price = precio_reg
    except Exception as e:
        print(f"[REG] no se pudo evaluar la regresión: {e}")
        m._reg_price = float("nan")


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
    ap.add_argument("--reg-model", type=str, default=None, help="Ruta opcional a un modelo de regresión lineal (joblib) para comparar precios")
    ap.add_argument("--reg-basecsv", type=str, default=None, help="CSV limpio utilizado para entrenar la regresión lineal")
    # Sensibilidades de diseño (bounds manuales)
    ap.add_argument("--min-beds", type=float, default=None, help="Cota inferior para Bedrooms")
    ap.add_argument("--max-beds", type=float, default=None, help="Cota superior para Bedrooms")
    ap.add_argument("--min-fullbath", type=float, default=None, help="Mínimo de baños completos")
    ap.add_argument("--max-fullbath", type=float, default=None, help="Máximo de baños completos")
    ap.add_argument("--min-halfbath", type=float, default=None, help="Mínimo de medios baños")
    ap.add_argument("--max-halfbath", type=float, default=None, help="Máximo de medios baños")
    ap.add_argument("--min-kitchen", type=float, default=None, help="Mínimo de cocinas")
    ap.add_argument("--max-kitchen", type=float, default=None, help="Máximo de cocinas")
    ap.add_argument("--min-overallqual", type=float, default=None, help="Mínimo de Overall Qual")
    ap.add_argument("--max-overallqual", type=float, default=None, help="Máximo de Overall Qual")
    ap.add_argument("--min-grliv", type=float, default=None, help="Mínimo de Gr Liv Area (ft2)")
    ap.add_argument("--max-grliv", type=float, default=None, help="Máximo de Gr Liv Area (ft2)")
    ap.add_argument("--min-garage-area", type=float, default=None, help="Mínimo de Garage Area (ft2)")
    ap.add_argument("--max-garage-area", type=float, default=None, help="Máximo de Garage Area (ft2)")
    ap.add_argument("--min-totalbsmt", type=float, default=None, help="Mínimo de Total Bsmt SF")
    ap.add_argument("--max-totalbsmt", type=float, default=None, help="Máximo de Total Bsmt SF")
    ap.add_argument("--tag", type=str, default=None, help="Etiqueta opcional para identificar la corrida (se guarda en el CSV)")
    args = ap.parse_args()

    reg_model = None
    reg_path = args.reg_model or str(getattr(PATHS, "reg_model_file", ""))
    if reg_path:
        try:
            reg_model = joblib.load(reg_path)
            print(f"[REG] modelo cargado desde {reg_path}")
        except Exception as e:
            print(f"[REG] no se pudo cargar la regresión ({reg_path}): {e}")
    reg_model_path = reg_path if reg_model is not None else None
    reg_df = load_regression_reference_df(args.reg_basecsv) if reg_model is not None else None
    reg_features = list(getattr(reg_model, "feature_names_in_", [])) if reg_model is not None else None

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

    neigh_arg = canonicalize_category(args.neigh, "Neighborhood", base_csv_path=args.basecsv) if args.neigh else args.neigh

    if args.pid is not None:
        base = get_base_house(args.pid, base_csv=args.basecsv)
        try:
            base_row = base.row
        except AttributeError:
            base_row = base if isinstance(base, pd.Series) else pd.Series(base)
    else:
        base_row = _make_seed_row(neigh_arg, args.lot)
    base_row["Neighborhood"] = canonicalize_category(base_row.get("Neighborhood"), "Neighborhood", base_csv_path=args.basecsv)
    if base_row.get("Neighborhood"):
        args.neigh = base_row["Neighborhood"]
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

    # Auto‑calibración del offset b0 (alinear y_log embed con margen del XGB)
    try:
        bundle.autocalibrate_offset(base_row)
    except Exception:
        bundle.autocalibrate_offset(None)

    m: gp.Model = build_mip_embed(base_row=base_row, budget=args.budget, ct=ct, bundle=bundle)
    _apply_user_constraints(m, args)

    # Si se especifica barrio, fuerza mínimos de atributos = cuantil bajo/min del barrio (guard-rail)
    neigh_token = base_row.get("Neighborhood", args.neigh)
    if neigh_token:
        means = compute_neigh_means(str(neigh_token), base_csv_path=args.basecsv, quantile=0.10)
        if means:
            enforce_neigh_means(m, means, percentile=0.10)
        modes = compute_neigh_modes(str(neigh_token), base_csv_path=args.basecsv)
        if modes:
            enforce_neigh_modes(m, modes)

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
    m.Params.FuncPieces = 500
    m.Params.FuncPieceError = 1e-6
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

    report_regression_comparison(m, reg_model, reg_df=reg_df, feature_names=reg_features, base_row=base_row)

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
    if args.audit:
        top_n = 15 if args.quiet else 50
        audit_cost_breakdown_vars(m, top=top_n)
    audit_predict_outside(m, bundle)

    print("\n" + "="*60)
    print("                     FIN RESULTADOS")
    print("="*60)

    reg_price = float(getattr(m, "_reg_price", float("nan")))
    precio_xgb = y_price
    if not np.isnan(reg_price):
        delta = precio_xgb - reg_price
        pct = (delta / reg_price * 100.0) if reg_price not in (0.0, None) else float("nan")
        print("\n==============================")
        print("     COMPARACIÓN CON REGRESIÓN")
        print("==============================")
        print(f"  Precio XGB (post):        {money(precio_xgb)}")
        print(f"  Precio regresión lineal:  {money(reg_price)}")
        print(f"  Diferencia XGB-REG:       {money(delta)} ({pct:.2f}%)\n")
    else:
        print("\n[CREG] no se pudo calcular comparación con regresión\n")

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
            reg_price = float(getattr(m, "_reg_price", float('nan')))
            delta_reg = (y_price - reg_price) if (np.isfinite(y_price) and np.isfinite(reg_price)) else float('nan')
            delta_reg_pct = ((delta_reg / reg_price) * 100.0) if np.isfinite(delta_reg) and reg_price not in (0.0, None) else float('nan')
            price_out = float(getattr(m, '_diag_y_price_out', float('nan')))
            y_log_in = float(getattr(m, '_y_log_var', None).X) if getattr(m, '_y_log_var', None) is not None else float('nan')
            y_log_out = float(getattr(m, '_diag_y_log_out', float('nan')))
            budget_val = float(getattr(m, "_budget_usd", args.budget))
            slack_val = budget_val - total_cost if np.isfinite(total_cost) else float('nan')
            opt_df = materialize_optimal_input_df(m)
            def xfeat(name):
                if opt_df is not None and name in opt_df.columns:
                    try:
                        return float(opt_df.loc[0, name])
                    except Exception:
                        return float('nan')
                return float('nan')
            row = {
                'timestamp': datetime.utcnow().isoformat(),
                'profile': args.profile,
                'tag': str(args.tag or ""),
                'neigh': str(base_row.get('Neighborhood', args.neigh or 'N/A')),
                'lot': float(base_row.get('LotArea', args.lot or 0.0)),
                'budget': float(args.budget),
                'status': int(getattr(m, 'Status', -1)),
                'runtime_s': float(getattr(m, 'Runtime', 0.0)),
                'gap': float(getattr(m, 'MIPGap', float('nan'))),
                'xgb_model': str(getattr(bundle, 'model_path', PATHS.xgb_model_file)),
                'reg_model': reg_model_path or "",
                'y_price': y_price,
                'y_price_out': price_out,
                'y_log_in': y_log_in,
                'y_log_out': y_log_out,
                'reg_price': reg_price,
                'delta_reg_abs': delta_reg,
                'delta_reg_pct': delta_reg_pct,
                'cost': total_cost,
                'obj': obj,
                'slack': slack_val,
                'floor1': vnum('Floor1'),
                'floor2': vnum('Floor2'),
                'area_1st': vnum('1st Flr SF'),
                'area_2nd': vnum('2nd Flr SF'),
                'bsmt': vnum('Total Bsmt SF'),
                'gr_liv_area': xfeat('Gr Liv Area'),
                'garage_area': xfeat('Garage Area'),
                'screen_porch': xfeat('Screen Porch'),
                'pool_area': xfeat('Pool Area'),
                'beds': vnum('Bedrooms'),
                'fullbath': vnum('FullBath'),
                'halfbath': vnum('HalfBath'),
                'kitchen': vnum('Kitchen'),
                'overall_qual': xfeat('Overall Qual'),
                'overall_cond': xfeat('Overall Cond'),
                'kitchen_qual': xfeat('Kitchen Qual'),
                'heating_qc': xfeat('Heating QC'),
                'bldg_type': args.bldg or "",
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
