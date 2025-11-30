#!/usr/bin/env python3
"""
Extrae EXHAUSTIVAMENTE todos los cambios recomendados en una optimización.
Muestra: base -> solución, costo asignado, y validación.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='ignore')

import pandas as pd
import numpy as np
from optimization.remodel.run_opt import (
    build_mip_embed, build_base_input_row, rebuild_embed_input_df,
    get_base_house, PARAMS
)
from optimization.remodel.costs import CostTables
from optimization.remodel.xgb_predictor import XGBBundle
from optimization.remodel.features import MODIFIABLE
import gurobipy as gp

def extract_all_changes(pid, budget, time_limit=90):
    """Ejecuta optimización y extrae TODOS los cambios"""
    
    print(f"\n{'='*100}")
    print(f"EXTRACCION DE CAMBIOS: PID {pid}")
    print(f"{'='*100}\n")
    
    # Setup
    base = get_base_house(pid)
    base_row = base.row if hasattr(base, 'row') else base
    ct = CostTables()
    bundle = XGBBundle()
    
    # Precios base
    X_base = build_base_input_row(bundle, base_row)
    precio_base = float(bundle.predict(X_base).iloc[0])
    
    print(f"[BASE] Precio casa original: ${precio_base:,.2f}\n")
    
    # Build & optimize
    m = build_mip_embed(
        base_row,
        budget,
        ct,
        bundle,
        base_price=precio_base,
        fix_to_base=False,
    )
    
    m.Params.MIPGap = PARAMS.mip_gap
    m.Params.TimeLimit = time_limit
    m.Params.LogToConsole = 0
    m.Params.FeasibilityTol = 1e-7
    m.Params.IntFeasTol = 1e-7
    m.Params.OptimalityTol = 1e-7
    
    print(f"[OPTIMIZATION] Ejecutando...", flush=True)
    m.optimize()
    
    if m.status != gp.GRB.OPTIMAL:
        print(f"[ERROR] No óptima (status={m.status})")
        return None
    
    # Extraer solución
    X_sol = rebuild_embed_input_df(m, m._X_base_numeric)
    
    print(f"\n{'='*100}")
    print(f"CAMBIOS DETECTADOS")
    print(f"{'='*100}\n")
    
    changes = []
    
    # Iterar sobre TODOS los features modificables
    for spec in MODIFIABLE:
        fname = spec.name
        
        # Obtener valores base y solución
        if fname not in X_sol.index or fname not in m._X_base_numeric.index:
            continue
        
        base_val = float(m._X_base_numeric.loc[fname, 0])
        sol_val = float(X_sol.loc[fname, 0])
        
        # Solo reportar si hay cambio
        if abs(base_val - sol_val) < 1e-6:
            continue
        
        delta = sol_val - base_val
        
        # Intentar asignar costo
        cost = _estimate_cost(fname, base_val, sol_val, ct, base_row)
        
        changes.append({
            'feature': fname,
            'base': base_val,
            'solution': sol_val,
            'delta': delta,
            'cost': cost,
            'cost_assigned': cost is not None and cost > 0
        })
    
    # Mostrar cambios
    if not changes:
        print("[NO CHANGES] La solución es igual a la base")
        return None
    
    print(f"Total cambios: {len(changes)}\n")
    
    # Agrupar por tipo
    numeric_changes = [c for c in changes if isinstance(c['solution'], (int, float)) and 'is_' not in c['feature']]
    binary_changes = [c for c in changes if 'is_' in c['feature']]
    
    if numeric_changes:
        print(f"[NUMERIC CHANGES] ({len(numeric_changes)})")
        print(f"{'-'*100}")
        for c in numeric_changes:
            cost_str = f"${c['cost']:,.2f}" if c['cost'] else "NO COSTO ASIGNADO"
            marker = "!" if not c['cost_assigned'] else ""
            print(f"  {marker} {c['feature']:<40s} {c['base']:>12.4f} -> {c['solution']:>12.4f} (Δ={c['delta']:>+10.4f}) | {cost_str}")
    
    if binary_changes:
        print(f"\n[BINARY CATEGORICAL CHANGES] ({len(binary_changes)})")
        print(f"{'-'*100}")
        for c in binary_changes:
            if c['solution'] > 0.5:  # Cambio de 0 a 1
                cost_str = f"${c['cost']:,.2f}" if c['cost'] else "NO COSTO"
                marker = "!" if not c['cost_assigned'] else ""
                print(f"  {marker} {c['feature']:<40s} disabled -> ENABLED | {cost_str}")
    
    # Resumen de cambios sin costo
    no_cost = [c for c in changes if not c['cost_assigned']]
    if no_cost:
        print(f"\n[ADVERTENCIA] {len(no_cost)} CAMBIOS SIN COSTO ASIGNADO:")
        print(f"{'-'*100}")
        for c in no_cost:
            print(f"  ! {c['feature']:<40s} {c['base']:>12.4f} -> {c['solution']:>12.4f}")
    
    # Totales
    total_cost = sum(c['cost'] or 0 for c in changes)
    
    # Precio final
    precio_opt = float(bundle.predict(X_sol).iloc[0])
    delta_precio = precio_opt - precio_base
    
    print(f"\n{'='*100}")
    print(f"RESUMEN ECONOMICO")
    print(f"{'='*100}")
    print(f"Precio base:           ${precio_base:>15,.2f}")
    print(f"Precio remodelado:     ${precio_opt:>15,.2f}")
    print(f"Ganancia (delta):      ${delta_precio:>15,.2f}")
    print(f"Costo total remodelación: ${total_cost:>13,.2f}")
    print(f"ROI (ganancia - costo):   ${delta_precio - total_cost:>13,.2f}")
    if total_cost > 0:
        print(f"ROI %:                 {100 * (delta_precio - total_cost) / total_cost:>15.1f}%")
    print(f"\nMIP Objective Value:   ${m.ObjVal:>15,.2f}")
    
    return {
        'changes': changes,
        'precio_base': precio_base,
        'precio_opt': precio_opt,
        'total_cost': total_cost,
        'model': m,
        'X_sol': X_sol
    }

def _estimate_cost(fname, base_val, sol_val, ct, base_row):
    """
    Estima el costo de un cambio.
    Retorna el costo en USD o None si no se puede estimar.
    """
    
    # Features numéricas de área
    if 'Area' in fname or 'SF' in fname or 'Gr Liv' in fname:
        # Costo de construcción
        if fname == "Gr Liv Area":
            delta = sol_val - base_val
            return abs(delta) * ct.construction_cost if abs(delta) > 0.1 else 0
        if fname in ["1st Flr SF", "BsmtFin SF 1", "BsmtFin SF 2"]:
            delta = sol_val - base_val
            return abs(delta) * ct.construction_cost if abs(delta) > 0.1 else 0
        if "Porch" in fname or "Deck" in fname or "Pool Area" in fname:
            # Costo por tipo de porch
            if "Wood Deck" in fname:
                return abs(sol_val - base_val) * ct.wooddeck_cost
            elif "Open Porch" in fname:
                return abs(sol_val - base_val) * ct.openporch_cost
            elif "Enclosed" in fname:
                return abs(sol_val - base_val) * ct.enclosedporch_cost
            elif "3Ssn" in fname:
                return abs(sol_val - base_val) * ct.threessnporch_cost
            elif "Screen" in fname:
                return abs(sol_val - base_val) * ct.screenporch_cost
            elif "Pool" in fname:
                return abs(sol_val - base_val) * ct.pool_area_cost
        if "Garage Area" in fname:
            return abs(sol_val - base_val) * ct.construction_cost
    
    # Bathrooms/Bedrooms
    if "Full Bath" in fname:
        delta = int(sol_val) - int(base_val)
        if delta > 0:
            return delta * ct.add_fullbath_cost
    if "Half Bath" in fname:
        delta = int(sol_val) - int(base_val)
        if delta > 0:
            return delta * ct.add_halfbath_cost
    
    # Ordinales con tabla de costo
    if fname == "Kitchen Qual":
        return ct.kitchen_level_cost(int(sol_val)) - ct.kitchen_level_cost(int(base_val))
    
    if fname == "Exter Qual":
        return ct.exter_qual_cost(f"{int(sol_val)}") - ct.exter_qual_cost(f"{int(base_val)}")
    
    if fname == "Exter Cond":
        return ct.exter_cond_cost(f"{int(sol_val)}") - ct.exter_cond_cost(f"{int(base_val)}")
    
    if fname == "Heating QC":
        return ct.heating_qc_cost(f"{int(sol_val)}") - ct.heating_qc_cost(f"{int(base_val)}")
    
    if fname == "Fireplace Qu":
        return ct.fireplace_cost(f"{int(sol_val)}") - ct.fireplace_cost(f"{int(base_val)}")
    
    if fname == "Bsmt Cond":
        return ct.bsmt_cond_cost(f"{int(sol_val)}") - ct.bsmt_cond_cost(f"{int(base_val)}")
    
    if fname == "Garage Qual":
        return ct.garage_qc_cost(f"{int(sol_val)}") - ct.garage_qc_cost(f"{int(base_val)}")
    
    if fname == "Garage Cond":
        return ct.garage_qc_cost(f"{int(sol_val)}") - ct.garage_qc_cost(f"{int(base_val)}")
    
    if fname == "Pool QC":
        pool_quals = {-1: "No aplica", 0: "Fa", 1: "TA", 2: "Gd", 3: "Ex"}
        ql_base = pool_quals.get(int(base_val), "No aplica")
        ql_sol = pool_quals.get(int(sol_val), "No aplica")
        return ct.poolqc_costs.get(ql_sol, 0) - ct.poolqc_costs.get(ql_base, 0)
    
    # Utilities
    if "Utilities" in fname:
        util_map = {0: "ELO", 1: "NoSeWa", 2: "NoSewr", 3: "AllPub"}
        u_base = util_map.get(int(base_val), "AllPub")
        u_sol = util_map.get(int(sol_val), "AllPub")
        return ct.util_cost(u_sol) - ct.util_cost(u_base)
    
    # Binarias one-hot (categorías)
    if 'is_' in fname:
        # Masonry veneer
        if 'mvt_is_' in fname:
            matl = fname.replace('mvt_is_', '')
            if sol_val > 0.5:
                mas_vnr_area = float(base_row.get("Mas Vnr Area", 0))
                return mas_vnr_area * ct.mas_vnr_cost(matl)
        
        # Roof material
        if 'roof_matl_is_' in fname:
            matl = fname.replace('roof_matl_is_', '')
            if sol_val > 0.5:
                return ct.get_roof_matl_cost(matl) + ct.roof_demo_cost
        
        # Electrical
        if 'elect_is_' in fname:
            elec = fname.replace('elect_is_', '')
            if sol_val > 0.5:
                return ct.electrical_cost(elec)
        
        # Garage finish
        if 'garage_finish_is_' in fname:
            finish = fname.replace('garage_finish_is_', '')
            if sol_val > 0.5:
                return ct.garage_finish_cost(finish)
        
        # Pool QC
        if 'poolqc_is_' in fname:
            qc = fname.replace('poolqc_is_', '')
            if sol_val > 0.5:
                return ct.poolqc_costs.get(qc, 0)
        
        # Paved drive
        if 'paved_drive_is_' in fname:
            drive = fname.replace('paved_drive_is_', '')
            if sol_val > 0.5:
                return ct.paved_drive_cost(drive)
        
        # Fence
        if 'fence_is_' in fname:
            fence = fname.replace('fence_is_', '')
            if sol_val > 0.5:
                return ct.fence_category_cost(fence)
        
        # Exterior material
        if 'ex1_is_' in fname or 'ex2_is_' in fname:
            matl = fname.replace('ex1_is_', '').replace('ex2_is_', '')
            if sol_val > 0.5:
                return ct.ext_mat_cost(matl)
        
        # Heating type
        if 'Heating_' in fname:
            heat = fname.replace('Heating_', '')
            if sol_val > 0.5:
                return ct.heating_type_cost(heat)
        
        # Basement type
        if 'b1_is_' in fname or 'b2_is_' in fname:
            btype = fname.replace('b1_is_', '').replace('b2_is_', '')
            if sol_val > 0.5:
                return ct.bsmt_type_cost(btype)
    
    # Otros flags especiales
    if fname == "central_air_yes" and sol_val > 0.5:
        return ct.central_air_install
    
    # Si no se puede estimar, retornar None
    return None

if __name__ == "__main__":
    pid = 526351010
    budget = 500000
    
    result = extract_all_changes(pid, budget, time_limit=90)
    
    if result:
        print(f"\n[SUCCESS] Cambios extraídos correctamente")
    else:
        print(f"\n[FAILED] No se pudo extraer información")
