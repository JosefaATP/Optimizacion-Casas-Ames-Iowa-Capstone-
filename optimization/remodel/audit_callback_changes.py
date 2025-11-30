#!/usr/bin/env python3
"""
Auditoría del callback: compara cambios ANTES y DESPUÉS de la corrección de y_log.
Detecta si se están eligiendo DIFERENTES cambios en cada pasada.
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

def get_solution_snapshot(m, base_X):
    """Captura snapshot de la solución actual"""
    X_sol = rebuild_embed_input_df(m, base_X)
    
    changes = {}
    for spec in MODIFIABLE:
        fname = spec.name
        if fname not in X_sol.index or fname not in base_X.index:
            continue
        
        base_val = float(base_X.loc[fname, 0])
        sol_val = float(X_sol.loc[fname, 0])
        
        if abs(base_val - sol_val) > 1e-6:
            changes[fname] = {
                'base': base_val,
                'solution': sol_val,
                'delta': sol_val - base_val
            }
    
    return changes, m.ObjVal

def run_optimization_with_snapshots(pid, budget, time_limit=90):
    """
    Ejecuta optimización y captura snapshots ANTES y DESPUÉS del callback.
    """
    
    print(f"\n{'='*100}")
    print(f"ANALISIS DE CALLBACK: Cambios antes/despues corrección y_log")
    print(f"{'='*100}\n")
    
    # Setup
    base = get_base_house(pid)
    base_row = base.row if hasattr(base, 'row') else base
    ct = CostTables()
    bundle = XGBBundle()
    
    X_base = build_base_input_row(bundle, base_row)
    precio_base = float(bundle.predict(X_base).iloc[0])
    
    # Build MIP
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
    
    # PRIMERA PASADA: Sin callback, solo optimizar
    print("[PASADA-1] Optimizando SIN corrección de y_log...")
    m.optimize()
    
    if m.status != gp.GRB.OPTIMAL:
        print(f"[ERROR] No óptima (status={m.status})")
        return None
    
    changes_before, obj_before = get_solution_snapshot(m, m._X_base_numeric)
    
    print(f"[PASADA-1] Objetivo: ${obj_before:,.2f}")
    print(f"[PASADA-1] Cambios: {len(changes_before)} features modificados")
    
    # Calcular costo de esta solución
    total_cost_var = m.getVarByName("cost_model")
    if total_cost_var is not None:
        cost_before = float(total_cost_var.X)
    else:
        cost_before = 0  # estimado
    
    print(f"[PASADA-1] Costo total: ${cost_before:,.2f}")
    
    # Mostrar cambios antes
    if changes_before:
        print(f"\n[CAMBIOS ANTES]:")
        for fname, info in sorted(changes_before.items())[:10]:
            print(f"  {fname:<40s}: {info['base']:>10.4f} -> {info['solution']:>10.4f}")
        if len(changes_before) > 10:
            print(f"  ... y {len(changes_before) - 10} mas")
    
    # SEGUNDA PASADA: Con factor de corrección
    print(f"\n[PASADA-2] Aplicando factor de corrección...")
    
    # Calcular y_log error
    X_sol_1 = rebuild_embed_input_df(m, m._X_base_numeric)
    ylog_mip = float(m._y_log_var.X)
    ylog_real = float(bundle.predict_log_raw(X_sol_1).iloc[0])
    error = abs(ylog_mip - ylog_real)
    reality_factor = np.exp(-error)
    
    print(f"[PASADA-2] y_log error: {error:.6f}")
    print(f"[PASADA-2] Factor de corrección: {reality_factor:.6f}")
    
    # Modificar objetivo y re-optimizar
    obj_expr = m.getObjective()
    if obj_expr is not None:
        m.setObjective(obj_expr * reality_factor, sense=gp.GRB.MAXIMIZE)
        print(f"[PASADA-2] Re-optimizando con factor...")
        m.optimize()
        
        if m.status == gp.GRB.OPTIMAL:
            changes_after, obj_after = get_solution_snapshot(m, m._X_base_numeric)
            
            print(f"[PASADA-2] Objetivo corregido: ${obj_after:,.2f}")
            print(f"[PASADA-2] Cambios: {len(changes_after)} features modificados")
            
            # Calcular costo después
            total_cost_var = m.getVarByName("cost_model")
            if total_cost_var is not None:
                cost_after = float(total_cost_var.X)
            else:
                cost_after = 0
            
            print(f"[PASADA-2] Costo total: ${cost_after:,.2f}")
        else:
            print(f"[ERROR] Segunda pasada no óptima (status={m.status})")
            return None
    else:
        print(f"[ERROR] No se pudo obtener objetivo")
        return None
    
    # COMPARACION
    print(f"\n{'='*100}")
    print(f"COMPARACION ANTES/DESPUES")
    print(f"{'='*100}\n")
    
    print(f"Objetivo:       ${obj_before:,.2f}  ->  ${obj_after:,.2f}  (Δ=${obj_after - obj_before:,.2f})")
    print(f"Costo total:    ${cost_before:,.2f}  ->  ${cost_after:,.2f}  (Δ=${cost_after - cost_before:,.2f})")
    print(f"# Cambios:      {len(changes_before):>3d}  ->  {len(changes_after):>3d}\n")
    
    # Analizar qué cambió
    features_before = set(changes_before.keys())
    features_after = set(changes_after.keys())
    
    added = features_after - features_before
    removed = features_before - features_after
    modified = features_before & features_after
    
    if added:
        print(f"\n[NUEVOS CAMBIOS en Pasada-2] ({len(added)}):")
        for fname in sorted(added)[:5]:
            info = changes_after[fname]
            print(f"  {fname:<40s}: {info['base']:>10.4f} -> {info['solution']:>10.4f}")
        if len(added) > 5:
            print(f"  ... y {len(added) - 5} mas")
    
    if removed:
        print(f"\n[CAMBIOS ELIMINADOS en Pasada-2] ({len(removed)}):")
        for fname in sorted(removed)[:5]:
            info = changes_before[fname]
            print(f"  {fname:<40s}: {info['base']:>10.4f} -> {info['solution']:>10.4f}")
        if len(removed) > 5:
            print(f"  ... y {len(removed) - 5} mas")
    
    if modified:
        modified_list = []
        for fname in sorted(modified):
            before = changes_before[fname]['solution']
            after = changes_after[fname]['solution']
            if abs(before - after) > 1e-6:
                modified_list.append((fname, before, after))
        
        if modified_list:
            print(f"\n[CAMBIOS MODIFICADOS en Pasada-2] ({len(modified_list)}):")
            for fname, before, after in modified_list[:5]:
                print(f"  {fname:<40s}: {before:>10.4f} -> {after:>10.4f}")
            if len(modified_list) > 5:
                print(f"  ... y {len(modified_list) - 5} mas")
    
    # Conclusion
    print(f"\n{'='*100}")
    print(f"CONCLUSION")
    print(f"{'='*100}")
    
    if len(changes_before) == len(changes_after) and not added and not removed:
        print("[OK] Se están haciendo los MISMOS cambios en ambas pasadas")
        print(f"     El costo cambio porque: {cost_before:,.2f} -> {cost_after:,.2f}")
        print(f"     Esto puede ocurrir si hay features con costo variable (ej: proporcional al area)")
    else:
        print("[ADVERTENCIA] Se están haciendo DIFERENTES cambios en ambas pasadas!")
        print(f"     Pasada-1 (sin correccion): {len(changes_before)} cambios, ${cost_before:,.2f}")
        print(f"     Pasada-2 (con correccion): {len(changes_after)} cambios, ${cost_after:,.2f}")
        print(f"     La corrección de y_log cambio las decisiones del optimizador")

if __name__ == "__main__":
    pid = 526351010
    budget = 500000
    
    run_optimization_with_snapshots(pid, budget, time_limit=90)
