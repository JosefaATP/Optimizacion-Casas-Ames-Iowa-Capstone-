#!/usr/bin/env python3
"""
FIX #5: Exterior Material Cost - Make incremental, not absolute

Problem: Code sums absolute material costs (lines 471-474) without subtracting base material cost
Current: lin_cost += ct.ext_mat_cost(nm) * vb  (sums all chosen materials)
Correct: lin_cost += (ct.ext_mat_cost(nm) - ct.ext_mat_cost(base_nm)) * vb  (incremental only)

This explains why changing VinylSd→Plywood appears "cost-free": 
- VinylSd: $17,410
- Plywood: $3,461.81
- Delta: -$13,948 (should be a credit, not appear as no cost)
"""

import re

def fix_exterior_material_cost():
    """Fix Exterior Material cost to be incremental"""
    
    filepath = r"c:\Users\valen\OneDrive - uc.cl\UNIVERSIDAD\2024-1\Bases de datos\Optimizacion-Casas-Ames-Iowa-Capstone-\optimization\remodel\gurobi_model.py"
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the absolute cost calculation with incremental
    old_code = '''    # COSTO ABSOLUTO para materiales (no incremental)
    for nm, vb in ex1.items():
        lin_cost += ct.ext_mat_cost(nm) * vb
    if Ilas2 == 1:
        for nm, vb in ex2.items():
            lin_cost += ct.ext_mat_cost(nm) * vb'''
    
    # Get base material costs
    new_code = '''    # INCREMENTAL cost for material change (subtract base material cost)
    # If changing from base material to new material, only pay the difference
    ex1_base_cost = ct.ext_mat_cost(ex1_base_name) if ex1_base_name in ex1 else 0.0
    for nm, vb in ex1.items():
        cost_delta = ct.ext_mat_cost(nm) - ex1_base_cost
        lin_cost += cost_delta * vb
    
    if Ilas2 == 1:
        ex2_base_cost = ct.ext_mat_cost(ex2_base_name) if ex2_base_name in ex2 else 0.0
        for nm, vb in ex2.items():
            cost_delta = ct.ext_mat_cost(nm) - ex2_base_cost
            lin_cost += cost_delta * vb'''
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("[OK] Exterior Material cost changed to incremental")
        return True
    else:
        print("[ERROR] Could not find exact match for exterior material cost code")
        return False

if __name__ == "__main__":
    success = fix_exterior_material_cost()
    
    if success:
        print("[OK] FIX #5 deployed - Exterior Material cost now incremental")
    else:
        print("[ERROR] FIX #5 deployment failed")
        exit(1)
