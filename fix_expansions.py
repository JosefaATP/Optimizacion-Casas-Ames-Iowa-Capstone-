#!/usr/bin/env python3
"""
FIX #3: Create missing z-variables for area expansions (z10_*, z20_*, z30_*)

Root Cause: z-variables are defined in features.py but never created in gurobi_model.py
Current Status: Lines 1198-1250 try to use z[c][s] but all values are None because 
                x.get(f"z{s}_{c}") fails (variables weren't extracted)

Solution: Add code block BEFORE line 913 to explicitly create z-variables from the model

Components: GarageArea, WoodDeckSF, OpenPorchSF, EnclosedPorch, 3SsnPorch, ScreenPorch, PoolArea
Scales: 10%, 20%, 30% (binary variables: 0=no expansion, 1=apply expansion)
"""

import re

def add_z_variable_creation():
    """Add z-variable creation code to gurobi_model.py"""
    
    filepath = r"c:\Users\valen\OneDrive - uc.cl\UNIVERSIDAD\2024-1\Bases de datos\Optimizacion-Casas-Ames-Iowa-Capstone-\optimization\remodel\gurobi_model.py"
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Code to insert BEFORE the z dict construction (before line 913)
    # This creates actual Gurobi variables for the z-variables
    z_creation_code = '''
    # ======== CREATE Z-VARIABLES FOR AREA EXPANSIONS ========
    # z[c][s] = binary variable indicating s% expansion of component c
    # Must be created as gurobi.Var objects BEFORE being used in constraints
    
    for c in COMPONENTES:
        for s in [10, 20, 30]:
            var_name = f"z{s}_{c.replace(' ', '')}"
            if var_name not in x:
                x[var_name] = m.addVar(vtype=gp.GRB.BINARY, name=var_name)
    
'''
    
    # Find the line with "z = {c: {s: x.get" and insert before it
    pattern = r"(\s+)z = \{c: \{s: x\.get\(f\"z\{s\}_\{c\.replace\(' ', ''\)\}\"\)"
    
    if re.search(pattern, content):
        # Insert the z-variable creation code before the z dict construction
        new_content = re.sub(
            pattern,
            z_creation_code + r"\1z = {c: {s: x.get(f\"z{s}_{c.replace(' ', '')}\")",
            content
        )
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("[OK] Z-variables creation code added to gurobi_model.py")
        return True
    else:
        print("[ERROR] Could not find z-variable dictionary pattern in gurobi_model.py")
        return False

if __name__ == "__main__":
    success = add_z_variable_creation()
    
    if success:
        print("[OK] FIX #3 deployed successfully - Area expansions z-variables now created")
    else:
        print("[ERROR] FIX #3 deployment failed")
        exit(1)
