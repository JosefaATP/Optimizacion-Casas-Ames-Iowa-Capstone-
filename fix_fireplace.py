#!/usr/bin/env python3
"""
FIX #4: Fireplace Quality Upgrade Rules - Path Restriction

Problem: Current code allows level-skipping (e.g., Po → Gd) which violates spec
Spec requires:
  - Po base → only {Po, Fa}
  - TA base → only {TA, Gd, Ex}
  - Fa/Gd/Ex base → fixed (no change)
  - NA base → NA only

Solution: Replace the simple "no downgrade" constraint with proper path restriction
"""

def fix_fireplace_paths():
    """Add path restrictions to Fireplace Quality constraints"""
    
    filepath = r"c:\Users\valen\OneDrive - uc.cl\UNIVERSIDAD\2024-1\Bases de datos\Optimizacion-Casas-Ames-Iowa-Capstone-\optimization\remodel\gurobi_model.py"
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the Fireplace section and replace the upgrade restriction logic
    # Current logic (lines 1350-1355):
    old_logic = '''        # NO permitir bajar calidad vs base (solo igual o subir)
        for nm in FQ_CATS:
            if FQ_ORD[nm] < base_ord:
                fq[nm].UB = 0.0'''
    
    new_logic = '''        # IMPLEMENT PATH RESTRICTIONS per specification
        # Po → {Po, Fa}
        # TA → {TA, Gd, Ex}
        # Fa/Gd/Ex → no change (fixed)
        # NA → NA (already handled above)
        
        allowed_paths = {
            -1: ["No aplica"],      # NA → NA
            0:  ["Po", "Fa"],       # Po → {Po, Fa}
            1:  ["Fa"],             # Fa → Fa (fixed)
            2:  ["TA", "Gd", "Ex"], # TA → {TA, Gd, Ex}
            3:  ["Gd"],             # Gd → Gd (fixed)
            4:  ["Ex"],             # Ex → Ex (fixed)
        }
        
        allowed = allowed_paths.get(base_ord, [base_fq_txt])
        for nm in FQ_CATS:
            if nm not in allowed:
                fq[nm].UB = 0.0'''
    
    if old_logic in content:
        content = content.replace(old_logic, new_logic)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("[OK] Fireplace path restrictions added")
        return True
    else:
        print("[ERROR] Could not find Fireplace upgrade logic in gurobi_model.py")
        print("[DEBUG] Looking for alternative match...")
        
        # Try a more flexible search
        if "NO permitir bajar calidad vs base" in content:
            print("[DEBUG] Found the comment, but exact match failed")
            print("[DEBUG] Check line 1350-1355 manually")
            return False
        
        return False

if __name__ == "__main__":
    success = fix_fireplace_paths()
    
    if success:
        print("[OK] FIX #4 deployed - Fireplace path restrictions now enforced")
    else:
        print("[ERROR] FIX #4 deployment failed")
        exit(1)
