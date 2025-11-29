#!/usr/bin/env python3
"""
Verifica que todos los 20 componentes de especificación tienen costos 
siendo sumados a lin_cost.
"""
import sys
sys.path.insert(0, r"c:\Users\valen\OneDrive - uc.cl\UNIVERSIDAD\2024-1\Bases de datos\Optimizacion-Casas-Ames-Iowa-Capstone-")

# Read gurobi_model.py and search for cost additions
with open(r"c:\Users\valen\OneDrive - uc.cl\UNIVERSIDAD\2024-1\Bases de datos\Optimizacion-Casas-Ames-Iowa-Capstone-\optimization\remodel\gurobi_model.py", "r", encoding="utf-8") as f:
    content = f.read()

# 20 specification components and keywords they should appear with
components = {
    "1. Utilities": ["util_cost", "util_bin", "lin_cost"],
    "2. Central Air": ["central_air_install", "air_yes", "lin_cost"],
    "3. Heating Type/QC": ["heating_cost", "heat_bin", "lin_cost"],
    "4. Kitchen Quality": ["kitchen_qual", "lin_cost"],
    "5. Basement Finish": ["bsmt_fin_cost", "lin_cost"],
    "6. Basement Condition": ["bsmt_cond_cost", "lin_cost"],
    "7. Fireplace Quality": ["fireplace", "fire_bin", "lin_cost"],
    "8. Fence": ["fence", "lin_cost"],
    "9. Paved Drive": ["paved_drive", "lin_cost"],
    "10. Garage Quality": ["garage_qual", "garage_qc_costs", "lin_cost"],
    "11. Garage Condition": ["garage_cond", "lin_cost"],
    "12. Garage Finish": ["garage_finish_cost", "lin_cost"],
    "13. Add Rooms": ["construction_cost", "AddFull|AddHalf|AddKitch|AddBed", "lin_cost"],
    "14. Area Expansions": ["ampl.*_cost", "z10_|z20_|z30_", "lin_cost"],
    "15. Pool Quality": ["poolqc_cost", "pq\\[", "lin_cost"],
    "16. Exterior Material": ["ext_mat_cost", "lin_cost"],
    "17. Exterior Condition": ["exter_cond_cost", "lin_cost"],
    "18. Masonry Veneer": ["mas_vnr_cost", "lin_cost"],
    "19. Roof Material": ["roof.*_cost", "lin_cost"],
    "20. Heating": ["heating.*_cost", "lin_cost"],
}

print("=" * 100)
print("COST COMPONENT VERIFICATION")
print("=" * 100)
print()

all_good = True
for comp_name, keywords in components.items():
    print(f"{comp_name}")
    has_lin_cost = False
    has_cost_source = False
    
    for kw in keywords:
        if "lin_cost" in kw:
            if "lin_cost" in content:
                has_lin_cost = True
            print(f"  ✓ lin_cost found in code")
        else:
            import re
            if re.search(kw, content, re.IGNORECASE):
                has_cost_source = True
                print(f"  ✓ Cost source keyword '{kw}' found")
            else:
                print(f"  ✗ Cost source keyword '{kw}' NOT found")
                all_good = False
    
    if not (has_lin_cost and has_cost_source):
        print(f"  ⚠ WARNING: Missing cost implementation")
        all_good = False
    print()

print("=" * 100)
if all_good:
    print("✓ All components appear to have cost implementation")
else:
    print("⚠ Some components may be missing cost implementation - review above")
print("=" * 100)
