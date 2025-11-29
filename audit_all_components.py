#!/usr/bin/env python3
"""
COMPREHENSIVE AUDIT: All 20 specification components vs implementation

Goal: Verify each constraint has proper cost modeling in gurobi_model.py
"""

import re

# Read the specification document
spec_file = r"Info proyecto\Informe_2 latex\chapters\appendix3.tex"
code_file = r"optimization\remodel\gurobi_model.py"

with open(spec_file, 'r', encoding='utf-8') as f:
    spec_content = f.read()

with open(code_file, 'r', encoding='utf-8') as f:
    code_content = f.read()

# Define all 20 components from the specification
components = {
    1: {
        "name": "Utilities",
        "section": "170-200",
        "spec_rule": "Can upgrade to higher cost utilities only",
        "cost_formula": "C_u * (UpgType_i - ChangeType_i) or C_new_u for type change",
        "keywords": ["Utilities", "UTIL"],
        "has_cost": True
    },
    2: {
        "name": "Central Air",
        "section": "~220",
        "spec_rule": "Add CentralAir if missing (No → Yes)",
        "cost_formula": "C_CentralAir * CentralAir_Yes",
        "keywords": ["Central Air", "CENTRAL"],
        "has_cost": True
    },
    3: {
        "name": "Heating Type",
        "section": "~240-290",
        "spec_rule": "Change heating type (GasA/GasW/Wall/Floor/etc) with path restriction + QC upgrade",
        "cost_formula": "C_heat_type + C_heating_qc if QC upgrade",
        "keywords": ["Heating", "HEAT", "heat_is_"],
        "has_cost": True
    },
    4: {
        "name": "Kitchen Quality",
        "section": "~310",
        "spec_rule": "Upgrade Kitchen Quality (TA→Gd→Ex, no downgrade)",
        "cost_formula": "C_k for quality level k",
        "keywords": ["Kitchen Qual", "KIT"],
        "has_cost": True
    },
    5: {
        "name": "Basement Finish",
        "section": "~330-380",
        "spec_rule": "Finish basement (unfinished→Type1 or Type2)",
        "cost_formula": "C_Bsmt * (x_1 + x_2) + C_BsmtType for finish type",
        "keywords": ["Bsmt", "basement", "BSMT"],
        "has_cost": True
    },
    6: {
        "name": "Basement Condition",
        "section": "~400",
        "spec_rule": "Upgrade basement condition (Po→Fa→TA→Gd→Ex)",
        "cost_formula": "C_b for condition b",
        "keywords": ["Bsmt Cond", "BC"],
        "has_cost": True
    },
    7: {
        "name": "Basement Finish Type",
        "section": "~420",
        "spec_rule": "Change basement finish type (ALQ/BLQ/GLQ/etc)",
        "cost_formula": "C_BsmtType * M_BsmtType",
        "keywords": ["BsmtFin Type", "BsmtFin"],
        "has_cost": True
    },
    8: {
        "name": "Fireplace Quality",
        "section": "~960-1010",
        "spec_rule": "Po→{Po,Fa}, TA→{TA,Gd,Ex}, Fa/Gd/Ex fixed",
        "cost_formula": "C_fen for fireplace level",
        "keywords": ["Fireplace", "fireplace", "FQ"],
        "has_cost": True
    },
    9: {
        "name": "Fence",
        "section": "~1020-1100",
        "spec_rule": "Build fence if NA, upgrade quality if exists (GdWo/MnWw→MnPrv/GdPrv)",
        "cost_formula": "C_Fence * LotFrontage or C_fence_level",
        "keywords": ["Fence", "FENCE"],
        "has_cost": True
    },
    10: {
        "name": "Paved Drive",
        "section": "~1110-1150",
        "spec_rule": "Add paved drive if missing, upgrade quality (N→P→Y)",
        "cost_formula": "C_d for drive type d",
        "keywords": ["Paved Drive", "DRIVE"],
        "has_cost": True
    },
    11: {
        "name": "Garage Quality",
        "section": "~1160-1250",
        "spec_rule": "Upgrade garage quality (Po→Fa→TA→Gd→Ex)",
        "cost_formula": "C_g for garage quality g",
        "keywords": ["Garage Qual", "GQ"],
        "has_cost": True
    },
    12: {
        "name": "Garage Condition",
        "section": "~1260-1320",
        "spec_rule": "Upgrade garage condition (Po→Fa→TA→Gd→Ex)",
        "cost_formula": "C_g for garage condition g",
        "keywords": ["Garage Cond", "GC"],
        "has_cost": True
    },
    13: {
        "name": "Garage Finish",
        "section": "~1550-1620",
        "spec_rule": "Finish unfinished garage (Unf→Fin)",
        "cost_formula": "C_GFin * area",
        "keywords": ["Garage Finish", "GAFIN"],
        "has_cost": True
    },
    14: {
        "name": "Add Rooms",
        "section": "~1640-1680",
        "spec_rule": "Add Full/Half bath, Kitchen, Bedrooms with fixed area costs",
        "cost_formula": "C_construction * (A_Full*AddFull + A_Half*AddHalf + ...)",
        "keywords": ["AddFull", "AddHalf", "AddKitch", "AddBed"],
        "has_cost": True
    },
    15: {
        "name": "Area Expansions",
        "section": "~1200-1270",
        "spec_rule": "Expand components (GarageArea, Porches, Pool, WoodDeck) by 10%/20%/30%",
        "cost_formula": "C_10/C_20/C_30 * delta_i,s * z_i,s",
        "keywords": ["z10_", "z20_", "z30_", "AMPL"],
        "has_cost": True
    },
    16: {
        "name": "Pool Quality",
        "section": "~1700-1750",
        "spec_rule": "Build pool if NA, upgrade quality (Po→Fa→TA→Gd→Ex)",
        "cost_formula": "C_p for pool quality p",
        "keywords": ["Pool QC", "Pool"],
        "has_cost": True
    },
    17: {
        "name": "Exterior Material",
        "section": "~415-539",
        "spec_rule": "Change exterior material (1st/2nd) OR upgrade quality, not both (XOR)",
        "cost_formula": "C_material for material type",
        "keywords": ["Exterior 1st", "Exterior 2nd", "EXT"],
        "has_cost": True
    },
    18: {
        "name": "Exterior Condition",
        "section": "~550-600",
        "spec_rule": "Upgrade exterior condition (Po→Fa→TA→Gd→Ex)",
        "cost_formula": "C_cond for condition",
        "keywords": ["Exter Cond", "Exter Qual"],
        "has_cost": True
    },
    19: {
        "name": "Masonry Veneer",
        "section": "~610-700",
        "spec_rule": "Change mason veneer type (BrkCmn/BrkFace/CBlock/Stone/No aplica)",
        "cost_formula": "C_mvt for veneer type",
        "keywords": ["Mas Vnr", "MVT"],
        "has_cost": True
    },
    20: {
        "name": "Roof Material",
        "section": "~710-800",
        "spec_rule": "Change roof material type",
        "cost_formula": "C_roof for roof type",
        "keywords": ["Roof Matl", "ROOF"],
        "has_cost": True
    },
}

print("=" * 90)
print("COMPREHENSIVE SPECIFICATION AUDIT: 20 Components Verification")
print("=" * 90)

issues = []

for comp_id, comp in components.items():
    print(f"\n[{comp_id:2d}] {comp['name']:25s} | Cost? {comp['has_cost']}")
    print(f"     Keywords: {', '.join(comp['keywords'])}")
    
    # Check if keywords appear in code
    found_in_code = False
    cost_mention = False
    
    for keyword in comp['keywords']:
        if keyword in code_content:
            found_in_code = True
        if keyword in code_content and ('lin_cost' in code_content[code_content.find(keyword)-500:code_content.find(keyword)+1000] or 
                                        'cost' in code_content[code_content.find(keyword)-500:code_content.find(keyword)+1000].lower()):
            cost_mention = True
    
    if found_in_code:
        print(f"     ✓ Found in code")
        if cost_mention:
            print(f"     ✓ Cost modeling detected")
        else:
            print(f"     ⚠ WARNING: Cost NOT explicitly mentioned near keywords")
            issues.append(f"{comp['name']}: Cost may be missing")
    else:
        print(f"     ✗ NOT found in code")
        issues.append(f"{comp['name']}: Component not implemented")

print("\n" + "=" * 90)
print("ISSUES FOUND:")
print("=" * 90)

if issues:
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")
else:
    print("No obvious issues found - all components referenced")

print("\n" + "=" * 90)
print("NEXT: Manual review of each cost formula in gurobi_model.py")
print("=" * 90)
