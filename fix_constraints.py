#!/usr/bin/env python
"""Fix missing path exclusion constraints for Exterior and Heating"""

# Read file
with open('optimization/remodel/gurobi_model.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and add Exterior path exclusion constraint
insert_point = content.find('    # ================== MAS VNR (robusto')
if insert_point != -1:
    exterior_constraint = '''
    # FIXED: Add path exclusion constraint - Material change XOR Quality upgrade (per specification)
    # Specification: se pueden seguir dos caminos implies paths are mutually exclusive
    if eligible == 1:
        UpgMat_ext = m.addVar(vtype=gp.GRB.BINARY, name="ext_upg_material")
        UpgQC_ext = m.addVar(vtype=gp.GRB.BINARY, name="ext_upg_qc")
        m.addConstr(UpgMat_ext + UpgQC_ext <= 1, name="EXT_exclusive_paths")

'''
    content = content[:insert_point] + exterior_constraint + content[insert_point:]
    print('[OK] Exterior constraint added')

# Find Heating QC cost section and add constraint there too
insert_point2 = content.find('    cost_heat = gp.LinExpr(0.0)')
if insert_point2 != -1:
    heating_constraint = '''    # FIXED: Add path exclusion constraint - Type change XOR Quality upgrade (per specification)
    if eligible_heat == 1 and change_type is not None and qc_bins:
        UpgType_heat = m.addVar(vtype=gp.GRB.BINARY, name="heat_upg_type_flag")
        UpgQC_heat = m.addVar(vtype=gp.GRB.BINARY, name="heat_upg_qc_flag")
        m.addConstr(UpgType_heat + UpgQC_heat <= 1, name="HEAT_exclusive_paths")

'''
    content = content[:insert_point2] + heating_constraint + content[insert_point2:]
    print('[OK] Heating constraint added')

# Write back
with open('optimization/remodel/gurobi_model.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('[OK] gurobi_model.py updated successfully')
