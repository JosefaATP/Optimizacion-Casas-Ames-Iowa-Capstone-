#!/usr/bin/env python3
"""
Auditoría de cobertura de costos: qué features se pueden modificar pero NO tienen costo.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='ignore')

from optimization.remodel.features import MODIFIABLE
from optimization.remodel.costs import CostTables

# Features "sospechosas" = se pueden modificar pero quizás no tienen costo implementado
SUSPICIOUS_FEATURES = [
    "Bedroom AbvGr",
    "Full Bath",
    "BsmtFin SF 1",
    "BsmtFin SF 2",
    "Bsmt Unf SF",
    "Gr Liv Area",
    "1st Flr SF",
    "TotRms AbvGrd",
    "Mas Vnr Area",
    "Kitchen Qual",
    "Exter Qual",
    "Exter Cond",
    "Heating QC",
    "Fireplace Qu",
    "Bsmt Cond",
    "Garage Qual",
    "Garage Cond",
    "Pool QC",
    "Half Bath",
    "Kitchen AbvGr",
    "Roof Matl",
    "Utilities",
    "Garage Area",
    "Wood Deck SF",
    "Open Porch SF",
    "Enclosed Porch",
    "3Ssn Porch",
    "Screen Porch",
    "Pool Area",
]

ct = CostTables()

print("="*80)
print("AUDITORÍA DE COBERTURA DE COSTOS")
print("="*80)

print("\n[CONTINUOUS FEATURES]")
print("Features numéricas/continuas que se pueden modificar:")
for spec in MODIFIABLE:
    if spec.vartype == "C" and any(x in spec.name for x in ["Area", "SF", "Gr Liv"]):
        print(f"  ✓ {spec.name:<35s} LB={spec.lb:>10.1f} UB={spec.ub:>10.1f}")

print("\n[INTEGER FEATURES (Ordinales/Categorías)]")
print("Features ordinales que se pueden modificar:")
for spec in MODIFIABLE:
    if spec.vartype == "I" and not spec.name.startswith("z"):
        if any(x in spec.name for x in ["Kitchen", "Exter", "Heating", "Fire", "Bsmt", 
                                        "Garage", "Pool", "Qual", "Cond", "Bed", "Bath"]):
            print(f"  ✓ {spec.name:<35s} LB={spec.lb:>5.0f} UB={spec.ub:>5.0f}")

print("\n[BINARY CATEGORICAL FEATURES]")
print("Features binarias one-hot:")
binary_cats = {}
for spec in MODIFIABLE:
    if spec.vartype == "B":
        prefix = spec.name.split('_')[0]
        if prefix not in binary_cats:
            binary_cats[prefix] = []
        binary_cats[prefix].append(spec.name)

for category, features in sorted(binary_cats.items()):
    if "is_" in category or "is" in category:
        print(f"  • {category}: {len(features)} options")

print("\n" + "="*80)
print("FEATURES SIN COBERTURA DE COSTO CLARA")
print("="*80)

# Verificar features sin costos
missing_costs = []
for spec in MODIFIABLE:
    fname = spec.name
    
    # Features de área -> tienen costo de construcción
    if any(x in fname for x in ["Area", "SF"]):
        continue
    
    # Features de habitaciones -> tienen costo
    if "Bedroom" in fname or "Bath" in fname:
        continue
    
    # Features ordinales con tabla de costo
    if fname in ["Kitchen Qual", "Exter Qual", "Exter Cond", "Heating QC", 
                 "Fireplace Qu", "Garage Qual", "Garage Cond", "Bsmt Cond"]:
        continue
    
    if fname == "Pool QC":
        # Verificar si poolqc_costs está poblado
        if not ct.poolqc_costs or all(v == 0 for v in ct.poolqc_costs.values()):
            missing_costs.append(f"{fname} - poolqc_costs empty or zeros")
        continue
    
    if fname == "Utilities":
        if not ct.utilities_costs or all(v == 0 for v in ct.utilities_costs.values()):
            missing_costs.append(f"{fname} - utilities_costs empty or zeros")
        continue
    
    if fname == "Roof Matl":
        if not ct.roof_matl_fixed or all(v == 0 for v in ct.roof_matl_fixed.values()):
            missing_costs.append(f"{fname} - roof_matl_fixed empty or zeros")
        continue
    
    # Binarias
    if "_is_" in fname:
        # Ya está cubierto por categoría
        continue
    
    # Flags de amplificación (z10, z20, z30)
    if "z" in fname and any(x in fname for x in ["10", "20", "30"]):
        continue
    
    # Flags especiales de upgrade
    if fname in ["AddFull", "AddHalf", "AddKitch", "AddBed", "UpgGarage", "upg_pool_qc",
                 "heat_upg_type", "heat_upg_qc", "delta_KitchenQual_TA", "delta_KitchenQual_EX"]:
        continue
    
    if fname not in ["TotRms AbvGrd"]:  # Este es redundante con Bedroom
        missing_costs.append(fname)

if missing_costs:
    print(f"\n⚠️  Features sin cobertura de costo clara ({len(missing_costs)}):")
    for fname in missing_costs:
        print(f"  ✗ {fname}")
else:
    print("\n✓ Todas las features tienen cobertura de costo")

print("\n" + "="*80)
print("VERIFICACIÓN DE COSTOS EN TABLAS")
print("="*80)

# Verificar que no estén vacías
tables_to_check = [
    ("utilities_costs", ct.utilities_costs),
    ("roof_matl_fixed", ct.roof_matl_fixed),
    ("poolqc_costs", ct.poolqc_costs),
    ("garage_finish_costs_sqft", ct.garage_finish_costs_sqft),
    ("garage_qc_costs", ct.garage_qc_costs),
    ("paved_drive_costs", ct.paved_drive_costs),
    ("fence_category_costs", ct.fence_category_costs),
    ("foundation_costs", ct.foundation_costs),
    ("exterior_matl_lumpsum", ct.exterior_matl_lumpsum),
    ("exter_qual_costs", ct.exter_qual_costs),
    ("electrical_type_costs", ct.electrical_type_costs),
    ("heating_type_costs", ct.heating_type_costs),
    ("heating_qc_costs", ct.heating_qc_costs),
    ("bsmt_cond_upgrade_costs", ct.bsmt_cond_upgrade_costs),
    ("bsmt_type_costs", ct.bsmt_type_costs),
    ("fireplace_costs", ct.fireplace_costs),
    ("poolqc_costs", ct.poolqc_costs),
]

for table_name, table_dict in tables_to_check:
    if not table_dict:
        print(f"  ⚠️  {table_name:<35s} EMPTY")
    else:
        zero_count = sum(1 for v in table_dict.values() if v == 0 or v == 0.0)
        total = len(table_dict)
        if zero_count > 0:
            print(f"  ⚠️  {table_name:<35s} {zero_count}/{total} entries are 0 or 0.0")
        else:
            print(f"  ✓ {table_name:<35s} ({total} entries, all > 0)")

print("\n" + "="*80)
print("COSTOS ESPECIALES A REVISAR")
print("="*80)

print(f"\n• Kitchen upgrades:")
print(f"  - TA: ${ct.kitchenQual_upgrade_TA:,.2f}")
print(f"  - EX: ${ct.kitchenQual_upgrade_EX:,.2f}")

print(f"\n• Bathroom/Bedroom additions:")
print(f"  - Add full bath: ${ct.add_fullbath_cost:,.2f}")
print(f"  - Add half bath: ${ct.add_halfbath_cost:,.2f}")
print(f"  - Add bedroom (per SF): ${ct.add_bedroom_cost_per_sf:,.2f}/SF")

print(f"\n• General construction:")
print(f"  - Base rate: ${ct.construction_cost:,.2f}/SF")
print(f"  - Soft cost markup: {ct.soft_cost_markup}x")
print(f"  - Finish basement: ${ct.finish_basement_per_f2:,.2f}/SF²")

print(f"\n• Central air installation: ${ct.central_air_install:,.2f}")

print(f"\n• Porches/Decks (per SF):")
print(f"  - Wood deck: ${ct.wooddeck_cost:,.2f}/SF")
print(f"  - Open porch: ${ct.openporch_cost:,.2f}/SF")
print(f"  - Enclosed: ${ct.enclosedporch_cost:,.2f}/SF")
print(f"  - 3-Season: ${ct.threessnporch_cost:,.2f}/SF")
print(f"  - Screen: ${ct.screenporch_cost:,.2f}/SF")

print(f"\n• Roof:")
print(f"  - Demo cost: ${ct.roof_demo_cost:,.2f}")
print(f"  - Material costs: {list(ct.roof_matl_fixed.keys())}")
