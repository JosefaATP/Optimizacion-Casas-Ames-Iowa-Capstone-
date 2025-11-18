#!/usr/bin/env python3
# optimization/remodel/test_quality_calc.py
"""
Script de prueba para el calculador de calidad.
Verifica que la fórmula funciona correctamente.
"""

import pandas as pd
from quality_calculator import QualityCalculator

# Crear datos de prueba
base_row = pd.Series({
    "Overall Qual": 5,  # TA - Typical
    "Kitchen Qual": 2,  # TA
    "Exter Qual": 2,    # TA
    "Exter Cond": 2,    # TA
    "Heating QC": 3,    # Gd
    "Fireplace Qu": -1, # No aplica
    "Bsmt Cond": 2,     # TA
    "Garage Qual": 2,   # TA
    "Garage Cond": 2,   # TA
    "Pool QC": -1,      # No aplica
})

# Fila óptima con mejoras
opt_row = pd.Series({
    "Overall Qual": 5,  # Será calculado
    "Kitchen Qual": 3,  # Mejorado a Gd (Good)
    "Exter Qual": 4,    # Mejorado a Ex (Excellent)
    "Exter Cond": 2,    # Sin cambio
    "Heating QC": 3,    # Sin cambio
    "Fireplace Qu": -1, # No aplica
    "Bsmt Cond": 3,     # Mejorado a Gd
    "Garage Qual": 3,   # Mejorado a Gd
    "Garage Cond": 2,   # Sin cambio
    "Pool QC": -1,      # No aplica
})

print("=" * 70)
print("TEST: Cálculo de mejora de Overall Qual")
print("=" * 70)

calc = QualityCalculator(max_boost=2.0)
result = calc.calculate_boost(base_row, opt_row)

print("\nDATA DE ENTRADA:")
print(f"  Base Overall Qual: {result['overall_base']}")
print(f"  Cambios esperados: Kitchen (TA→Gd), Exterior (TA→Ex), Basement (TA→Gd), Garage (TA→Gd)")

print("\nDETALLE DE CÁLCULO:")
print(f"  weighted_sum = {result['weighted_sum']:.4f}")
print(f"  max_boost = {calc.max_boost}")
print(f"  boost = {result['weighted_sum']:.4f} × {calc.max_boost} = {result['boost']:.4f}")

print("\nRESULTADO:")
print(f"  Overall Qual Nuevo: {result['overall_base']:.1f} + {result['boost']:.2f} = {result['overall_new']:.2f}")
print(f"  % de mejora: {result['boost_pct']:.1f}%")

print("\n" + calc.format_changes_report(result))

print("\n" + "=" * 70)
print("RESUMEN:")
print("=" * 70)
print(f"✓ Casa mejorada de Overall Qual {result['overall_base']} a {result['overall_new']:.2f}")
print(f"✓ Incremento: {result['boost']:.2f} puntos ({result['boost_pct']:.1f}%)")
print(f"✓ {len(result['changes'])} atributos mejoraron")
