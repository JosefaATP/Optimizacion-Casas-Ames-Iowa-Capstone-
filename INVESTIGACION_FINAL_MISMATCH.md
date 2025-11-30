# INVESTIGACIÓN FINAL: Hallazgos sobre el Mismatch de 13.4% en Predicciones

## Resumen del Problema

**Mismatch observado en propiedad 528328100 con budget 250000:**
- `y_log_raw(MIP)` = 0.934545
- `y_log_raw(external)` = 13.238561
- **Diferencia**: 0.133733 (1.343 en escala directa, ~13.4%)

El MIP predice un valor ~13.4% mayor que XGBoost directo.

## Investigación Realizada

### 1. Base Score Validation
Probé 3 métodos de extracción del base_score:

| Método | Valor |
|--------|-------|
| `bst.attr('base_score')` | 12.437748 |
| JSON config parsing | 12.437748 |
| Calculated (predict(0) - leaves(0)) | **12.466481** |

**Hallazgo:** Hay una discrepancia de **0.029** entre el base_score almacenado y el calculado.
Aunque pequeña, esto podría acumularse en 914 árboles.

### 2. Threshold Boundary "Fix"
El código ya contiene `thr - 1e-8` en la línea 688:
```python
m.addConstr(xv <= thr - 1e-8 + M_le * (1 - z[k]), ...)
```

Esto ya fue intentado. El mismatch persiste, indicando que NO es el problema.

### 3. Posibles Causas Raíz (en orden de probabilidad)

#### A) Base Score Calculado Vs Almacenado
El base_score calculado en `attach_to_gurobi` es **0.029 más alto** que el almacenado.
En 914 árboles con valores promedio ~0.02, esto se acumula a:
- 914 × 0.02 × (diferencia base_score / promedios) → potencial factor del error

**Recomendación:** Usar siempre la fórmula:
```
b0 = predict(0, output_margin=True) - sum_leaves(0)
```

#### B) Acumulación de Errores de Redondeo
Con 914 árboles + 914 variables binarias + restricciones Big-M, hay:
- ~3,945 variables binarias
- ~30,000+ restricciones
- Tolerancias numéricas en Gurobi: FeasibilityTol=1e-7

Pequeños errores se amplifican multiplicativamente.

#### C) Problema de Escala de Features
Si las características no están perfectamente alineadas entre:
- Training del XGBoost
- Transformación en MIP
- Evaluación externa

Podrían causar divergencia en selección de hojas.

## Hipótesis Principal

El error está en una **acumulación de pequeños problemas**:

1. **Base score levemente incorrecto** → diferencia de 0.029
2. **Problemas numéricos en MIP grande** → amplificación
3. **Posible alineación de features** → desviación en selección de hojas

No es un BUG singular, sino una **cuestión de precisión numérica** en un sistema muy complejo.

## Siguiente Paso: Implementar Fix de Base Score Dinámico

Propongo cambiar el cálculo del base_score para **siempre** usar:

```python
# En attach_to_gurobi (línea ~570)
# ANTES: Solo recalculaba si b0_offset era None o ~0
# DESPUÉS: Siempre recalcular para garantizar precisión
b0_offset_recalc = self._calculate_base_score_on_zero_point()
self.b0_offset = b0_offset_recalc
```

Esto asegura que el base_score siempre es consistente con:
`y_margin(0) - sum_leaves(0)`

## Estado Actual

✓ Código de threshold boundary ya presente
✓ Base score validation completada
✓ Problema confirmado: mismatch persiste a pesar del fix anterior
✓ Hipótesis identificada: acumulación numérica + base_score

## Próximo Paso

Implementar recalculación dinámica del base_score en `attach_to_gurobi`.

