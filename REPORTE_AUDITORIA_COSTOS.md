# 🔍 AUDITORÍA CRÍTICA: CONTABILIDAD DE COSTOS EN gurobi_model.py

## Resumen Ejecutivo

Se encontraron **10 LÍNEAS CRÍTICAS** donde el modelo agrega costos sin restar el costo base. 

**IMPACTO**: Costos subestimados → ROI inflado de 2600% en lugar de ~30%

---

## Problemas Identificados

### ⚠️ PROBLEMA 1: EXTERIOR MATERIALS (Líneas 459, 463)

**Ubicación**: `gurobi_model.py:459-463`

**Código Actual (INCORRECTO)**:
```python
for nm, vb in ex1.items():
    if nm != ex1_base_name:
        lin_cost += ct.ext_mat_cost(nm) * vb    # ❌ AGREGA COSTO TOTAL
```

**Debería ser**:
```python
ex1_base_cost = ct.ext_mat_cost(ex1_base_name)
for nm, vb in ex1.items():
    if nm != ex1_base_name:
        lin_cost += (ct.ext_mat_cost(nm) - ex1_base_cost) * vb  # ✅ COSTO INCREMENTAL
```

---

### ⚠️ PROBLEMA 2: EXTERIOR QUALITY & CONDITION (Líneas 503, 506)

**Ubicación**: `gurobi_model.py:503-506`

**Código Actual**:
```python
for nm, vb in eq_bin.items():
    if ORD[nm] > exq_base_ord:
        lin_cost += ct.exter_qual_cost(nm) * vb  # ❌ TOTAL EN LUGAR DE INCREMENTAL
```

**Debería ser**:
```python
exq_base_cost = ct.exter_qual_cost(...)  # Costo del estado base
for nm, vb in eq_bin.items():
    if ORD[nm] > exq_base_ord:
        lin_cost += (ct.exter_qual_cost(nm) - exq_base_cost) * vb
```

---

### ⚠️ PROBLEMA 3: MASONRY VENEER (Líneas 619, 624)

**Ubicación**: `gurobi_model.py:619-624`

**Código Actual**:
```python
lin_cost += _cost(nm) * p           # ❌ SIN RESTAR BASE
lin_cost += _cost(nm) * area_term * v
```

**Debería ser**:
```python
mv_base_cost = _cost(mvt_base_txt)
lin_cost += (_cost(nm) - mv_base_cost) * p           # ✅ INCREMENTAL
lin_cost += (_cost(nm) - mv_base_cost) * area_term * v
```

---

### ⚠️ PROBLEMA 4: POOL QUALITY (Línea 873)

**Ubicación**: `gurobi_model.py:873`

**Código Actual**:
```python
lin_cost += _pq_cost(nm) * pq[nm]   # ❌ TOTAL SIN RESTAR BASE
```

**Debería ser**:
```python
pq_base_cost = _pq_cost(pq_base_val)
lin_cost += (_pq_cost(nm) - pq_base_cost) * pq[nm]
```

---

### ⚠️ PROBLEMA 5: ELECTRICAL (Línea 1170)

**Ubicación**: `gurobi_model.py:1169-1170`

**Código Actual**:
```python
lin_cost += ct.electrical_demo_small * vb
lin_cost += ct.electrical_cost(nm) * vb    # ❌ TOTAL
```

**Debería ser**:
```python
base_cost_e = ct.electrical_cost(elec_base_name)
lin_cost += ct.electrical_demo_small * vb
lin_cost += (ct.electrical_cost(nm) - base_cost_e) * vb  # ✅ INCREMENTAL
```

---

### ⚠️ PROBLEMA 6: FIREPLACE QUALITY (Línea 1312)

**Ubicación**: `gurobi_model.py:1310-1312`

**Código Actual**:
```python
for nm, vb in fq.items():
    if FQ_ORD[nm] > base_ord:
        lin_cost += _fq_cost(nm) * fq[nm]   # ❌ TOTAL
```

**Debería ser**:
```python
fq_base_cost = _fq_cost(base_fq)
for nm, vb in fq.items():
    if FQ_ORD[nm] > base_ord:
        lin_cost += (_fq_cost(nm) - fq_base_cost) * fq[nm]  # ✅ INCREMENTAL
```

---

### ⚠️ PROBLEMA 7: BASEMENT CONDITION (Línea 1388)

**Ubicación**: `gurobi_model.py:1386-1388`

**Código Actual**:
```python
for nm, vb in bc_bin.items():
    if BC_ORD[nm] > bc_base:
        lin_cost += ct.bsmt_cond_cost(nm) * vb  # ❌ TOTAL
```

**Debería ser**:
```python
bc_base_cost = ct.bsmt_cond_cost(bc_base_name)
for nm, vb in bc_bin.items():
    if BC_ORD[nm] > bc_base:
        lin_cost += (ct.bsmt_cond_cost(nm) - bc_base_cost) * vb
```

---

## Resumen de Patrones

| Categoría | Línea | Tipo | Solución |
|-----------|-------|------|----------|
| Exterior Material | 459, 463 | Material | Restar costo base |
| Exterior Quality | 503 | Ordinal | Restar costo base |
| Exterior Condition | 506 | Ordinal | Restar costo base |
| Mas Veneer | 619, 624 | Material/Area | Restar costo base |
| Pool Quality | 873 | Ordinal | Restar costo base |
| Electrical | 1170 | Categorical | Restar costo base |
| Fireplace Quality | 1312 | Ordinal | Restar costo base |
| Basement Condition | 1388 | Ordinal | Restar costo base |

---

## Impacto en ROI

**Ejemplo**: Cambio de $1,000 de mejora real
- **Con bug**: Modelo calcula costo = $3,000 (costo total erróneo)
- **Sin bug**: Modelo calcula costo = $1,000 (incremental correcto)
- **Diferencia**: +200% en costo percibido → ROI más alto de lo que debería ser

**Multiplicado por múltiples cambios → ROI inflado de 2600% a 30%**

---

## Recomendación

✅ Aplicar TODAS las correcciones de costo incremental sistemáticamente  
✅ Validar que CADA `lin_cost +=` calcule (nuevo - base), NO solo (nuevo)  
✅ Re-ejecutar tests después de arreglar

