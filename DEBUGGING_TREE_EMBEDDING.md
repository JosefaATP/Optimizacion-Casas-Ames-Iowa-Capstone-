# Debugging Tree Embedding y_log Mismatch

## Estado Actual: 29 de Noviembre 2025

### Problema Principal
Cuando el MIP optimiza las características de una casa, hay un **mismatch en la predicción y_log de 0.133733** (aproximadamente 13.4% de diferencia). Esto indica que el árbol XGBoost no se está embebiendo correctamente en las restricciones de Gurobi.

### Ejemplo Concreto: Propiedad 528328100

**Predicción Externa (correcta):**
- y_log_raw = 0.800813
- y_log (con base_score) = 13.238561

**Predicción MIP (incorrecta):**
- y_log_raw = 0.934545
- y_log (con base_score) = 13.372293

**Diferencia:** Δ = +0.133733 (MIP es demasiado optimista)

### Hallazgo Clave: Selección Incorrecta de Hojas en 77 Árboles

Del debugging exhaustivo ejecutado, se encontraron **77 mismatches en 914 árboles** donde el MIP está seleccionando hojas DIFERENTES a las que seleccionaría el predictor externo.

**Ejemplo crítico - Árbol 31:**
```
Característica: Kitchen AbvGr = 2.0
Threshold: 2.0

Esperado (correcto): Kitchen AbvGr >= 2 (x=2)
  → Hoja 5, valor = -0.0059

Elegido (incorrecto): Kitchen AbvGr <= 2 (x=2)  ← EQUIVOCADO
  → Hoja 4, valor = +0.0030

Diferencia en valor de hoja: 0.0030 - (-0.0059) = 0.0089
```

**Ejemplo 2 - Árbol 46:**
```
Kitchen AbvGr = 2.0, threshold = 2.0
Esperado: Kitchen AbvGr >= 2 → Hoja 7, valor = +0.0030
Elegido: Kitchen AbvGr <= 2 → Hoja 6, valor = +0.0124 ← EQUIVOCADO
Diferencia: 0.0124 - 0.0030 = 0.0094
```

### Causa Raíz Identificada

En `optimization/remodel/xgb_predictor.py`, líneas 667-691, las restricciones Big-M no manejan correctamente el **caso límite cuando una característica es EXACTAMENTE IGUAL al threshold**.

**La lógica XGBoost:**
- Si `x < threshold` → ir a rama izquierda (yes_child)
- Si `x >= threshold` → ir a rama derecha (no_child)

**La restricción actual (BUGGY):**
```python
# Left child (x < threshold):
xv <= thr + M_le * (1 - z[k])

# Right child (x >= threshold):
xv >= thr - M_ge * (1 - z[k])
```

**El problema:**
Cuando `x == threshold` exactamente:
1. La rama izquierda debería ser FALSA (x < thr es FALSE cuando x == thr)
2. Pero la restricción `xv <= thr + M_le * (1 - z[k])` PERMITE `xv = thr` incluso cuando z[k] = 0
3. Esto permite que el MIP seleccione la hoja izquierda cuando debería ser la derecha

**Ejemplo numérico:**
- Kitchen AbvGr = 2, threshold = 2
- UB = 3, LB = -0 (rango de decisión)
- M_le = 3 - 2 = 1

Con z[k] = 0 (hoja NO seleccionada):
- `xv <= 2 + 1 * (1 - 0) = 3` ✓ La restricción se satisface con xv = 2
- Esto permite que la hoja izquierda sea seleccionada aunque x ≥ thr

### Soluciones Probadas (Sin éxito aún)

#### Intento 1: Agregar pequeño epsilon
```python
# LEFT (tried):
xv <= thr - 1e-8 + M_le * (1 - z[k])
```
**Resultado:** No funcionó - Gurobi ignora epsilon tan pequeño

#### Intento 2: M_ge grande cuando M_ge ≈ 0
```python
if M_ge < 1e-6:
    m.addConstr(xv >= thr - 1e6 * (1 - z[k]), ...)
```
**Resultado:** Sin cambios, aún 77 mismatches

#### Intento 3: Epsilon más agresivo (1e-8) en restricción LEFT
```python
xv <= thr - 1e-8 + M_le * (1 - z[k])
```
**Resultado:** Sin cambios

### Soluciones a Explorar

#### Opción A: Reformular como restricción bidireccional
En lugar de usar solo `x < thr` para left y `x >= thr` para right, crear restricciones que se refuercen mutuamente:

```python
# Para hoja izquierda (x < thr):
m.addConstr(xv <= thr - 1e-6 + M_le * (1 - z[k]))

# Para hoja derecha (x >= thr):
m.addConstr(xv >= thr + M_ge * (1 - z[k]))  # Nota: + en lugar de -
```

#### Opción B: Usar formulación de "disjunctive constraints" (OR)
En lugar de Big-M simple, usar:
```python
# Si z[k]=1, ENTONCES la rama izquierda está activa
# Lo cual significa: x < thr Y todas las condiciones previas
# Agregar constraint que une múltiples ramas:
m.addConstr(xv <= thr - 1e-6 + BIG_M * (1 - z[k]))
m.addConstr(thr - xv <= BIG_M * z[k])  # Si z[k]=0, fuerza xv >= thr
```

#### Opción C: Usar variable auxiliar para threshold boundary
```python
# Crear variable binaria que indica: xv < thr ?
b_left = m.addVar(vtype=GRB.BINARY, name=f"b_left_{t_idx}_{k}_{f_idx}")
m.addConstr(xv <= thr - 1e-6 + BIG_M * (1 - b_left))
m.addConstr(xv >= thr + 1e-6 * b_left - BIG_M * b_left)

# Ahora usar b_left en lugar de z[k] para forzar la rama
m.addConstr(b_left == z[k])
```

#### Opción D: Cambiar tolerancias de Gurobi
En `optimization/remodel/run_opt.py`, línea 515, aumentar la precisión:
```python
m.Params.FeasibilityTol = 1e-9  # (actual: 1e-7)
m.Params.OptimalityTol = 1e-9   # (actual: 1e-7)
m.Params.NumericFocus = 3        # (actual: 3, ya está al máximo)
```

### Archivos Críticos

1. **`optimization/remodel/xgb_predictor.py`** (líneas 667-691)
   - Contiene la lógica Big-M que está buggy
   - Función: `attach_to_gurobi()`
   - Necesita fix en la formulación de restricciones

2. **`optimization/remodel/run_opt.py`** (línea 515)
   - Donde se setean parámetros de Gurobi
   - Podría necesitar ajustes de tolerancia

3. **`optimization/remodel/gurobi_model.py`** (línea 1839)
   - Donde se define la restricción: `y_log == y_log_raw + b0`
   - Actualmente correcto

### Pasos para Reproducir el Bug

```bash
cd "c:\Users\valen\OneDrive - uc.cl\UNIVERSIDAD\2024-1\Bases de datos\Optimizacion-Casas-Ames-Iowa-Capstone-"

# Ejecutar con debug flags
.venv311\Scripts\python.exe -m optimization.remodel.run_opt --pid 528328100 --budget 500000 --debug-tree-mismatch

# Buscar en output:
# [TREE-MISMATCH] y_log_manual=13.238551 | y_log(MIP)=13.372293 | Δ=+0.133742
# [TREE-MISMATCH] Mismatches en 77 árboles
```

### Próximos Pasos Recomendados

1. **Implementar Opción B o C:** Reformular restricciones para crear fuerza bidireccional
   - Esto debería forzar que cuando z[k]=1, TODAS las condiciones (incluyendo < vs >=) se respeten
   
2. **Aumentar tolerancias de Gurobi** en run_opt.py
   - Aunque poco probable que ayude, vale la pena intentar

3. **Validar con propiedad base 526351010**
   - Esta propiedad tiene y_log_raw correcto (sin MIP optimization)
   - Usar para verificar que la fix no rompe el caso ya funcionando

4. **Testing con múltiples propiedades**
   - Después del fix, correr en varias propiedades para confirmar patrón

### Notas Técnicas

- **base_score (b0):** 12.437748 (este offset está correcto)
- **Número de árboles:** 914 (todos embebidos)
- **Número de variables binarias:** 3,961 (selectores de hojas)
- **Tolerancias actuales:** FeasibilityTol=1e-7, OptimalityTol=1e-7

### Recursos Útiles

- XGBoost tree split semantics: https://xgboost.readthedocs.io/
- Big-M formulation theory: Mixed-Integer Programming, Wolsey & Nemhauser
- Gurobi MIP modeling: https://www.gurobi.com/documentation/

---

**Última actualización:** 29 Nov 2025
**Estado:** BLOQUEADO en selección de hojas con threshold boundaries
**Prioridad:** CRÍTICA (impacta precisión de optimización)
