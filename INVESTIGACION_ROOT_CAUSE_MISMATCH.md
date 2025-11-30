# INVESTIGACIÓN PROFUNDA: El Verdadero Problema del Mismatch 13.4%

## 1. Contexto del Hallazgo

**Descubrimiento clave:** El "fix" de threshold boundary (`thr - 1e-8`) **ya estaba en el código** cuando lo revisamos, pero el mismatch persiste.

- Propiedad 528328100 con budget 250000:
  - y_log(MIP) = 13.372293
  - y_log(externa/correcta) = 13.238561
  - **Δ = 0.133733 (13.4% - exactamente igual al bug reportado)**

## 2. Hipótesis Investigadas

### ❌ Hypothesis 1: Threshold Boundary Issue
**Status:** REFUTADO
- La corrección (`thr - 1e-8`) ya estaba en el código en el commit original
- A pesar de estar presente, el mismatch persiste
- **Conclusión:** No es el problema principal

## 3. Áreas de Investigación Activas

### 🔍 Área A: Base Score
**Archivo:** `optimization/remodel/xgb_predictor.py`, líneas 549-565

```python
if (self.b0_offset is None) or (abs(self.b0_offset) < 1e-12):
    try:
        bs_attr = bst.attr("base_score")
        if bs_attr is not None:
            self.b0_offset = float(bs_attr)
        else:
            # fallback: evalúa predict(output_margin) en el origen y resta suma de hojas
            import numpy as _np
            zeros = _np.zeros((1, len(x_list)))
            y_out = float(self.reg.predict(zeros, output_margin=True)[0])
            y_in = self._eval_sum_leaves(zeros.ravel())
            self.b0_offset = float(y_out - y_in)
    except Exception:
        self.b0_offset = 0.0
```

**Problemas potenciales:**
1. ¿Se obtiene correctamente el base_score del booster?
2. ¿Cuándo se ejecuta este código? (una sola vez? cada vez?)
3. ¿Se preserva b0_offset entre llamadas?
4. ¿La selección de `len(x_list)` es correcta? (debería ser el número de features)

**Investigar:**
- Verificar que `bst.attr("base_score")` devuelve el valor correcto
- Comprobar si b0_offset se actualiza correctamente en el MIP
- ¿Se aplica b0_offset al constraint de y_log?

### 🔍 Área B: Suma de Hojas
**Archivo:** `optimization/remodel/xgb_predictor.py`, línea 700

```python
total_expr += gp.quicksum(z[k] * leaves[k][1] for k in range(len(leaves)))
```

**Problemas potenciales:**
1. ¿Los valores `leaves[k][1]` se extraen correctamente?
2. ¿Se redondean o truncan numéricamente?
3. ¿Hay acumulación de errores con 914 árboles?

**Investigar:**
- Verificar precisión de los valores de hojas: ¿Tienen suficientes decimales?
- Comparar suma manual vs suma en MIP
- Revisar si hay pérdida de precisión con `gp.quicksum()`

### 🔍 Área C: Constraint de y_log
**Archivo:** `optimization/remodel/xgb_predictor.py`, línea 703

```python
m.addConstr(y_log == total_expr, name="YLOG_XGB_SUM")
```

**Problemas potenciales:**
1. ¿Es una igualdad estricta o tiene tolerancia?
2. ¿Gurobi cumple exactamente con `y_log == total_expr` o tiene tolerancia numérica?
3. ¿Se considera el base_score aquí?

**Investigar:**
- Verificar que `total_expr` incluye base_score
- ¿Debería ser `y_log == total_expr + b0_offset`?
- Revisar la tolerancia numérica de Gurobi (FeasibilityTol=1e-7)

### 🔍 Área D: Selección de Hojas
**Archivo:** `optimization/remodel/xgb_predictor.py`, líneas 657-697

```python
m.addConstr(xv <= thr - 1e-8 + M_le * (1 - z[k]), name=f"T{t_idx}_L{k}_f{f_idx}_lt")
m.addConstr(xv >= thr - M_ge * (1 - z[k]), name=f"T{t_idx}_R{k}_f{f_idx}_ge")
```

**Problemas potenciales:**
1. ¿La lógica Big-M funciona correctamente con Gurobi?
2. ¿Hay conflictos con múltiples constraints simultáneamente activos?
3. ¿Hay soluciones NO óptimas pero factibles que el solver elige?

**Investigar:**
- Verificar que una sola hoja se selecciona por árbol (one-hot constraint)
- Comprobar que cada árbol está correctamente desacoplado
- Ver si el gap de optimalidad es realmente 0.0%

### 🔍 Área E: Alineación de Características
**Archivo:** Múltiples localizaciones

**Problemas potenciales:**
1. ¿Las características en el MIP están en el mismo orden que en XGBoost?
2. ¿Hay problemas de normalización/escala?
3. ¿Se transforman las características antes de pasarlas al MIP?

**Investigar:**
- Verificar que `x_list[f_idx]` en el MIP corresponde a la misma característica que en XGBoost
- Comprobar el orden de One-Hot Encoding
- Revisar transformaciones del preprocessor

## 4. Hipótesis Principal: Base Score NO se Suma

**Mi intuición:** El problema es que `y_log = total_expr` DEBERÍA ser `y_log = total_expr + b0_offset`

**Razonamiento:**
- XGBoost predice: $\hat{y} = \sum_{i=1}^{914} \text{leaf}_i + \text{base\_score}$
- En el MIP: `total_expr = sum of selected leaves`
- El constraint es: `y_log == total_expr` (SIN base_score)
- Esto significa `y_log` NO incluye base_score, pero debería

**Comprobación:**
- Si esto es correcto, entonces el mismatch sería aproximadamente igual a base_score
- base_score = 12.437748
- Mismatch = 0.133733
- Ratio: 0.133733 / 12.437748 = 0.0107 (1.07%)

Esto NO coincide. El base_score es ~12, pero el mismatch es solo 0.13. Así que no es simplemente que falta el base_score.

## 5. Otra Posibilidad: Precision Numérica en Gurobi

**Problema:** Con 914 árboles × múltiples constraints, hay acumulación de errores numéricos.

**Evidence:**
- Gurobi FeasibilityTol = 1e-7
- Tolerancia relativa probablemente es mayor
- Con 914 árboles, los errores se acumulan

**Investigar:**
- Ejecutar MIP sin optimización y ver qué valores toman z[k]
- Verificar que exactamente 1 z[k] = 1 y el resto = 0 para cada árbol
- Comparar la suma manual con el valor de y_log

## 6. HALLAZGO CRÍTICO: Doble Cálculo de Base Score

**Descubrimiento:** El base_score se calcula DOS VECES en código INCONSISTENTE:

### Primera Ubicación (XGBBundle.__init__, líneas 314-324)
```python
# Se extrae del JSON del modelo
self.b0_offset: float = 0.0
try:
    bst = self.reg.get_booster()
    json_model = bst.save_raw("json")
    data = json.loads(json_model)
    bs_str = data.get("learner", {}).get("learner_model_param", {}).get("base_score", "[0.5]")
    if isinstance(bs_str, str) and "[" in bs_str:
        m = re.match(r"\[\s*([0-9.eE+-]+)\s*\]", bs_str)
        if m:
            self.b0_offset = float(m.group(1))
    else:
        self.b0_offset = float(bs_str) if bs_str else 0.5
except Exception:
    self.b0_offset = 0.5  # fallback
```

### Segunda Ubicación (attach_to_gurobi, líneas 568-578)
```python
if (self.b0_offset is None) or (abs(self.b0_offset) < 1e-12):
    try:
        bs_attr = bst.attr("base_score")
        if bs_attr is not None:
            self.b0_offset = float(bs_attr)
        else:
            # fallback: evalúa predict(output_margin) en el origen...
            zeros = _np.zeros((1, len(x_list)))
            y_out = float(self.reg.predict(zeros, output_margin=True)[0])
            y_in = self._eval_sum_leaves(zeros.ravel())
            self.b0_offset = float(y_out - y_in)
    except Exception:
        self.b0_offset = 0.0
```

**Problema:** El segundo código NUNCA se ejecuta porque el primero ya establece `b0_offset = 12.437748`, lo que NO es None ni cercano a cero.

### Application en gurobi_model.py (línea 1840)
```python
m.addConstr(y_log == y_log_raw + b0, name="YLOG_with_offset")
```

Donde `b0` se obtiene de:
```python
b0 = float(bundle.b0_offset if hasattr(bundle, "b0_offset") else 0.0)
```

**Implicación:** El base_score está siendo usado, pero podría estar **mal calculado** en la primera ubicación.

## 7. Hipótesis Principal: Base Score Incorrecto

**Razonamiento:**
1. El método de extracción JSON con regex (`\[\s*([0-9.eE+-]+)\s*\]`) podría malinterpretar el valor
2. El formato puede haber cambiado en diferentes versiones de XGBoost
3. Los dos métodos usan fuentes diferentes:
   - Método 1 (JSON): Posiblemente desactualizado
   - Método 2 (attr): El método "oficial" de XGBoost

**Test Recomendado:** Ejecutar `investigate_base_score.py` para comparar:
- `b0_offset` almacenado vs calculado manualmente
- Si Δ > 1e-6, hemos encontrado el culpable

## 8. Plan de Acción Inmediato

### Paso 1: Ejecutar investigate_base_score.py
Ver si el base_score se calcula correctamente

### Paso 2: Si hay discrepancia en base_score
Corregir el cálculo en `__init__` para usar `bst.attr("base_score")` directamente

### Paso 3: Verificar constraint de y_log
Confirmar que en `gurobi_model.py` línea 1840 se aplica correctamente

### Paso 4: Debuggear selección de hojas en MIP
Extraer z[k] después de resolver para verificar que se selecciona la hoja correcta por árbol

### Paso 5: Análisis de precisión numérica
Si los anteriores pasos no resuelven el problema, investigar:
- FeasibilityTol más estricta
- Numerical Focus mayor

