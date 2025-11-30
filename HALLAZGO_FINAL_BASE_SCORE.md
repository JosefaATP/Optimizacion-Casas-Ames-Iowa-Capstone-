# CONCLUSIÓN: Root Cause del Mismatch 13.4%

## Hallazgo Clave

Se ha identificado una **duplicación de lógica de base_score** con **métodos conflictivos**:

**XGBBundle.__init__ (línea 314-324):**
- Extrae base_score del JSON del modelo usando regex
- Se ejecuta UNA VEZ al cargar el bundle
- Establece `self.b0_offset = 12.437748`

**XGBBundle.attach_to_gurobi (línea 568-578):**
- Intenta RECALCULAR base_score si es None o ~0
- Nunca se ejecuta porque b0_offset ya = 12.437748
- Contiene lógica de fallback más robusta (predict(zeros) - sum_leaves(zeros))
- **Esta lógica NUNCA se ejecuta**

**Impacto en gurobi_model.py (línea 1840):**
- El constraint `y_log == y_log_raw + b0` usa el valor potencialmente incorrecto de 12.437748
- Si este valor es incorrecto, TODO el MIP predecirá valor incorrecto

## Sospecha Principal

El método de extracción JSON (regex) puede estar:
1. **Desactualizado** - formato de XGBoost ha cambiado
2. **Frágil** - la regex no captura correctamente el valor
3. **Inconsistente** - diferentes versiones de XGBoost guardan base_score diferente

El método de `bst.attr("base_score")` es el método **oficial** y **robusto** de XGBoost.

## Solución Propuesta

**FIX inmediato:** Modificar `XGBBundle.__init__` para usar el método robusto:

```python
# LÍNEAS 314-324: Cambiar de:
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
    self.b0_offset = 0.5

# A:
try:
    bst = self.reg.get_booster()
    # Intentar método 1: attr (oficial de XGBoost)
    bs_attr = bst.attr("base_score")
    if bs_attr is not None:
        self.b0_offset = float(bs_attr)
    else:
        # Fallback método 2: manual (predict@zero - leaves@zero)
        X_zero = np.zeros((1, len(self.feature_names_in())))
        y_margin_zero = self.reg.predict(X_zero, output_margin=True)[0]
        sum_leaves_zero = self._eval_sum_leaves(X_zero.ravel())
        self.b0_offset = float(y_margin_zero - sum_leaves_zero)
except Exception:
    # Último fallback: valor por defecto
    self.b0_offset = 0.5
```

**Ventajas del FIX:**
1. ✅ Usa el método oficial de XGBoost (`attr`)
2. ✅ Fallback manual garantiza precisión
3. ✅ Elimina regex frágil
4. ✅ Consistente con lógica en `attach_to_gurobi`

## Verificación

Para verificar si este es el problema:

1. Ejecutar `investigate_base_score.py`
2. Comparar:
   - `Bundle.b0_offset` (valor actual/incorrecto)
   - `Manual calculation` (valor correcto)
3. Si Δ > 1e-6: ENCONTRAMOS EL BUG
4. Aplicar FIX
5. Re-ejecutar optimización
6. Verificar que mismatch disminuye

## Impacto Estimado

Si b0_offset es incorrecto:
- **Escenario 1:** Error pequeño (ej, 12.4 vs 12.5)
  - Impacto en y_log: ±0.1 (posible causa del ~0.13 mismatch observado)
  - Después de exp(y_log): cambio de ~13% en precio
  
- **Escenario 2:** Error grande (ej, 12.437748 vs algo completamente diferente)
  - Impacto sería mucho mayor

La magnitud del mismatch observado (13.4% o 0.133733 en escala log) es consistente con un error pequeño en base_score.

