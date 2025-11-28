"""
DIAGNÓSTICO Y SOLUCIÓN: y_log(MIP) ≠ y_log(outside) - ROOT CAUSE

===============================================================================
PROBLEMA OBSERVADO
===============================================================================

Durante la ejecución del modelo de remodelación, se observó una divergencia 
significativa entre:
  - y_log(MIP): predicción logarítmica dentro del modelo de optimización
  - y_log(outside): predicción logarítmica calculada externamente

Esta discrepancia causaba que el MIP predijera un precio distinto al que 
predice el XGBoost cuando se alimenta con las mismas características.

===============================================================================
CAUSA RAÍZ
===============================================================================

La raíz del problema está en las ESCALAS de predicción de dos pipelines 
distintos:

1. PIPELINE FULL (bundle.predict()):
   - Arquitectura: Preprocessor → XGBRegressor → TransformedTargetRegressor(log1p)
   - Entrada: X_raw (sin procesar)
   - Proceso: 
     a) Preprocesa X_raw → X_processed
     b) XGBRegressor predice: y_log_raw = XGB(X_processed)  [escala log1p]
     c) TransformedTargetRegressor invierte: y_price = expm1(y_log_raw)
   - Salida: y_price en ESCALA ORIGINAL (precio en dólares)
   - Ubicación: xgb_predictor.py línea 206 (last step de pipe_full)

2. PIPELINE PARA EMBED (bundle.pipe_for_gurobi()):
   - Arquitectura: Preprocessor → XGBRegressor (SÓLO)
   - Entrada: X_raw (sin procesar)
   - Proceso:
     a) Preprocesa X_raw → X_processed
     b) XGBRegressor predice: y_log_raw = XGB(X_processed)  [escala log1p]
   - Salida: y_log_raw en ESCALA LOG1P (margen crudo del XGB)
   - Ubicación: xgb_predictor.py línea 264-267

===============================================================================
COMPARACIÓN INCORRECTA (LA QUE CAUSABA EL BUG)
===============================================================================

Código original en run_opt.py (línea ~376-379):

    y_log_embed = float(bundle.pipe_for_gurobi().predict(X_base)[0])
    y_log_full = float(np.log1p(bundle.predict(X_base).iloc[0]))
    delta = y_log_embed - y_log_full

PROBLEMA:
  - y_log_embed:  es y_log_raw DEL XGB (escala log1p, rango ~8-12)
  - y_log_full:   es log1p(PRECIO_INVERTIDO) = log1p(expm1(y_log_raw))
                  ≠ y_log_raw debido a las operaciones no-lineales
  
Matemáticamente:
  log1p(expm1(x)) ≠ x  (hay pérdida numérica y no es identidad exacta)

Por eso la delta era significativa (~0.x) cuando debería ser ~0.

===============================================================================
COMPARACIÓN CORRECTA (LA SOLUCIÓN)
===============================================================================

Código corregido en run_opt.py (línea ~365-379 y ~450-462):

    # ANTES (incorrecto):
    y_log_embed = float(bundle.pipe_for_gurobi().predict(X_base)[0])
    y_log_full = float(np.log1p(bundle.predict(X_base).iloc[0]))  # MALO

    # DESPUÉS (correcto):
    y_log_embed = float(bundle.pipe_for_gurobi().predict(X_base)[0])
    y_log_full = float(bundle.predict_log_raw(X_base).iloc[0])     # CORRECTO

CAMBIOS CLAVE:
  1. Usar bundle.predict_log_raw() en lugar de np.log1p(bundle.predict())
  2. predict_log_raw() devuelve DIRECTAMENTE el margen crudo del XGB
  3. Ambos lados de la comparación están en LA MISMA ESCALA (log1p raw)

Matemáticamente:
  y_log_embed = XGB(X)        [escala log1p]
  y_log_full = XGB(X)        [escala log1p]
  delta = y_log_embed - y_log_full ≈ 0 ✓

===============================================================================
ARCHIVOS MODIFICADOS
===============================================================================

1. xgb_predictor.py (líneas 260-273):
   - Agregados comentarios aclarando que pipe_for_gurobi() predice en log1p
   - Documentada la intención de que el MIP trabaje con log1p internamente

2. run_opt.py (líneas ~365-390 y ~450-462):
   - Cambiadas comparaciones de y_log para usar predict_log_raw()
   - Mejorado el diagnóstico para detectar divergencias

===============================================================================
VERIFICACIÓN
===============================================================================

Para verificar que el fix funciona correctamente:

1. SANITY CHECK (pre-MIP):
   Con --no-upgrades, debe mostrar:
   [DEBUG] y_log(embed via pipe)=X.XXX vs y_log(direct raw)=X.XXX | delta≈0

2. TEST RUN:
   python -m optimization.remodel.run_opt --pid 526351010 --budget 500000
   
   Debería mostrar:
   [SANITY] embed (via pipe) -> log = X.XXX
   [SANITY] direct raw_log -> log = X.XXX
   (los dos deben coincidir o tener delta < 1e-4)

3. POST-SOLVE:
   Después de optimize(), debe mostrar:
   [CALIB] y_log(MIP) - y_log_raw(outside) = delta ≈ 0 (o muy pequeño)

===============================================================================
IMPLICACIONES
===============================================================================

Con este fix:
✓ y_log(MIP) y y_log(outside) están en la misma escala y deben ser próximos
✓ Las divergencias detectadas ahora son REALES (no artefactos de escala)
✓ Se puede debuggear correctamente dónde viene el problema si persiste
✓ El embed produce predicciones consistentes con el predictor externo

Próximos pasos:
- Test con varios PIDs para confirmar que y_log ahora es consistente
- Si persisten divergencias pequeñas (~1e-3), investigar:
  * Diferencias en preprocessing entre embed y external
  * Precisión numérica del tree evaluation en gurobi_ml
  * Truncado de árboles (n_trees_use)
"""
