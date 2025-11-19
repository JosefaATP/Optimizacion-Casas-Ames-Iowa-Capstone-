# ✅ CHECKLIST: Corregir Overfitting en XGBoost

**Creado**: 18 de noviembre de 2025  
**Modelo**: `completa_present_log_p2_1800_ELEGIDO`  
**Estado Actual**: 🔴 SOBREAJUSTE SEVERO (MAPE ratio 3.08x)

---

## 📋 Pre-Vuelo: Verificar Estado Actual

- [ ] **Confirmar problema**: MAPE test = 7.20% (3.08x peor que train)
  ```bash
  cat models/xgb/completa_present_log_p2_1800_ELEGIDO/metrics.json
  ```

- [ ] **Respaldar modelo actual** (por si acaso)
  ```bash
  cp -r models/xgb/completa_present_log_p2_1800_ELEGIDO \
        models/xgb/completa_present_log_p2_1800_ELEGIDO_BACKUP
  ```

- [ ] **Verificar datos disponibles**
  ```bash
  wc -l data/raw/df_final_regresion.csv
  # Debe ser ~1460 filas
  ```

---

## 🥇 FASE 1: Early Stopping Agresivo (5 minutos)

### ✅ Paso 1: Preparación
- [ ] Crear carpeta de salida
  ```bash
  mkdir -p models/xgb/completa_present_log_p2_early50
  ```

- [ ] Verificar que `src/train_xgb_es.py` existe y tiene early stopping
  ```bash
  grep -q "EarlyStopping" src/train_xgb_es.py && echo "✅ OK" || echo "❌ Falta"
  ```

### ✅ Paso 2: Ejecutar Training
- [ ] Ejecutar con patience=50 (reducido de 200)
  ```bash
  PYTHONPATH=. python3 src/train_xgb_es.py \
    --csv data/raw/df_final_regresion.csv \
    --target SalePrice_Present \
    --outdir models/xgb/completa_present_log_p2_early50 \
    --log_target \
    --patience 50
  ```

- [ ] Esperar a que termine (típicamente 3-5 minutos)

### ✅ Paso 3: Validar Resultados
- [ ] Verificar que se generó metrics.json
  ```bash
  [ -f models/xgb/completa_present_log_p2_early50/metrics.json ] && echo "✅ OK"
  ```

- [ ] Comparar MAPE test
  ```bash
  python3 << 'EOF'
  import json
  
  with open("models/xgb/completa_present_log_p2_1800_ELEGIDO/metrics.json") as f:
      old = json.load(f)['test']['MAPE_pct']
  
  with open("models/xgb/completa_present_log_p2_early50/metrics.json") as f:
      new = json.load(f)['test']['MAPE_pct']
  
  mejora = ((old - new) / old) * 100
  print(f"MAPE anterior:  {old:.2f}%")
  print(f"MAPE nuevo:     {new:.2f}%")
  print(f"Mejora:         {mejora:.1f}%")
  
  if new < 6.0:
      print("\n✅ EXCELENTE - Problema resuelto")
  elif new < 6.5:
      print("\n🟡 BUENO - Puede intentar Fase 2 para más mejora")
  else:
      print("\n⚠️  Insuficiente - Pasar a Fase 2")
  EOF
  ```

- [ ] **Decisión**:
  - [ ] ✅ Si MAPE < 6.0%: **¡PARAR AQUÍ! Problema resuelto**
  - [ ] ⚠️ Si MAPE ≥ 6.0%: Continuar a Fase 2

---

## 🥈 FASE 2: Reducir Complejidad (15 minutos - Opcional)

### ✅ Paso 1: Modificar Hiperparámetros

- [ ] Identificar ubicación de `xgb_params`
  ```bash
  grep -n "xgb_params" src/config.py | head -5
  ```

- [ ] Crear nuevo archivo de config: `src/config_reduced.py`
  ```python
  from config import Config
  
  cfg = Config()
  cfg.xgb_params = {
      "n_estimators": 800,          # ← Reducido de 1800
      "learning_rate": 0.025,
      "max_depth": 5,
      "min_child_weight": 10,
      "subsample": 0.6,              # ← Reducido de 0.7
      "colsample_bytree": 0.6,       # ← Reducido de 0.7
      "reg_lambda": 4.0,             # ← Aumentado de 2.0
      "reg_alpha": 1.0,              # ← NUEVO (L1)
      "tree_method": "hist",
      "objective": "reg:squarederror",
      "n_jobs": -1,
      "random_state": 42,
  }
  ```

- [ ] Crear script de training customizado: `scripts/train_reduced.py`
  ```python
  import sys
  sys.path.insert(0, '.')
  
  from src.config_reduced import cfg
  from src.train_xgb_es import main
  
  # Ejecutar con config reducida
  # ... (copiar lógica de train_xgb_es.py pero con cfg reducida)
  ```

### ✅ Paso 2: Ejecutar Training
- [ ] Ejecutar training con config reducida
  ```bash
  mkdir -p models/xgb/completa_present_log_p2_reduced
  
  PYTHONPATH=. python3 src/train_xgb_es.py \
    --csv data/raw/df_final_regresion.csv \
    --target SalePrice_Present \
    --outdir models/xgb/completa_present_log_p2_reduced \
    --log_target \
    --patience 50
  ```

- [ ] Esperar a que termine

### ✅ Paso 3: Comparar Resultados
- [ ] Ejecutar comparación de 3 modelos
  ```bash
  python3 << 'EOF'
  import json
  
  models = {
      "Original (1800)": "models/xgb/completa_present_log_p2_1800_ELEGIDO",
      "Early50": "models/xgb/completa_present_log_p2_early50",
      "Reduced": "models/xgb/completa_present_log_p2_reduced"
  }
  
  print("\n📊 COMPARATIVA DE MODELOS")
  print("=" * 60)
  print(f"{'Modelo':<25} {'MAPE Test':<15} {'Mejora':<15}")
  print("=" * 60)
  
  baseline = None
  for name, path in models.items():
      try:
          with open(f"{path}/metrics.json") as f:
              mape = json.load(f)['test']['MAPE_pct']
              
          if baseline is None:
              baseline = mape
              mejora = "-"
          else:
              mejora_pct = ((baseline - mape) / baseline) * 100
              mejora = f"{mejora_pct:+.1f}%"
          
          print(f"{name:<25} {mape:>6.2f}%{'':<8} {str(mejora):>14}")
      except:
          print(f"{name:<25} {'N/A':<15}")
  
  print("=" * 60)
  EOF
  ```

- [ ] **Decisión**:
  - [ ] ✅ Si MAPE < 5.5%: **¡EXCELENTE! Modelo mejorado**
  - [ ] 🟡 Si MAPE 5.5-6.0%: **Bueno, pero considerar Fase 3**
  - [ ] ⚠️ Si MAPE > 6.0%: **Pasar a Fase 3**

---

## 🥉 FASE 3: Grid Search (2 horas - Opcional)

**Nota**: Solo si las Fases 1-2 no son suficientes

### ✅ Paso 1: Crear Script de Grid Search
- [ ] Crear `scripts/grid_search_xgb.py` (ver template abajo)
- [ ] Verificar que es ejecutable
  ```bash
  python3 -m py_compile scripts/grid_search_xgb.py
  ```

### ✅ Paso 2: Ejecutar Grid Search
- [ ] Ejecutar búsqueda exhaustiva
  ```bash
  mkdir -p models/xgb/grid_search_results
  
  PYTHONPATH=. python3 scripts/grid_search_xgb.py \
    --csv data/raw/df_final_regresion.csv \
    --target SalePrice_Present \
    --outdir models/xgb/grid_search_results \
    --log_target
  ```

- [ ] Esperar a que termine (2-3 horas típicamente)

### ✅ Paso 3: Revisar Resultados
- [ ] Ver mejor combinación encontrada
  ```bash
  cat models/xgb/grid_search_results/best_config.json
  ```

- [ ] Entrenar modelo final con mejor config
  ```bash
  mkdir -p models/xgb/completa_present_log_p2_gridsearch_best
  # ... comando de training con best_config
  ```

---

## 📊 Paso Final: Generar Reporte

- [ ] Ejecutar análisis completo
  ```bash
  python3 scripts/analizar_overfitting.py
  ```

- [ ] Generar tabla comparativa
  ```bash
  python3 << 'EOF'
  import json
  import pandas as pd
  
  results = {}
  
  models_to_compare = [
      ("Original", "models/xgb/completa_present_log_p2_1800_ELEGIDO"),
      ("Early50", "models/xgb/completa_present_log_p2_early50"),
      ("Reduced", "models/xgb/completa_present_log_p2_reduced"),
  ]
  
  for name, path in models_to_compare:
      try:
          with open(f"{path}/metrics.json") as f:
              m = json.load(f)['test']
              results[name] = {
                  'MAPE': f"{m['MAPE_pct']:.2f}%",
                  'MAE': f"${m['MAE']:,.0f}",
                  'R²': f"{m['R2']:.4f}"
              }
      except:
          pass
  
  df = pd.DataFrame(results).T
  print(df)
  df.to_csv("analisis/comparativa_modelos.csv")
  EOF
  ```

- [ ] Crear documento de conclusiones: `RESULTADO_MEJORA_XGBOOST.md`

---

## 🎯 Criterios de Éxito

### Mínimo Aceptable ✅
- [ ] MAPE test < 6.5%
- [ ] Mejora > 5% respecto a original

### Bueno 🟢
- [ ] MAPE test < 6.0%
- [ ] Mejora > 10% respecto a original

### Excelente 🟢🟢
- [ ] MAPE test < 5.5%
- [ ] Mejora > 20% respecto a original
- [ ] Ratio MAPE < 2.0x

---

## 🚀 Comandos Rápidos (Copy-Paste Ready)

### Solo Early Stopping:
```bash
mkdir -p models/xgb/completa_present_log_p2_early50 && \
PYTHONPATH=. python3 src/train_xgb_es.py \
  --csv data/raw/df_final_regresion.csv \
  --target SalePrice_Present \
  --outdir models/xgb/completa_present_log_p2_early50 \
  --log_target --patience 50
```

### Comparar Resultados:
```bash
python3 scripts/analizar_overfitting.py
```

### Ver Gráficos:
```bash
open analisis/overfitting_analisis.png
open analisis/deterioro_metricas.png
```

---

## 📝 Documentación Generada

Archivos creados durante este análisis:

1. **DIAGNOSTICO_FINAL_OVERFITTING.txt** ← EMPEZAR AQUÍ
2. **ANALISIS_OVERFITTING_XGBOOST.md** ← Análisis técnico
3. **RESUMEN_OVERFITTING_Y_SOLUCIONES.md** ← Plan de acción
4. **GUIA_PRACTICA_OVERFITTING.md** ← Tutorial paso a paso
5. **scripts/analizar_overfitting.py** ← Script de análisis
6. **scripts/test_solucion_1.sh** ← Early stopping
7. **analisis/overfitting_analisis.png** ← Gráficos
8. **analisis/deterioro_metricas.png** ← Comparativa

---

## ⚠️ Notas Importantes

1. **NO SOBRESCRIBAS** el modelo original, siempre crea versiones nuevas
2. **RESPALDA** los datos antes de hacer cambios grandes
3. **DOCUMENTA** cada intento con timestamps
4. **COMPARA** siempre con métricas de test, no train
5. **VALIDA** que los números tienen sentido (no confíes ciegamente)

---

## 📞 Soporte

Si algo falla:
1. Revisa `GUIA_PRACTICA_OVERFITTING.md`
2. Ejecuta `scripts/analizar_overfitting.py` para diagnóstico
3. Revisa logs de ejecución

---

**Estado Actual**: 🔴 Sobreajuste SEVERO  
**Objetivo**: 🟢 MAPE test < 6.0%  
**Tiempo Estimado**: 5-30 minutos (Fase 1-2) o 2+ horas (Fase 3)

¡Vamos a mejorar este modelo! 🚀

