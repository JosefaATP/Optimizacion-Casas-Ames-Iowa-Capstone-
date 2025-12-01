# README

> 1) Guía completa para correr y entender el módulo de **remodelación** del proyecto *Optimizacion Casas Ames Iowa Capstone*.  
> 2) Guía completa para entender el módulo de **training** del XGBoost.  
> 3) Guía rápida de los scripts de **análisis**, **sensibilidad** y **comparación** añadidos.

---

# Módulo `optimization/remodel`

## 1) Qué hace este módulo
Construye y resuelve un MIP en Gurobi que decide **qué cambios de remodelación conviene hacer** a una casa para **maximizar la utilidad**:  
`utilidad = (precio_remodelado - costo_total) - costo_inicial`.

- El **precio** lo predice un modelo XGBoost ya entrenado y encapsulado en un pipeline de preprocesamiento.
- Los **costos** de cambiar atributos están definidos en tablas y reglas de negocio.
- El MIP respeta políticas como no degradar calidades, no crear features prohibidas, límites de presupuesto, etc.

## 2) Requisitos rápidos
- Python 3.11 recomendado
- Gurobi 10+ con licencia activa
- Windows, macOS o Linux
- Paquetes del archivo `requirements.txt` en la raíz del repo

Instalación rápida:
```bash
python -m venv .venv311
# activar
# Windows: .venv311\Scripts\activate
# macOS/Linux: source .venv311/bin/activate
pip install -r requirements.txt
```

## 3) Estructura del módulo y para qué sirve cada archivo
Ruta base: `optimization/remodel/`

- `__init__.py`: inicializador sin lógica de negocio.
- `benchmark_remodel.py`: benchmarks de punta a punta; guarda resultados en `bench_out/`.
- `check_env.py`: verifica entorno (Python, paquetes, licencia Gurobi).
- `compat_sklearn.py`: helpers de compatibilidad scikit-learn.
- `compat_xgboost.py`: utilidades para extraer arboles/hojas del XGBoost y conectarlos al MIP.
- `config.py`: **config central** (rutas de datos/modelos, seeds, defaults, flags).
- `costs.py`: tablas de costos y políticas por categoría (+1 dormitorio, baño, m², garage, deck, etc.).
- `features.py`: features modificables, dominios y metadatos del MIP (ordinales, binarios, etc.).
- `gurobi_model.py`: **corazón** del MIP (variables, restricciones, objetivo, políticas de no empeorar, PWL, debug).
- `io.py`: carga/guarda datos, lee casa base por `pid`, exporta resultados, snapshots y reportes.
- `run_opt.py`: **driver principal** de un caso (PID + presupuesto): arma el MIP, resuelve y resume utilidad, precios, costos y cambios.
- `utils.py`: seeds, formatos, timers, trazas, checks.
- `xgb_predictor.py`: encapsula el pipeline de precio (preprocesa, ejecuta XGB y expone funciones al MIP).

## 4) Flujo completo
1. Configura rutas en `optimization/remodel/config.py` (CSV base, modelo XGB, `bench_out/`, presupuestos).
2. Verifica entorno:
   ```bash
   python -m optimization.remodel.check_env
   ```
3. Corre un caso:
   ```bash
   python -m optimization.remodel.run_opt --pid 534402170 --budget 50000
   ```
4. Revisa consola y archivos en `bench_out/`.
5. (Opcional) Benchmark:
   ```bash
   python -m optimization.remodel.benchmark_remodel --n 0 --budget 50000
   ```

## 5) Datos y modelos
- CSV base: `data/processed/base_completa_sin_nulo.csv`
- Modelo XGB: `models/xgb/...` con `model_xgb.joblib` y `booster.json` (ruta en `config.py`).

## 6) Salidas esperadas
En consola: solución óptima, utilidad, precio base/remodelado, costos y lista de cambios.  
En `bench_out/`: CSVs agregados, snapshot de cambios, logs con gap/tiempos si debug está ON.

---

# Módulo `training`

## `bayes_tune_with_retrain.py`
Script de **búsqueda bayesiana** (gp_minimize) para XGBoost; reentrena en cada iteración con `retrain_xgb`, registra métricas y finalistas.

## `cv10_simple.py`
Validación cruzada 10×10 (configurable) para XGBoost con hiperparámetros fijos (`BEST_PARAMS`), soporta categórico nativo u one-hot; genera `fold_scores.csv`, `repeat_means.csv`, `summary.json`.

## `retrain_xgb_same_env.py`
Reentrena el XGB con el mismo preprocesamiento/config del pipeline original; guarda `model_xgb.joblib`, `booster.json`, `metrics.json`, `meta.json`.

## `build_present_from_year_cpi` (ajuste a valor presente)
Recalcula `SalePrice` a `SalePrice_Present` usando CPI y guarda el CSV resultante.

## `grafico.py` (gráficos de búsqueda bayesiana)
Genera gráficos de evolución de R²/RMSE/MAPE a partir de `all_iterations.csv`.

---

# Guía rápida de scripts (análisis, sensibilidad, comparación)

## Análisis de construcción (`analysis/construction_batch/`)
- `batch_run_by_neighborhood.py`: barridos de construcción por barrio/presupuesto.
- Carpetas `results*` (`results`, `results_nomindims`, `results_fullfeatures`, `results_fullbath32`, `results_fullbath48`, `results_2811`): incluyen `analysis.txt`, `construction_runs.csv`, `category_summary.csv`, `neighborhood_comparison.csv`, y subcarpetas `x_inputs/` o `cost_breakdown_*.csv`.
- `combined_temp.csv`: auxiliar para combinar corridas.

## Remodelación (optimización, sensibilidad y comparación)
- `optimization/Copia de sensibilidad_remodelacion 2/sensitivity.py` (“sensitivity 2”): barre barrios/percentiles/presupuestos; guarda `resumen.csv` y `detalles.jsonl` en el `--outdir` elegido.
- En la misma carpeta: `compare_batch_preds.py` y `compare_xgb_vs_reg.py` para comparar predicciones (XGB vs regresión/otros lotes).
- `scripts/summarize_roi.py`: genera `roi_summary.txt` + gráficos (heatmaps/barras) a partir de `resumen.csv` y `detalles.jsonl`. Personaliza rutas con `SENSI_RESUMEN`, `SENSI_DETALLES`, `SENSI_OUT_TXT`, `SENSI_OUT_DIR`.
- `optimization/remodel/run_opt.py`, `gurobi_model.py`, `xgb_predictor.py`: núcleo de optimización y predicción (ver secciones arriba).
- `training/retrain_xgb_same_env.py`: reentrena el XGB de remodelación.

## Construcción (MIP + XGB)
- Caso puntual con pid (o semilla barrio/lote) y presupuesto:
  ```bash
  python -m optimization.construction.run_opt --pid <PID> --budget <USD>
  # o: python -m optimization.construction.run_opt --neigh <Barrio> --lot 7000 --budget 500000
  ```
  Flags útiles: `--xgbdir` (modelo XGB de construcción), `--basecsv`, `--bldg` (tipo edificio), límites manuales (`--min-beds`, `--max-beds`, etc.), `--xinput-outdir` (guarda el vector optimizado), `--outcsv` (append de resultados), `--no-min-dims` (relaja mínimos de áreas/piezas).
- Batch/sensibilidad de construcción: `analysis/construction_batch/batch_run_by_neighborhood.py` ejecuta barridos masivos y deja resultados en `analysis/construction_batch/results*` (CSV de corridas, resúmenes, x_inputs y desgloses de costo según variante).

### Cómo correr sensibilidad sin pisar resultados previos
1. Usa un `--outdir` distinto:
   ```bash
   .venv/bin/python "optimization/Copia de sensibilidad_remodelacion 2/sensitivity.py" \
     --neighborhood all --budgets 20000 50000 100000 --percentiles 0.25 0.5 0.75 \
     --outdir optimization/sensibilidad_remodelacion_relajada
   ```
2. Genera summary y gráficos apuntando al nuevo outdir:
   ```bash
   MPLBACKEND=Agg MPLCONFIGDIR=optimization/sensibilidad_remodelacion_relajada/.mplconfig \
   SENSI_RESUMEN=optimization/sensibilidad_remodelacion_relajada/resumen.csv \
   SENSI_DETALLES=optimization/sensibilidad_remodelacion_relajada/detalles.jsonl \
   SENSI_OUT_TXT=optimization/sensibilidad_remodelacion_relajada/roi_summary.txt \
   SENSI_OUT_DIR=optimization/sensibilidad_remodelacion_relajada \
   .venv/bin/python scripts/summarize_roi.py
   ```

### Notas de uso rápido
- Optimización puntual: `python -m optimization.remodel.run_opt --pid <PID> --budget <USD>`.
- Reentrenar XGB: `python -m training.retrain_xgb_same_env --csv <data/processed/...> --outdir models/xgb/...`.
- Los heatmaps/barras usan paleta verde→rojo y se guardan en el outdir configurado.

---

## Uso y referencias a la IA
El código fue hecho con ayuda de la IA. Links de referencia consultados:
- https://chatgpt.com/g/g-p-67e7192d867c8191b6cc57c7d70b885a-universidad-vale/shared/c/68f15667-a570-832e-a6dd-232c3b7b75cb?owner_user_id=user-WSMVGktdRc8FZkmHo76oEWUE
- https://chatgpt.com/g/g-p-67e7192d867c8191b6cc57c7d70b885a-universidad-vale/shared/c/68fa7f1e-dc9c-8329-b102-4779c19ece59?owner_user_id=user-WSMVGktdRc8FZkmHo76oEWUE
- https://chatgpt.com/g/g-p-67e7192d867c8191b6cc57c7d70b885a-universidad-vale/shared/c/68e5109e-b2b4-832b-8a8a-24b882354dd1?owner_user_id=user-WSMVGktdRc8FZkmHo76oEWUE
- https://chatgpt.com/g/g-p-67e7192d867c8191b6cc57c7d70b885a-universidad-vale/shared/c/68f4a28b-bd48-832e-b6ec-421c75ceb765?owner_user_id=user-WSMVGktdRc8FZkmHo76oEWUE
- https://chatgpt.com/g/g-p-67e7192d867c8191b6cc57c7d70b885a-universidad-vale/shared/c/68e5109e-b2b4-832b-8a8a-24b882354dd1?owner_user_id=user-WSMVGktdRc8FZkmHo76oEWUE
- https://chatgpt.com/share/68fc07e9-e768-8008-9e11-0dead2fd7fc9
- https://chatgpt.com/share/68fc084a-71dc-8008-8b9c-3c0ca6b18326
- https://chatgpt.com/share/68fc08b0-708c-8008-b337-4af0640732e0

### Referencias APA (interacción en línea, sin URL pública)
- Codex (ChatGPT). (2025, 30 de Noviembre). Asistencia en “Optimizacion-Casas-Ames-Iowa-Capstone-” sobre sensibilidad, merge y documentación [Interacción en línea]. OpenAI.
- Codex (ChatGPT). (2025, 29 de Noviembre). “Análisis sensibilidad” sobre análisis y soporte técnico [Interacción en línea]. OpenAI.
