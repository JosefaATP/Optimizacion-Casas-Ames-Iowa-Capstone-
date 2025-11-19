# Benchmark de Construcción (Guía Operativa)

Este documento explica todo el flujo que alimenta el benchmark de construcción: cómo se generan las corridas del optimizador MIP, cómo se consolidan los resultados y cómo se ensamblan las láminas del reporte en LaTeX. Está orientado a cualquier persona que no haya trabajado antes con el proyecto pero necesite reproducir o extender la capa de construcción.

## 1. Conceptos clave

- **Modelos predictivos:**  
  - *Regresión lineal* (`reg_model.joblib`): baseline conservador.  
  - *XGBoost (`model_xgb.joblib`):* referencia comercial que captura atributos no lineales.
- **Optimizador MIP (`optimization.construction.run_opt`):** dado un barrio/lote o PID y un presupuesto, decide el set de atributos (metros, dormitorios, calidades, etc.) que maximizan el precio esperado (XGB) sujeto a costos y restricciones.
- **Escenarios vs. grid:**  
  - *Modo grid:* barrido manual sobre presupuestos, perfiles de solver y tipos de edificio.  
  - *Modo escenarios:* se leen configuraciones predefinidas (JSON) con restricciones y tags de negocio.

Todo el pipeline vive bajo `bench_out/construction_benchmark/` y se alimenta del histórico de corridas `bench_out/benchmark_runs.csv`.

## 2. Organización de carpetas

```
bench_out/construction_benchmark/
├── run_mip_sensitivity.py          # Barrido de escenarios/grid (llama a run_opt)
├── mip_scenarios.json              # Escenarios por defecto (pueden editarse)
├── mip_scenarios*.csv              # Resultados crudos y resumidos
├── process_mip_scenarios.py        # Limpia/agrega los CSV de escenarios
├── export_plots.py                 # Genera los PNG usados en el reporte
├── update_aggregates.py            # Recalcula métricas base (budget, barrio, lote)
├── figures/                        # Carpeta donde caen los gráficos finales
├── benchmark_report.tex            # Reporte LaTeX (salida: benchmark_report.pdf)
└── sensitivity/                    # Resultados del análisis analítico (run_sensitivity.py)
```

Archivos auxiliares importantes:

- `benchmark_runs.csv`: tabla maestra con todas las corridas del benchmark de construcción (precio XGB/Reg, costos, slack, ROI, features, etc.).
- `summary_overview.json`: estadísticas globales (media, percentiles) consumidas en el reporte.
- `remodel_benchmark.csv`: referencia rápida para el estudio de remodelaciones (mismo stack predictivo).

## 3. Flujo operativo resumido

1. **Generar o actualizar escenarios MIP**  
   ```bash
   # Ejemplo usando los escenarios por defecto definidos en mip_scenarios.json
   PYTHONPATH=. python bench_out/construction_benchmark/run_mip_sensitivity.py \
       --out bench_out/construction_benchmark/mip_scenarios.csv \
       --neigh Veenker --lot 7000 \
       --scenario-file bench_out/construction_benchmark/mip_scenarios.json
   ```
   - Usa `--use-default-scenarios` para los `DEFAULT_SCENARIOS` embebidos en el script.
   - En modo grid (sin escenarios) agrega `--budgets ... --profiles ... --bldgs ...` y restricciones globales (`--min-beds`, etc.).

2. **Procesar los escenarios y generar agregados dedicados**  
   ```bash
   python bench_out/construction_benchmark/process_mip_scenarios.py
   ```
   Salidas clave:
   - `mip_scenarios_processed.csv`: corridas con tags limpios y ROI calculado.
   - `mip_scenarios_summary.csv`: promedio/máximo de ROI y slack por escenario.
   - `mip_scenarios_roi_by_budget.csv`: curva ROI vs. presupuesto, usada en la Figura “ROI (%) por presupuesto”.

3. **Actualizar agregados globales del benchmark de construcción**  
   ```bash
   python bench_out/construction_benchmark/update_aggregates.py
   ```
   Este script lee `benchmark_runs.csv` y genera:
   - `price_by_budget.csv`, `delta_by_neighborhood_top.csv`, `price_by_lot.csv`.
   - `roi_hist.csv`, `roi_by_neighborhood.csv`, `features_vs_baseline*.csv`.
   - `summary_overview.json` (estadísticos globales) y otros CSV de soporte.

4. **Exportar los gráficos**  
   ```bash
   python bench_out/construction_benchmark/export_plots.py
   ```
   Produce los PNG que consume LaTeX (`figures/budget_prices.png`, `gap_by_neighborhood.png`, `mip_roi_vs_budget.png`, etc.). Si cambias los archivos de entrada, vuelve a correr este paso antes de compilar.

5. **Compilar el informe**  
   ```bash
   cd bench_out/construction_benchmark
   pdflatex benchmark_report.tex
   ```
   (En Windows/MiKTeX ejecuta también `python bench_out/construction_benchmark/update_aggregates.py` previo a `pdflatex` para asegurar que los CSV estén al día.)

## 4. Detalle de cada script

### 4.1 `run_mip_sensitivity.py`
- **Entradas obligatorias:** `--out` (CSV destino) y un identificador base (`--pid` o `--neigh + --lot`).
- **Modos de uso:**
  - *Grid:* especifica `--budgets`, `--profiles` y `--bldgs`. Se pueden sumar restricciones globales (`--min-beds`, `--max-grliv`, etc.).
  - *Escenarios:* `--scenario-file mip_scenarios.json` o `--use-default-scenarios`. Cada escenario puede definir `budgets`, `profiles`, `bldgs`, `constraints` y `pid/neigh/lot`.
- **Internamente:** construye el CLI para `optimization.construction.run_opt` y agrega tags semiestructurados (`scenario=...,profile=...,budget=...`) que después lee `process_mip_scenarios.py`.

### 4.2 `process_mip_scenarios.py`
- Limpia los tags, calcula ROI (`(precio - costo)/costo`), agrega por escenario y por tipo de edificio y deja todo listo para las tablas/figuras del reporte.
- No depende de `pandas`, sólo usa `csv` para mantenerlo ligero.

### 4.3 `update_aggregates.py`
- Recorre `benchmark_runs.csv` y arma todos los agregados que alimentan tanto los párrafos como las tablas del reporte.
- Calcula estadísticas por:
  - *Presupuesto (price_by_budget.csv):* precio XGB/Reg, brecha USD/%, slack, ROI, presupuesto usado y conteo de corridas negativas.
  - *Barrio (delta_by_neighborhood_top.csv, roi_by_neighborhood.csv):* brecha, ROI y cantidad de corridas.
  - *Lote (price_by_lot.csv):* respuesta del XGB al tamaño de lote.
  - *Features promedio (features_vs_baseline*.csv):* compara el diseño óptimo vs. los promedios históricos por barrio.
  - *Resumen global (summary_overview.json):* medias, percentiles y desvíos de precio, ROI, slack, obj, etc.

### 4.4 `export_plots.py`
- Carga los CSV anteriores y genera los PNG finales (Matplotlib).  
- Incluye gráficos específicos para sensibilidad MIP: `mip_scenario_roi_summary.png` y `mip_roi_vs_budget.png`. Asegúrate de correr este paso después de `process_mip_scenarios.py` para refrescar estas figuras.

## 5. Cómo editar/agregar escenarios

1. Duplica o modifica `mip_scenarios.json`. Cada entrada acepta:
   ```json
   {
     "name": "starter_1fam",
     "budgets": [350000, 400000, 450000],
     "profile": "balanced",
     "bldgs": ["1Fam"],
     "constraints": {
       "min_beds": 2,
       "min_fullbath": 1,
       "min_overallqual": 6,
       "min_grliv": 900
     },
     "neigh": "Veenker",
     "lot": 7000
   }
   ```
   - Puedes mezclar `profiles` (lista) y `bldgs`.
   - Si omites `neigh/lot`, debes pasar `--neigh` y `--lot` al comando principal o definir `pid`.
2. Ejecuta `run_mip_sensitivity.py --scenario-file tu_archivo.json`.
3. Procesa los resultados y actualiza el reporte siguiendo el flujo del punto 3.

## 6. Notas sobre sensibilidad analítica (no-MIP)

La carpeta `sensitivity/` contiene los barridos de `run_sensitivity.py` (controlado desde notebooks/scripts fuera de este README). Los CSV (`univariate.csv`, `bivariate_*.csv`, `tornado.csv`, `scenarios.csv`) se referencian en el reporte dentro de la sección “Sensibilidad general de regresión/XGB”. Si regeneras estos archivos, basta con volver a compilar LaTeX; no hace falta tocar los scripts de MIP.

## 7. Buenas prácticas y troubleshooting

- **PYTHONPATH:** cuando ejecutes `run_mip_sensitivity.py`, antepone `PYTHONPATH=.` para que el script encuentre los módulos `optimization.*`.
- **Logs del optimizador:** cada corrida imprime `[i/n] scenario=...`; si aparece `[WARN] run_opt terminó con status=...`, revisa los parámetros (presupuesto bajo, restricciones incompatibles, etc.).
- **Reporte LaTeX:** si `pdflatex` marca referencias indefinidas, asegúrate de haber corrido `export_plots.py` y que los PNG existan en `figures/`.  
- **Rendimiento:** usa `--quiet`, `--fast` o `--deep` según necesites controlar verbosity y profundidad del solver (los flags se pasan directamente a `run_opt`).
- **Backups:** `mip_scenarios.csv` se sobreescribe; haz una copia si quieres conservar corridas previas.

Con estos pasos deberías poder replicar completamente el flujo de construcción: generar combinaciones con el MIP, agregar resultados, producir los gráficos y compilar el informe final.
