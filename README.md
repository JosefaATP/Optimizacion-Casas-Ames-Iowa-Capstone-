# Optimizacion-Casas-Ames-Iowa-Capstone-

## Guía rápida de scripts

### Análisis de construcción (`analysis/construction_batch/`)
- `batch_run_by_neighborhood.py`: corre barridos de construcción por barrio/presupuesto y escribe CSVs de resultados.
- Carpetas `results*` (`results`, `results_nomindims`, `results_fullfeatures`, `results_fullbath32`, `results_fullbath48`, `results_2811`): cada una contiene:
  - `analysis.txt`: resumen textual.
  - `construction_runs.csv`: ejecuciones individuales.
  - `category_summary.csv`: métricas agregadas por categoría/barrio.
  - `neighborhood_comparison.csv`: comparativa entre barrios.
  - Subcarpetas `x_inputs/` o `cost_breakdown_*.csv` (según variante) con las entradas y desglose de costos usados en cada corrida.
- `combined_temp.csv`: auxiliar para combinar corridas.

### Remodelación (optimización y sensibilidad)
- `optimization/remodel/run_opt.py`: ejecuta la optimización de remodelación para un PID/presupuesto usando el modelo XGBoost y el MIP de Gurobi; imprime trazas y resultados.
- `optimization/remodel/gurobi_model.py`: define el modelo MIP (variables, restricciones, costos) usado por `run_opt.py` y los scripts de sensibilidad.
- `optimization/remodel/xgb_predictor.py`: carga el bundle XGBoost y prepara features/ordinales para inferencia.
- `optimization/Copia de sensibilidad_remodelacion 2/sensitivity.py`: análisis de sensibilidad para remodelación (barre barrios/percentiles/presupuestos) y guarda `resumen.csv` y `detalles.jsonl` en el `--outdir` indicado. (Es el “sensitivity 2”.)
- Otros scripts en esa carpeta de copia para comparar modelos: `compare_batch_preds.py` y `compare_xgb_vs_reg.py` (comparan preds XGB vs regresión lineal y lote de predicciones).
- `scripts/summarize_roi.py`: genera `roi_summary.txt` y gráficos (heatmaps/barras) a partir de `resumen.csv` y `detalles.jsonl`. Rutas personalizables con variables de entorno `SENSI_RESUMEN`, `SENSI_DETALLES`, `SENSI_OUT_TXT`, `SENSI_OUT_DIR`.
- `training/retrain_xgb_same_env.py`: reentrena el XGBoost de remodelación sobre un CSV limpio y guarda el modelo/bundle.

### Cómo correr sensibilidad sin pisar resultados previos
1. Ejecuta sensibilidad con un `--outdir` distinto (ejemplo relajado):
   ```sh
   .venv/bin/python "optimization/Copia de sensibilidad_remodelacion 2/sensitivity.py" \
     --neighborhood all --budgets 20000 50000 100000 --percentiles 0.25 0.5 0.75 \
     --outdir optimization/sensibilidad_remodelacion_relajada
   ```
2. Genera el summary y gráficos apuntando al nuevo outdir:
   ```sh
   MPLBACKEND=Agg MPLCONFIGDIR=optimization/sensibilidad_remodelacion_relajada/.mplconfig \
   SENSI_RESUMEN=optimization/sensibilidad_remodelacion_relajada/resumen.csv \
   SENSI_DETALLES=optimization/sensibilidad_remodelacion_relajada/detalles.jsonl \
   SENSI_OUT_TXT=optimization/sensibilidad_remodelacion_relajada/roi_summary.txt \
   SENSI_OUT_DIR=optimization/sensibilidad_remodelacion_relajada \
   .venv/bin/python scripts/summarize_roi.py
   ```

### Notas de uso rápido
- Para correr una optimización puntual: `python -m optimization.remodel.run_opt --pid <PID> --budget <USD>`.
- Para reentrenar el modelo XGB: `python -m training.retrain_xgb_same_env --csv <data/processed/...> --outdir models/xgb/...`.
- Los heatmaps/barras usan paleta verde→rojo y se guardan en el outdir configurado.
