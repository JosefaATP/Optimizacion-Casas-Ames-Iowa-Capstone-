# Construction Benchmark Assets

Este directorio agrupa los resultados derivados del barrido de construcción
almacenado en `bench_out/benchmark_runs.csv`.

## Contenido

- `benchmark_runs.csv`: corrida cruda (una fila por `neigh`–`lot`–`budget`).
- `delta_by_neighborhood.csv`: diferencia promedio XGB–Regresión por barrio.
- `price_by_budget.csv`: promedio de precio XGB/Reg, delta y slack por presupuesto.
- `price_by_lot.csv`: promedios por tamaño de lote (muestra que el precio depende
  más del barrio que del área disponible en este setup).
- `delta_by_neighborhood_top.csv`: top 10 barrios con mayor diferencial XGB–Reg, para graficar rápidamente.
- `features_vs_baseline.csv`: promedio de atributos (Gr Liv Area, sótano, garage,
  dormitorios, baños) según el benchmark y según la base histórica.
- `features_vs_baseline_top.csv`: subset para seis barrios representativos, útil
  para graficar comparaciones lado a lado.
- `summary_overview.json`: métricas globales (media, mediana, etc.) para
  `y_price`, `reg_price`, `delta`, `slack`, `cost` y `runtime`.
- `benchmark_report.tex`: informe LaTeX con gráficos (pgfplots) que
  referencian las tres tablas agregadas. Puedes compilarlo con `pdflatex`.

## Reproducir / Ampliar

1. **Ejecutar barrido** (PowerShell):
   ```powershell
   $env:PYTHONPATH = "."
   python scripts/benchmark_construction.py `
       --out bench_out/benchmark_runs.csv `
       --neigh-all `
       --lots 1300 1700 3200 5000 7500 11500 31700 `
       --budgets 200000 400000 600000 800000 1000000 `
       --quiet
   Remove-Item Env:PYTHONPATH
   ```
   > Nota: añadí los extremos `1300` (mínimo observado) y `31700` (percentil 99)
   para capturar mejor la sensibilidad al tamaño de lote.

2. **Regenerar agregados**:
   ```bash
   python bench_out/construction_benchmark/update_aggregates.py
   ```
   (El script crea/actualiza las tablas y `summary_overview.json`.)

3. **Informe LaTeX**:
   ```bash
   cd bench_out/construction_benchmark
   pdflatex benchmark_report.tex
   ```

Ajusta los comandos según tu entorno (activar `.venv311`, etc.).
