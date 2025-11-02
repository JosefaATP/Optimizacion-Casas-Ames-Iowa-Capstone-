# Estado del modelo de Construcción (MIP + XGB)

## 1) Implementado desde el PDF
- 7.1.x Lote y pisos: `x2 <= x1`, `GrLivArea = 1st + 2nd`, límites por lote y por porches/piscina/garage.
- 7.1.6 / 7.1.7: al menos 1 baño completo y 1 cocina en primer piso.
- 7.1.8: mínimos por piso (primer piso `ε1`, segundo piso si se activa; por defecto `ε1=450 ft²` para 1Fam).
- 7.2: conteos agregados (Full/Half/Kitchen) y vínculos a XGB.
- 7.3: límites por tipo de edificio (Bedrooms/Full/Half/Kitchen/Fireplaces).
- 7.5: garage (cars ↔ área, tipo y terminación consistentes).
- 7.7: techo (estilo/material, plan vs. real con indicadores fuertes y `Actual = γ·Plan`), exclusividad estilo/material.
- 7.8 / 7.9 / 11.10: área total, cotas de subáreas y `GrLivArea <= 0.8·LotArea`.
- 7.11: piscina (espacio, máximos y mínimos; `Pool QC = -1` si no hay piscina).
- 7.12 / 7.13: porches y deck (identidad `TotalPorch = sum(componentes)` y mínimos/máximos por tipo).
- 7.18: sótano (exposición, tipos 1/2 y sumatoria de áreas; consistencias con NA).
- 7.20: `AreaFoundation = 1st Flr SF` y `FA__{tipo} <= AreaFoundation`.
- 7.23: perímetros y área exterior (versión lineal con big‑M apretado y cota `AreaExterior <= Hext·(Uperim1+Uperim2)`).
- Enlace completo al XGB: `y_log = sumárboles(x)` y `y_price = PWL(exp(y_log) - 1)`.

## 2) Ajustes y cambios (razonados)
- Indicadores fuertes para techo (7.7.5) y “exterior seleccionado ⇒ Z = Plan”; evitan bilinealidad débil.
- Tightening de bounds: usa solo los árboles `best_iteration` del XGB (early stopping) para reducir big‑M y tamaño.
- Fijación robusta de OHE en X (ms zoning, street, alley, lot shape, land contour/slope, lot config, condition 1/2, functional, sale type/condition, neighborhood) con sum‑to‑one.
  - Alias tolerante para `Functional` ↔ `Functiono aplical`.
- Calidad/condición de garage: `Garage Qual/Cond = 4` si hay garage, `= -1` si “NA”.
- `Pool QC = -1` si no hay piscina.
- `Misc Val = 0` si `MiscFeature = No aplica`.
- Relación sótano ↔ fundación: `Total Bsmt SF <= ρ·AreaFoundation` (por defecto `ρ=1.0`), y `ρ_exposed` si hay exposición (p.ej. 1.1).

## 3) Extensiones extra al PDF (opcionales/parametrizables)
- Early stopping en entrenamiento XGB y en embed (usa solo los primeros `N` árboles efectivos).
- Perfil de solver (balanced/feasible/bound) y banderas `--fast/--deep`.
- `--xgbdir` para usar modelos alternos.
- `--outcsv` para guardar resultados por corrida (grid runs/reporte).
- `--bldg` para fijar tipo de edificio desde CLI; opcional `ct.eps1_by_bldg` para mínimos por tipo.

## 4) Cosas por seguir puliendo (y cómo mirarlas)
- Elasticidad de precio a área sobre rasante: si aparece mucho sótano vs. 1er piso, subir `ρ_exposed` solo un poco (1.05–1.15) y observar.
- Límites de habitaciones/baños: hoy vienen de nuestro diccionario por BldgType (no del PDF). Se pueden mover a `CostTables`.
- PWL de `exp`: aumentar puntos (40→80) si quieres que `y_price` y `[AUDIT] predict fuera` calcen aún más fino.

## 5) Chequeos al correr
- Ver `[COST-CHECK]` ⇒ `suma_terms == cost_model`.
- `[COST-CATEGORIES]` muestra selección efectiva de grupos y costo unitario.
- `X_input_after_opt.csv`: revisar que OHE exclusivas muestren un único `1` por grupo.
- Sótano vs. fundación: `Total Bsmt SF <= ρ(·)·AreaFoundation` y `Pool QC/Misc Val` consistentes con “No aplica”.

## 6) Cómo correr
```
python -m optimization.construction.run_opt --neigh "NWAmes" --lot 7000 --budget 250000 --profile feasible --fast --quiet
```
Opcionales:
- Usar XGB alterno: `--xgbdir models/xgb/with_es_test`
- Guardar CSV: `--outcsv bench_out/grid_results.csv`
- Modo edificios: `--bldg 1Fam` (defecto), `--bldg Duplex`, etc. y configurar `eps1_by_bldg` en `CostTables` si se desea.

