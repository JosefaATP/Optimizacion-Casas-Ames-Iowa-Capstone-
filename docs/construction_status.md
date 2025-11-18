# Estado del Modelo de Construcción (MIP + XGB)

**Objetivo**: Maximizamos `precio_predicho - costo_total` embebiendo un XGBoost (target en log1p) dentro del MIP.

**Caminos clave**
- `optimization/construction/gurobi_model.py`: arma el MIP, restricciones, costo y XGB embebido.
- `optimization/construction/xgb_predictor.py`: carga el pipeline, orden de features, árboles y unión `x -> y_log -> y_price`.
- `optimization/construction/run_opt.py`: CLI, perfiles de solver, auditorías y CSV.
- `optimization/construction/costs.py`: tablas y parámetros de costos (configurables).

**Cómo Correr**
- Caso único: `python -m optimization.construction.run_opt --neigh "NWAmes" --lot 7000 --budget 250000 --profile feasible --quiet`
- Guardar resultados: añadir `--outcsv bench_out/run.csv`
- Usar modelo XGB alterno: `--xgbdir models/xgb/with_es_test`
- Grid: `scripts\run_grid.py --outcsv bench_out\grid.csv --budgets 250000,500000 --profile feasible --sample-neigh 10 --seed 123`

**Qué Mirar en el Output**
- `[COST-CHECK]` la suma de términos debe coincidir con `cost_model`.
- `[COST-CATEGORIES]` confirma la selección efectiva por grupos (Roof, Heating, Exterior, etc.).
- `X_input_after_opt.csv`: vector X exacto que entró al XGB (útil para revisar OHE exclusivas).
- `[AUDIT] y_log_in vs y_log_out`: hoy estamos verificando un delta casi constante (ver “Trabajo Actual”).
- `*_model.lp` e `*_conflict.ilp` para diagnóstico si hay infeasibilidad.

**Restricciones del PDF Implementadas**
- 7.1.x Lote y pisos
  - `x2 <= x1` (7.1.2), `GrLivArea = 1st + 2nd` (7.1.3), `1st + porches + pool <= LotArea` (7.1.1).
  - Mínimos por piso (7.1.8): `1st >= ε1·Floor1`, `2nd >= ε2·Floor2`; parámetros en `CostTables` (`eps_floor_min_first`, etc.).
- 7.4 Casa con 1 o 2 pisos
  - Binarias `IsOneStory/IsTwoStory` con `IsOneStory + IsTwoStory = 1`, `Floor2 = IsTwoStory` y `IsTwoStory == HouseStyle_2Story`.
- Exclusividades OHE
  - Además de los grupos existentes, `MS SubClass_*` ahora tiene `∑ = 1` para mantener consistencia con el PDF.
- 7.2 Conteos totales y vínculos
  - `Full/Half/Kitchen = sum por piso` y `HalfBath <= FullBath` (11.11). Enlaces a X si existen.
- 7.3 Límites por BldgType
  - Caps opcionales de `Bedrooms/Full/Half/Kitchen/Fireplaces` por tipo (si `ct.bldg_caps`).
  - Kitchens en 1Fam por tamaño (si `ct.kitchen_by_area_thresholds`), o guard‑rail `Kitchen ≤ 1` en 1Fam.
- 7.5 Garage
  - Área vs. autos `150·Cars ≤ GarageArea ≤ 250·Cars`. Tipo/terminación consistentes; `GarageCars ≥ 1` si hay.
  - `Garage Yr Built = Year Built` si hay garage; `=0` si NA. `Garage Qual/Cond = 4` si hay; `=-1` si NA.
- 7.7 Techo (plan vs. real)
  - `PlanRoofArea = PR1 + PR2` con on/off por piso; `ActualRoofArea = γ(s,mat)·Z` (indicadores fuertes, 7.7.5).
- 7.8 / 7.9 / 11.10 Caps y sumas globales
  - `TotArea = 1st + 2nd + Bsmt` (7.8), `1st ≤ 0.6·Lot`, `2nd ≤ 0.5·Lot`, `Bsmt ≤ 0.5·Lot` (7.9),
    `GrLivArea ≤ 0.8·Lot` (11.10), `GarageArea ≤ 0.2·Lot`.
- 7.10 Baños vs dormitorios
  - `3·FullBath ≤ 2·Bedrooms` (tope razonable alineado con la versión previa del modelo/PDF).
- 7.11 Piscina
  - Espacio/máx/mín por lot; `Pool QC = -1` si no hay piscina.
- 7.12 / 7.13 Porches/Decks
  - `Total Porch SF = suma componentes`, mínimos funcionales y ahora también cotas superiores proporcionales al lote por tipo (`Open ≤ 0.10·LotArea`, `Screen ≤ 0.05·LotArea`, etc.).
- 7.15 Exterior 1st/2nd material
  - Igualamos explícitamente `Exterior1st == Exterior2nd` (y mantenemos la bandera `SameMaterial` para trazabilidad).
- 7.16 Enchapados (Masonry Veneer)
  - `MasVnrType` OHE, `MasVnrArea ≤ fmas·AreaExterior1st`, `=0` si `None`, mínimo si usado, `MvProd_t` por tipo y nueva cota `MasVnrArea ≤ TotalArea`.
- 7.18 Sótano
  - `Total Bsmt SF ≤ ρ_base·AreaFoundation` y relajación `ρ_exposed` si hay exposición Gd/Av/Mn`, además fijamos `BsmtUnfSF = 0` para respetar el supuesto de obra nueva 100% terminada.
- 7.20 Fundación
  - `AreaFoundation = 1st Flr SF` y descomposición `FA__{tipo}` con big‑M seguro; `Slab/Wood ⇒ BsmtExposure=NA` y ahora también `BsmtExposure=NA ⇒ Slab ∨ Wood`.
- 7.21 Caps por piso de áreas por ambiente
  - Límite por `Floor1/Floor2` para `AreaBedroom/FullBath/HalfBath/Kitchen/Other`, y las *cantidades* del segundo piso (`Bedroom2`, `FullBath2`, `Kitchen2`, etc.) ahora se apagan automáticamente con `Floor2 = 0`.
- 7.22 Áreas de sala/otros y TotRms
  - `AreaOther = AreaOther1 + AreaOther2`, `OtherRooms = OtherRooms1 + OtherRooms2` y `TotRms AbvGrd = Bedrooms + FullBath + HalfBath + OtherRooms` como en el PDF.

**Calidades (nuevo ajuste)**
- El límite inferior “Average hacia arriba” se aplica solo a calidades que tienen costos definidos en `ct` (p.ej. `Heating QC`, `Fireplace Qu`, `Pool QC`, `Bsmt Cond`). Otras calidades quedan libres (ya no se fijan a Ex ni se les fuerza LB).

**Guard-rails por barrio**
- HouseStyle ya no se fija al `base_row`; es decisión del modelo.
- Para atributos de área accionables (`1st/2nd Flr SF`, `Total Bsmt SF`, `Garage Area`, `Bedroom/Full/Half/Kitchen AbvGr`, `Fireplaces`, `Mas Vnr Area`, `Gr Liv Area`), se impone como piso el percentil 10 del barrio (reforzado con el mínimo observado).
- Para categorías (`Heating`, `Electrical`, `PavedDrive`, `Exterior1st/2nd`, `Foundation`, `Roof Style`, `Roof Matl`, `Garage Finish`), se fija la modalidad más frecuente del barrio como cota inferior (se elige la moda del vecindario).
- 7.23 Perímetro y fachada (lineal)
  - `P1/P2` con on/off por piso, `P1 ≥ P2` si hay segundo piso; versión lineal segura sin bilinealidad.
- 11.3 Sumas de áreas agregadas
  - `AreaKitchen/AreaBedroom = sum por piso`.
- 11.20 Áreas mínimas por ambiente
  - `AreaFullBath ≥ 40·FullBath`, `AreaHalfBath ≥ 20·HalfBath`, `AreaKitchen ≥ 75·Kitchen`, `AreaBedroom ≥ 70·Bedrooms`.

**Restricciones del PDF Faltantes o en Revisión**
- Detalles finos posteriores a 7.23 (si el PDF incluye variantes no reflejadas aún).
- Reglas adicionales de circulación/interiores si el PDF las especifica (no estructurales al precio/costo por ahora).
- Validar valores exactos de γ(s,mat) para techo con el anexo (hoy están parametrizados).

**Cambios vs. PDF (y por qué)**
- 7.23 Perímetros: versión lineal con big‑M seguro (evitar bilinealidad y tiempos altos) conservando límites geométricos.
- 7.7 Techo: indicadores fuertes para `Z = Plan` cuando se elige estilo/material (fortalece LP y acelera).
- Caps razonables (7.9/11.10) vinculados a `Lot Area` para estabilidad numérica y realismo urbano.

**Nuevas Restricciones Implementadas (no literales del PDF)**
- Guard‑rail por defecto `Kitchen ≤ 1` para 1Fam si no se supera umbral de área (desactivable).
- `HalfBath ≤ FullBath` (11.11), para evitar combinaciones no plausibles.
- Bound‑tightening desde umbrales de árboles XGB (reduce UB efectivos, baja M grandes).

**Cómo Funciona el XGB Embebido**
- Se carga el pipeline entrenado (`pre` + `xgb`) y se usa el orden de columnas del `ColumnTransformer` (numérico) como orden del Booster.
- Se agregan restricciones por nodo de árbol con splits estrictos (izquierda si `x < thr`, derecha si `x ≥ thr`).
- `y_log` es la suma de hojas; `y_price` se obtiene con una PWL de `expm1(y_log)` en rango razonable.
- Auditorías imprimen `y_log_in` (embed) vs `y_log_out` (pipeline fuera) y guardan `X_input_after_opt.csv`.

**Trabajo Actual**
- Verificar que el delta `y_log_in − y_log_out` sea casi constante en muchos barrios/lotes/budgets (mini‑grid en curso).
- Si es constante, calibrar un offset `b0` (constante) en `y_log` para que “adentro=afuera” sin tocar ruteo ni restricciones.
- Revisar/afinar tramos del PWL si fuera necesario (hoy 80 puntos en [10.3, 13.8]).

**Posibles Mejoras**
- Calibración de intercepto `b0` para alinear exactamente adentro/afuera (evita sesgos en utilidad).
- Escalado de unidades/costos si vemos coeficientes muy dispares (mejor estabilidad numérica).
- Expandir `bldg_caps` y `eps1_by_bldg` desde el PDF/anexo para tipologías adicionales.
- Tests pequeños de consistencia (p.ej., `X_input_after_opt.csv` con un validador simple de OHE exclusivas).
- Micro‑perfiles de Gurobi por presupuesto (estrategia `feasible→bound`) en grids largos para mejorar GAP/tiempos.

**Preocupaciones**
- siempre hace la casa mínima (o casi siempre)
- No estoy segura si está prediciendo bien
- intenté con el xgb de ignacio, pero no quedo conforme con resultado (agregué early stopping o sino el gap era mas de 1000%). Revisar tema XGB. 

**Referencias de Código (útiles)**
- Modelo MIP: `optimization/construction/gurobi_model.py`
- XGB/árboles: `optimization/construction/xgb_predictor.py`
- Costos: `optimization/construction/costs.py`
- CLI/Auditoría: `optimization/construction/run_opt.py`
