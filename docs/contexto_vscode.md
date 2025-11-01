Proyecto: Construcción MIP + XGBoost (Ames)

Resumen de cambios y estado (actualizado por desarrollo)

- Objetivo: maximizar `precio_predicho - costo_total` con XGBoost embebido en MIP.
- Entradas fijas por CLI: `--neigh`, `--lot`, `--budget`. Flags: `--fast` (60s), `--deep` (900s).
- Auditoría: se guarda `X_input_after_opt.csv`, se imprime breakdown de costos (top 50) y diagnóstico IIS cuando aplica.

Extras vs. PDF (decisiones de implementación)

- One‑hot típicos fijados (no decisiones constructivas): `MS Zoning=RL`, `Street=Pave`, `Alley=No aplica`, `Lot Shape=Reg`, `LandContour=Lvl`, `LotConfig=Inside`, `LandSlope=Gtl`, `Condition1/2=Norm`, `Functional=Typ`, `Sale Type=WD`, `Sale Condition=Normal`.
- Fechas: `Year Built=2025`, `Year Remod/Add=2025`, venta `Year/Month Sold=2025/6` usando helper seguro (numérico u OHE) para evitar IIS.
- Exclusividades y ligaduras a X (one‑hot): `Heating`, `CentralAir`, `Electrical`, `PavedDrive`, `RoofStyle`, `RoofMatl`, `Exterior1st/2nd`, `Foundation`, `Fence`, `MiscFeature`, `Garage Type/Finish`, `HouseStyle` (con `Floor2 ≤ HS[2Story]`).
- Costeo: se evita doble conteo en obra gruesa.
  - Base construcción: se cobran solo `Remainder1`, `Remainder2` y `Bsmt Unf SF` con `construction_cost`.
  - Específicos por ambiente: `AreaKitchen`, `AreaFullBath`, `AreaHalfBath`, `AreaBedroom` (valores en `CostTables`).
  - Exterior: porches/deck/piscina por ft²; `Utilities` como lump‑sum; `PavedDrive`, `Electrical`, `Heating` como lump‑sum por categoría.
  - Techo: costo fijo por `Roof Matl`; opcional por `Roof Style` si se carga tabla.
  - Mampostería: `Mas Vnr` por ft² según tipo (área ligada dentro del MIP).
  - Garage: costo por área (`garage_area_cost`), calidades EX/condiciones EX si hay garage (sin sobrecargo adicional).
  - Fundación: por ft² según tipo (`foundation_cost_per_sf`).
  - MiscFeature: lump‑sum por categoría (tabla lista para poblar con el anexo).
  - Chimenea: costo adicional por `Fireplace EX` (por unidad), tal como se solicitó.
- Breakdown: imprime top 50 términos; si Gurobi no expone expr, usa fallback con términos registrados.
- Infeasibilidad: al detectar IIS, si `budget` está en el IIS, se imprime “Presupuesto insuficiente para la construcción mínima”.

Alineamiento con Sección 11 (principales)

- 11.1 HouseStyle 1Story/2Story + gating de `Floor2`.
- 11.2–11.6 Consistencias, mínimos, y límites (áreas, relación 2nd ≤ 1st, `Gr Liv Area`, mínimos de baños/cocina, límites por `BldgType`).
- 11.7 Techos: `PlanRoofArea` vs `ActualRoofArea` con factores γ y linealización de producto.
- 11.10/11.16 Caps globales de áreas.
- 11.11: `HalfBath ≤ FullBath`.
- 11.12–11.14 Piscina, porches/deck, sótano coherente con exposure NA.
- 11.20 Fundación: `AreaFoundation == 1st Flr SF` y variables `FA__{tipo}` ligadas por big‑M seguro.
- 11.23–11.23.1 Perímetro y fachada lineal (sin bilinealidad) con gating por `Floor1/2` y `AreaExterior = Hext*(P1+P2)`; repartición `W__ext1`.
- 11.xx Fence/MiscFeature como decisiones (one‑hot) y ligadas a X.

Pendientes / To‑Dos (checklist)

- [ ] Poblar `misc_feature_costs` con valores del anexo (Elev, Gar2, Othr, Shed, TenC).
- [ ] Validar que todo grupo OHE tenga exactamente un “1” en `X_input_after_opt.csv` (Fence, MiscFeature, Electrical, Heating, PavedDrive, Exterior1st/2nd, Foundation, RoofStyle/Matl, MS Zoning, etc.).
- [ ] Confirmar que `Garage Yr Built` se liga al año de construcción cuando `GarageType ≠ NA`; 0 si `NA` (agregar restricción si falta).
- [ ] Verificar que el breakdown incluya: Garage Area, Foundation (FA {tipo}), MiscFeature, Heating/Electrical/PavedDrive, Roof Matl, Exterior1st/2nd, Mas Vnr, Fireplace EX, Utilities, Porches/Deck/Pool y áreas de ambientes.
- [ ] Revisar que `Mas Vnr Area` y su tipo no generen simultáneamente “No aplica” y un tipo en el CSV (one‑hot bien atado).
- [ ] Confirmar reglas suaves de baños vs. dormitorios (proporción por crecimiento, con mínimos básicos).
- [ ] Performance: evaluar aumentar `TimeLimit` cuando se valide correctitud; `--fast` para pruebas rápidas.

Cómo correr

- Ejemplo: `python -m optimization.construction.run_opt --neigh "NAmes" --lot 7000 --budget 250000 --fast`

Archivos clave

- `optimization/construction/run_opt.py`: CLI, parámetros de solver, auditorías.
- `optimization/construction/gurobi_model.py`: MIP, costos, restricciones, tie con XGB.
- `optimization/construction/costs.py`: tablas de costos centralizadas.

Notas para el siguiente desarrollador

- Si un costo no aparece en el breakdown, revisar el nombre del var/one‑hot: el patrón usado es `NombreCategoria__Valor` y se agregan con `add("Etiqueta", coef, var)`. Usa `_tie_one_hot_to_x` para mantener CSV y MIP sincronizados.
- Si aparece más de una categoría activada en CSV para un grupo, agrega un bloque de mapeo como el de `Garage Type/Finish` para forzar igualdad entre X y el one‑hot interno.

