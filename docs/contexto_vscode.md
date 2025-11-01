# Contexto VS Code — Construcción Capstone

- MIP para diseñar una casa “desde cero” embebiendo un XGBoost que predice log(precio) y precio.
- Objetivo: maximizar `precio_predicho - costo_total` con un presupuesto como cota superior.
- Entradas fijas por CLI: `--neigh`, `--lot`, y `--budget`. Barrio y superficie de terreno quedan fijos.
- El modelo usa dummies one‑hot y nombres exactos de features para calzar con el pipeline del XGB.
- Salida esperada: resumen de decisiones (pisos/áreas/ambientes/roof), breakdown de costos, predicción XGB y CSV con el vector X pasado al XGB.

## Archivos Clave

- `optimization/construction/run_opt.py`: parsea args, arma bundle XGB, llama `build_mip_embed`, corre y reporta.
- `optimization/construction/gurobi_model.py`: arma el MIP, costos y constraints, integra el XGB y PWL exp.
- `optimization/construction/xgb_predictor.py`: wrapper XGB, traduce árboles a MIP y liga `x -> y_log -> y_price`.

## Ya Corregido

- 7.23 perímetro y fachada lineal, sin bilinealidad, con `P1`, `P2`, `AreaExterior` y gating por `Floor1/2`.
- `Exterior 1st/2nd` via one‑hot (y tie a X con `_tie_one_hot_to_x`).
- Barrio (`Neighborhood_*`) y `Lot Area` fijados a la entrada.
- Diagnóstico de infactibilidad: `DualReductions=0`, `computeIIS`, guarda `*_conflict.ilp` y `*_model.lp`.
- Auditoría: se escribe `X_input_after_opt.csv` para ver exactamente lo que entró al XGB.
- “House Summary”: imprime pisos, áreas, ambientes, roof y top de costos.

## Pendientes Importantes

- Etiquetar costos como términos lineales para imprimir `COST-BREAKDOWN` siempre (ya se almacena `_lin_cost_expr` y `_cost_terms`; revisar completitud de tablas de costo).
- PATCH B de min‑áreas y definiciones sumatorias (añadido: TotRms y mínimos por ambiente, respetando `x1+x2`).
- Revisar escalas si hay violaciones numéricas grandes, evaluar normalizar unidades de costo.
- Completar restricciones del PDF desde p.22 en adelante (ver bloque 7.XX que falte).
- Asegurar cotas de `y_log` y `y_price` dentro del dominio PWL (ya acotado por `np.linspace`, ajustar si el rango de tu modelo cambia).

## Cómo Correr Rápido

```bash
python -m optimization.construction.run_opt --neigh "NAmes" --lot 7000 --budget 250000
```

## Checklist de Pruebas Rápidas

- Ejecuta el comando anterior y confirma:
  - `[STATUS]` es OPTIMAL o TIME_LIMIT con `SolCount > 0`.
  - Se imprime “==== HOUSE SUMMARY ====” con pisos, áreas y roof elegido.
  - Se guarda `X_input_after_opt.csv` en la raíz del repo.
  - `COST-BREAKDOWN` muestra términos no nulos (al menos construcción y piscina si existe).
- Cambia `--neigh` (p.ej. `NWAmes`) y verifica que el one‑hot de barrio se fija correctamente en el `.lp`.
- Cambia `--lot` (p.ej. 9000) y observa que caps (0.6/0.5/0.8/0.2) se mueven en el `.lp` y en solución.
- Baja `--budget` (p.ej. 150000) y confirma que la restricción de presupuesto activa recorta áreas.
- Si hay infeasibilidad, confirma existencia de `*_conflict.ilp` y abre `*_model.lp` para inspección.

## Notas

- No cambiar nombres de columnas ni dummies usados por el XGB.
- Siempre ligar dummies con `_tie_one_hot_to_x` (evitar `getVarByName` para eso).
- Barrio y `Lot Area` van fijos; lo demás lo decide el modelo.
- Presupuesto es cota superior; no introducir slack oculto.

