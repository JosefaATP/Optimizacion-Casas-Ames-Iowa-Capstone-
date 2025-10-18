# Optimizacion de remodelacion con XGBoost + Gurobi


Este modulo arma un MIP que maximiza **rentabilidad = precio_estimado - costo_remodelacion - costo_inicial**.


- **precio_estimado**: predicho por XGBoost embebido via gurobi-ml.
- **costo_remodelacion**: suma de costos de acciones/ajustes.
- **costo_inicial**: costo de la casa base sin cambios (constante para el caso base, solo afecta el reporte).


## flujo rapido
1. Ajusta rutas y parametros en `config.py`.
2. Define costos base/acciones en `costs.py` (o conecta tu fuente).
3. Revisa y ajusta features y bounds en `features.py`.
4. Revisa/edita restricciones en `constraints.py` segun tu PDF de modelo matematico.
5. Ejecuta:


```bash
python -m optimization.remodel.run_opt \
--pid 123456 \
--budget 40000 \
--method embed # o "iter"