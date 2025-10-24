# README 

>1. Guia completa para correr y entender el modulo de **remodelacion** del proyecto *Optimizacion Casas Ames Iowa Capstone*.
>2. Guia completa para entender el modulo de **training** del XGBoost.
---
# Módulo `optimization/remodel`

## 1) Que hace este modulo

Este modulo construye y resuelve un MIP en Gurobi que decide **qué cambios de remodelacion conviene hacer** a una casa para **maximizar la utilidad**:
utilidad = (precio_remodelado - costo_total) - costo_inicial.

* El **precio** lo predice un modelo XGBoost ya entrenado y encapsulado en un pipeline de preprocesamiento.
* Los **costos** de cambiar atributos estan definidos en tablas y reglas de negocio.
* El MIP respeta politicas como no degradar calidades, no crear features prohibidas, limites de presupuesto, etc.

---

## 2) Requisitos rapidos

* Python 3.11 recomendado
* Gurobi 10+ con licencia activa
* Windows, macOS o Linux
* Paquetes del archivo `requirements.txt` en la raiz del repo

### Instalacion en una terminal limpia

```bash
# crear venv
python -m venv .venv311
# activar
# Windows
.venv311\Scripts\activate
# macOS/Linux
# source .venv311/bin/activate

# instalar dependencias
pip install -r requirements.txt
```
---

## 3) Estructura del modulo y para que sirve cada archivo

Ruta base: `optimization/remodel/`

* `__init__.py`
  inicializador del paquete, sin logica de negocio.

* `benchmark_remodel.py`
  corre *benchmarks* de punta a punta en varios casos, guarda resultados en `bench_out/`.
  util para pruebas rapidas y para validar tiempos y brechas.

* `check_env.py`
  chequea que el entorno este OK, por ejemplo version de Python, paquetes clave, y licencia Gurobi.
  primer comando que conviene correr.

* `compat_sklearn.py`
  helpers para compatibilidad scikit-learn, por ejemplo wrappers de pipeline y transformaciones.

* `compat_xgboost.py`
  utilidades para extraer arboles y hojas del XGBoost y conectarlos con el MIP.

* `config.py`
  **archivo de configuracion central**: rutas de datos y modelos, seeds, defaults de ejecucion, y flags.
  si algo no corre por rutas, revisa aqui primero.

* `costs.py`
  tablas de costos y funciones para mapear **costo por cambio** de cada categoria, por ejemplo:
  +1 dormitorio, +1 baño, metros agregados, garage por auto, deck m2, etc.
  tambien contiene politicas por categoria.

* `features.py`
  define el set de **features modificables**, dominios y codificaciones, ademas de meta-datos usados por el MIP
  (por ejemplo calidades ordinales, binarios de garage, piscina, etc.).

* `gurobi_model.py`
  **corazon del modulo**. Construye el **MIP embedido** con la prediccion del XGBoost, agrega restricciones de negocio,
  arma la funcion objetivo y retorna un `gp.Model` listo para resolver.
  tambien contiene politicas de "no empeorar", PWL para deshacer el log si aplica, y flags de debug.

* `io.py`
  carga/guarda datos, lee la casa base por `pid`, exporta resultados de corrida, snapshots, y reportes en texto.

* `run_opt.py`
  **driver principal** para correr un caso: lee config, carga una casa base, construye el MIP, resuelve,
  e imprime un **resumen legible**: aumento de utilidad, precio base, precio remodelado, costos y lista de cambios.

* `utils.py`
  funciones auxiliares: manejo de seeds, formatos, timers, trazas, checks.

* `xgb_predictor.py`
  encapsula el pipeline de precio: preprocesa features, ejecuta el XGBoost y expone funciones para el MIP.

---

## 4) Flujo completo 

1. Configura rutas en `optimization/remodel/config.py`

   * ruta del CSV base con casas
   * ruta del modelo XGBoost guardado
   * carpeta de salida `bench_out/`
   * defaults de presupuesto, tier, etc.

2. Verifica entorno

```bash
python -m optimization.remodel.check_env
```

3. Corre un caso simple por `pid` y presupuesto

```bash
python -m optimization.remodel.run_opt --pid 534402170 --budget 50000
```

4. Revisa la salida en consola y los archivos en `bench_out/`.

5. (Opcional) Corre un benchmark sobre una lista de casos

```bash
python -m optimization.remodel.benchmark_remodel --n 0 --budget 50000
```

> Nota: Cuidado con la cantidad de casas a correr en este último punto, pus se demora bastante (aprox, 1 minuto por casa x 3 budgets)
---

## 5) Datos y modelos

* **Datos**: el CSV base está en    `data/processed/base_completa_sin_nulo.csv`

* **Modelo XGB**: carpeta bajo `models/xgb/...` con `model_xgb.joblib` y `booster.json`.
  La ruta se configura en `config.py`.

---

## 6) Salidas esperadas

En consola se verá algo como:

```
Optimal solution found (tolerance 1.00e-03)
Aumento de Utilidad: $220,671
precio casa base: $103,700
precio casa remodelada: $324,371
costos totales de remodelacion: $0

Cambios hechos en la casa
- Total Bsmt SF: base -> 816  ;  nuevo -> 264
- ...
```

Métricas y gráficos se guardan en `bench_out/` (al correr el archivo `benchmark_remodel.py`):

* CSV con resultados agregado por corrida
* Snapshot de cambios propuestos por atributo
* Logs con gap, tiempo y restricciones activas si debug esta ON

---
# Módulo `training`
--
##  Uso y referencias a la IA
El código fue echo con ayuda de la IA. A continuación links utilizados para la elaboración de partes del código:

* https://chatgpt.com/g/g-p-67e7192d867c8191b6cc57c7d70b885a-universidad-vale/shared/c/68f15667-a570-832e-a6dd-232c3b7b75cb?owner_user_id=user-WSMVGktdRc8FZkmHo76oEWUE
* https://chatgpt.com/g/g-p-67e7192d867c8191b6cc57c7d70b885a-universidad-vale/shared/c/68fa7f1e-dc9c-8329-b102-4779c19ece59?owner_user_id=user-WSMVGktdRc8FZkmHo76oEWUE
* https://chatgpt.com/g/g-p-67e7192d867c8191b6cc57c7d70b885a-universidad-vale/shared/c/68e5109e-b2b4-832b-8a8a-24b882354dd1?owner_user_id=user-WSMVGktdRc8FZkmHo76oEWUE
* https://chatgpt.com/g/g-p-67e7192d867c8191b6cc57c7d70b885a-universidad-vale/shared/c/68f4a28b-bd48-832e-b6ec-421c75ceb765?owner_user_id=user-WSMVGktdRc8FZkmHo76oEWUE
* https://chatgpt.com/g/g-p-67e7192d867c8191b6cc57c7d70b885a-universidad-vale/shared/c/68e5109e-b2b4-832b-8a8a-24b882354dd1?owner_user_id=user-WSMVGktdRc8FZkmHo76oEWUE
