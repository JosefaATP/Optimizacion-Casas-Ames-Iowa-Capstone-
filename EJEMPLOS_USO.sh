#!/bin/bash
# EJEMPLOS DE USO: Comparación de Predictores
# ==============================================

# EJEMPLO 1: Ejecución estándar
# Resuelve optimización + compara XGB vs Regresión automáticamente
./venv/bin/python -m optimization.remodel.run_opt --pid 526301100 --budget 80000

# EJEMPLO 2: Diferentes casa y budget
./venv/bin/python -m optimization.remodel.run_opt --pid 526350040 --budget 120000

# EJEMPLO 3: Especificar modelo de regresión custom
./venv/bin/python -m optimization.remodel.run_opt \
  --pid 526301100 \
  --budget 80000 \
  --reg-model /Users/tu_usuario/Desktop/mi_modelo_regresion.joblib

# EJEMPLO 4: Con límite de tiempo (para Gurobi solver)
./venv/bin/python -m optimization.remodel.run_opt \
  --pid 526301100 \
  --budget 80000 \
  --time-limit 300

# EJEMPLO 5: Todos los parámetros
./venv/bin/python -m optimization.remodel.run_opt \
  --pid 526301100 \
  --budget 80000 \
  --reg-model models/regression_model.joblib \
  --time-limit 300 \
  --basecsv data/raw/base_data.csv

# EJEMPLO 6: Si necesitas entrenar modelo nuevo primero
python3 training/train_regression_model.py  # Genera models/regression_model.joblib
./venv/bin/python -m optimization.remodel.run_opt --pid 526301100 --budget 80000

# EJEMPLO 7: Debug - Ver sensibilidades de XGBoost
./venv/bin/python -m optimization.remodel.run_opt \
  --pid 526301100 \
  --budget 80000 \
  --debug-xgb

# EJEMPLO 8: Con output redirecto a archivo
./venv/bin/python -m optimization.remodel.run_opt \
  --pid 526301100 \
  --budget 80000 > resultados_opt.txt 2>&1

# EJEMPLO 9: Ver solo la sección de comparación
./venv/bin/python -m optimization.remodel.run_opt \
  --pid 526301100 \
  --budget 80000 2>&1 | grep -A 10 "COMPARACIÓN"

# EJEMPLO 10: Correr varios PIDs seguidos
for pid in 526301100 526350040 526351010; do
  echo "Procesando PID: $pid"
  ./venv/bin/python -m optimization.remodel.run_opt --pid $pid --budget 100000
  echo "---"
done

# ARGUMENTOS DISPONIBLES
# =======================
# --pid INT                (requerido) ID de la casa
# --budget FLOAT           (requerido) Presupuesto en USD
# --reg-model STRING       (default: models/regression_model.joblib) Ruta al modelo
# --time-limit FLOAT       (opcional) Límite de tiempo para solver (segundos)
# --basecsv STRING         (opcional) Ruta alternativa al CSV base
# --debug-xgb             (flag) Imprime sensibilidades del XGBoost

# ESTRUCTURA DE OUTPUT ESPERADO
# =============================
# 
# [Inicio]
# Cargando casa base...
# Construyendo modelo de optimización...
# Resolviendo MIP...
# 
# [Resultados]
# Precio casa base: $XXX,XXX
# Precio remodelada: $XXX,XXX
# Δ Precio: $XX,XXX
# 
# [Detalles de cambios]
# Cambios recomendados por atributo...
# 
# [Comparación de Predictores]
# Precio base (XGB):           $XXX,XXX
# Precio remodelado (XGBoost): $XXX,XXX  (+X.X%)
# Precio remodelado (Regresión): $XXX,XXX  (+X.X%)
# 
# Diferencia XGBoost vs Regresión:
#   Absoluta: $XX,XXX
#   Porcentaje: ±X.X%
# 
# [Checks]
# Verificación de restricciones...
# 

# NOTAS IMPORTANTES
# ==================

# 1. Asegúrate de tener el modelo de regresión entrenado:
#    python3 training/train_regression_model.py

# 2. El modelo busca por default en models/regression_model.joblib
#    Si no existe, te pedirá entrenar uno nuevo

# 3. La regresión está entrenada en log(SalePrice_Present)
#    El deslogaritmización es automática (np.exp)

# 4. Si los precios de regresión parecen extraños:
#    - Verifica que los features estén alineados
#    - La alineación usa feature_names_in_ del modelo

# 5. Puedes desactivar comparación si el modelo está roto:
#    Rename models/regression_model.joblib
#    O crea un symlink a uno distinto

# TROUBLESHOOTING
# ===============

# ❌ "ModuleNotFoundError: No module named 'gurobi_ml'"
# → ./venv/bin/pip install gurobi-machinelearning

# ❌ "Model not found: models/regression_model.joblib"
# → python3 training/train_regression_model.py

# ❌ "ImportError: attempted relative import"
# → Usa: python3 -m optimization.remodel.run_opt (no python3 run_opt.py)

# ❌ Precios de regresión muy bajos/altos
# → Verifica alineación de features
# → Compara X_opt.columns vs reg_model.feature_names_in_

