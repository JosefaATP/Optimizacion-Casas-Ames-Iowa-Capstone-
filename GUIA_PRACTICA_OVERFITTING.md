# 🎯 Guía Práctica: Corregir Sobreajuste en XGBoost

## 📊 Situación Actual

Tu modelo XGBoost tiene **SOBREAJUSTE SEVERO**:

```
┌─────────────────────────────────────────────┐
│  Train MAPE: 2.34% ✅ (Perfecto)            │
│  Test MAPE:  7.20% ❌ (Malo)                │
│  Ratio:      3.08x ❌ (Severo)              │
└─────────────────────────────────────────────┘
```

---

## 🔧 Soluciones (En Orden de Facilidad)

### **1️⃣ OPCIÓN RÁPIDA: Early Stopping (5 minutos)**

**Idea**: Detener el training cuando el modelo deja de mejorar en datos de validación.

```bash
# Ejecutar con early stopping más agresivo
cd /Users/josefaabettdelatorrep./Desktop/PUC/College/Semestre\ 8/Taller\ de\ Investigación\ Operativa\ \(Capstone\)\ \(ICS2122-1\)/Optimizacion-Casas-Ames-Iowa-Capstone-/

PYTHONPATH=. python3 src/train_xgb_es.py \
  --csv data/raw/df_final_regresion.csv \
  --target SalePrice_Present \
  --outdir models/xgb/test_early50 \
  --log_target \
  --patience 50
```

**Qué pasa**: 
- El script entrena con validación interna
- Para cuando no hay mejora en 50 rondas
- Usa la mejor iteración encontrada

**Resultado esperado**: MAPE test de 6.0-6.5%

---

### **2️⃣ OPCIÓN MODERADA: Reducir Complejidad (15 minutos)**

**Idea**: Usar menos árboles y más regularización.

Crea un nuevo archivo: `src/config_reduced.py`

```python
from config import Config

# Copia la config original y ajusta:
cfg = Config()
cfg.xgb_params = {
    "n_estimators": 800,          # ← Reducido (de 1800)
    "learning_rate": 0.025,
    "max_depth": 5,
    "min_child_weight": 10,
    "subsample": 0.6,              # ← Reducido (de 0.7)
    "colsample_bytree": 0.6,       # ← Reducido (de 0.7)
    "reg_lambda": 4.0,             # ← Aumentado (de 2.0)
    "reg_alpha": 1.0,              # ← Nuevo (agregar L1)
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "n_jobs": -1,
    "random_state": 42,
}
```

Luego entrena:
```bash
PYTHONPATH=. python3 src/train_xgb_es.py \
  --csv data/raw/df_final_regresion.csv \
  --target SalePrice_Present \
  --outdir models/xgb/test_reduced \
  --log_target \
  --patience 50
```

**Resultado esperado**: MAPE test de 5.5-6.0%

---

### **3️⃣ OPCIÓN EXHAUSTIVA: Grid Search (2 horas, automático)**

Crea: `scripts/grid_search_xgb.py`

```python
#!/usr/bin/env python3
import argparse, json, os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from src.config import Config
from src.preprocess import infer_feature_types, build_preprocessor
from src.metrics import regression_report

# Parámetros a probar
param_grid = {
    'n_estimators': [600, 900, 1200],
    'max_depth': [3, 4, 5],
    'reg_lambda': [2.0, 4.0, 6.0],
    'subsample': [0.5, 0.6, 0.7]
}

# Generar todas las combinaciones
from itertools import product
configs = [dict(zip(param_grid.keys(), values)) 
           for values in product(*param_grid.values())]

print(f"Probando {len(configs)} combinaciones...")

df = pd.read_csv("data/raw/df_final_regresion.csv")
cfg = Config()

# ... resto del código de CV ...
# Reportar mejor configuración
```

**Resultado esperado**: MAPE test de 5.0-5.5%

---

## 📋 Checklist de Implementación

### Paso 1: Prueba Rápida (Ahora, 5 min)
```bash
cd '/Users/josefaabettdelatorrep./Desktop/PUC/College/Semestre 8/Taller de Investigación Operativa (Capstone) (ICS2122-1)/Optimizacion-Casas-Ames-Iowa-Capstone-/'

PYTHONPATH=. python3 src/train_xgb_es.py \
  --csv data/raw/df_final_regresion.csv \
  --target SalePrice_Present \
  --outdir models/xgb/test_quick \
  --log_target \
  --patience 50
```

✅ **Luego:**
```bash
# Ver métricas
cat models/xgb/test_quick/metrics.json
```

---

### Paso 2: Validación (10 min después)
```bash
# Crear script de comparación
python3 << 'EOF'
import json

print("\n📊 COMPARACIÓN DE MODELOS\n")

# Modelo original
with open("models/xgb/completa_present_log_p2_1800_ELEGIDO/metrics.json") as f:
    original = json.load(f)

# Modelo nuevo
with open("models/xgb/test_quick/metrics.json") as f:
    nuevo = json.load(f)

print("MÉTRICA                 ORIGINAL    NUEVO      MEJORA")
print("-" * 55)

mape_orig = original['test']['MAPE_pct']
mape_new = nuevo['test']['MAPE_pct']
mejora = ((mape_orig - mape_new) / mape_orig) * 100

print(f"MAPE Test            {mape_orig:7.2f}%  {mape_new:7.2f}%  {mejora:+.1f}%")

mae_orig = original['test']['MAE']
mae_new = nuevo['test']['MAE']
mejora = ((mae_orig - mae_new) / mae_orig) * 100

print(f"MAE Test             ${mae_orig:7,.0f}  ${mae_new:7,.0f}  {mejora:+.1f}%")

r2_orig = original['test']['R2']
r2_new = nuevo['test']['R2']
mejora = ((r2_new - r2_orig) / (1 - r2_orig)) * 100

print(f"R² Test                {r2_orig:6.4f}   {r2_new:6.4f}   {mejora:+.1f}%")

print("\n")
EOF
```

---

## 🎯 Métricas Objetivo

| Métrica | Actual | Objetivo | Alcanzable |
|---------|--------|----------|-----------|
| **MAPE Test** | 7.20% | < 6.0% | ✅ Sí |
| **MAE Test** | $21,224 | < $18,000 | ✅ Sí |
| **Ratio MAPE** | 3.08x | < 2.0x | ✅ Sí |
| **R² Test** | 0.9304 | > 0.94 | ⚠️ Quizá |

---

## 🎓 Entendimiento del Problema

```
¿POR QUÉ OCURRE EL SOBREAJUSTE?

Tienes 1800 árboles con learning_rate=0.025
        ↓
Esto = 1800 × 0.025 ≈ 45 unidades de "fuerza"
        ↓
Con esa capacidad, el modelo MEMORIZA el training set
        ↓
Aprende patrones reales PERO TAMBIÉN ruido específico
        ↓
El ruido NO está en el test set
        ↓
Por eso test MAPE es 3x peor
```

---

## ✅ Recomendación Final

**COMIENZA CON LA OPCIÓN 1** (Early Stopping):
- ⏱️ Toma 5 minutos
- 📊 Debería mejorar MAPE test a ~6%
- 🔧 Sin cambios de hiperparámetros

**SI NO ES SUFICIENTE**, prueba Opción 2:
- ⏱️ Toma 15 minutos
- 📊 Debería mejorar MAPE test a ~5.5%
- 🔧 Reduce complejidad

**SI QUIERES LO MEJOR**, usa Opción 3:
- ⏱️ Toma 2 horas
- 📊 Debería alcanzar MAPE test ~5%
- 🔧 Búsqueda exhaustiva

---

## 📞 ¿Necesitas Ayuda?

Los scripts y documentación están en:
- `ANALISIS_OVERFITTING_XGBOOST.md` - Análisis técnico
- `RESUMEN_OVERFITTING_Y_SOLUCIONES.md` - Plan de acción
- `scripts/analizar_overfitting.py` - Script de diagnóstico

**Gráficos generados:**
- `analisis/overfitting_analisis.png`
- `analisis/deterioro_metricas.png`

---

