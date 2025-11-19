# 🔍 Análisis de Sobreajuste (Overfitting) en XGBoost

**Fecha**: 18 de noviembre de 2025
**Modelo evaluado**: `models/xgb/completa_present_log_p2_1800_ELEGIDO/`

---

## 📊 Métricas del Modelo

### **TRAIN SET**
| Métrica | Valor |
|---------|-------|
| **RMSE** | $8,586 |
| **MAE** | $6,090 |
| **MAPE** | 2.34% ✅ Excelente |
| **R² Score** | 0.9947 ✅ Muy alto |

### **TEST SET**
| Métrica | Valor |
|---------|-------|
| **RMSE** | $35,901 |
| **MAE** | $21,224 |
| **MAPE** | 7.20% ⚠️ Moderado |
| **R² Score** | 0.9304 ⚠️ Bueno pero más bajo |

---

## 🚨 Diagnóstico de Sobreajuste

### **1. Brecha Train-Test (El Indicador Principal)**

```
MAPE Train → Test:    2.34% → 7.20%  (RATIO: 3.08x) ⚠️ PREOCUPANTE
MAE Train → Test:     $6,090 → $21,224 (RATIO: 3.48x) ⚠️ PREOCUPANTE  
R² Train → Test:      0.9947 → 0.9304 (DIFERENCIA: -6.43%) ⚠️ MODERADO
RMSE Train → Test:    $8,586 → $35,901 (RATIO: 4.18x) ⚠️ PREOCUPANTE
```

### **2. Asimetría de Residuos (Distribución sospechosa)**

| Métrica | Train | Test |
|---------|-------|------|
| **Skew** | 0.465 | 1.203 | ↑ Aumenta en test |
| **Kurtosis** | 6.97 | 12.77 | ↑ Colas muy pesadas en test |

**⚠️ Interpretación**: 
- El modelo aprende residuos **simétricos** en training
- En test, los residuos tienen **sesgo positivo** (predice más bajo de lo esperado en casos extremos)
- Las colas pesadas indican **outliers** no capturados en training

---

## 🔧 Hiperparámetros Actuales

```json
{
  "n_estimators": 1800,          ← 💰 ALTO
  "learning_rate": 0.025,        ← CONSERVADOR
  "max_depth": 5,                ← CONSERVADOR
  "min_child_weight": 10,        ← CONSERVADOR
  "subsample": 0.7,              ← MODERADO
  "colsample_bytree": 0.7,       ← MODERADO
  "reg_lambda": 2.0,             ← MODERADO
  "reg_alpha": 0.0               ← NO HAY L1
}
```

### **Análisis de los hiperparámetros:**

✅ **Bien calibrados**:
- `max_depth=5` → Árboles bajos, reduce complejidad
- `min_child_weight=10` → Evita split en nodos pequeños
- `learning_rate=0.025` → Aprendizaje lento y controlado
- `reg_lambda=2.0` → Regularización L2

⚠️ **Posible problema**:
- **`n_estimators=1800`** es ALTO
- Con `learning_rate=0.025` bajo, 1800 árboles puede estar capturando **ruido** en training
- El modelo tiene **capacidad para sobreajustar** aunque los hiperparámetros sean conservadores

---

## 📈 Indicadores de Sobreajuste

| Indicador | Valor | Severidad |
|-----------|-------|-----------|
| MAPE Gap (train→test) | 3.08x | **🔴 SEVERA** |
| MAE Gap (train→test) | 3.48x | **🔴 SEVERA** |
| RMSE Gap (train→test) | 4.18x | **🔴 SEVERA** |
| Kurtosis en test | 12.77 | **🟡 MODERADA** |
| R² gap | -6.43pp | **🟡 MODERADA** |

---

## 🎯 Conclusión

### **SÍ, EL MODELO ESTÁ SOBREAJUSTANDO** ✅ Confirmado

**Evidencia:**
1. ✅ El MAPE en train es **3x mejor** que en test
2. ✅ El MAE se **multiplica por 3.5** en test
3. ✅ Los residuos en test tienen **distribución muy diferente** (más asimétrica)
4. ✅ El modelo generaliza **menos bien** de lo que podría

---

## 💡 Recomendaciones de Mejora

### **Opción 1: Reducir complejidad (RECOMENDADO)**
```python
# Reducir n_estimators
"n_estimators": 800,  # de 1800 (reducción del 55%)

# O aumentar regularización
"reg_lambda": 5.0,     # de 2.0 (aumento del 150%)
"reg_alpha": 0.5,      # agregar L1 (nuevo)

# O aumentar subsampling
"subsample": 0.5,      # de 0.7 (mayor variancia entre árboles)
"colsample_bytree": 0.5, # de 0.7
```

### **Opción 2: Early Stopping (IDEAL)**
```python
# Usar validación interna durante training
# Detener cuando validation MAPE deje de mejorar

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[EarlyStopping(rounds=100, save_best=True)]
)
```

### **Opción 3: Cross-Validation + Ensemble**
```python
# Entrenar múltiples modelos con diferentes random_states
# Promediar predicciones para mejor generalización
```

---

## 📋 Plan de Acción Sugerido

### **Paso 1: Ajuste Rápido (10 min)**
- Reducir `n_estimators` de 1800 a 1000
- Aumentar `reg_lambda` de 2.0 a 4.0
- Entrenar y comparar test MAPE

### **Paso 2: Early Stopping (30 min)**
- Implementar validación interna
- Dejar que XGBoost auto-ajuste el número de árboles
- Esperar reducción en test MAPE de ~1-2%

### **Paso 3: Validación Cruzada (1 hora)**
- Entrenar con 5-fold CV
- Reportar media y desviación estándar de MAPE
- Verificar que gap train-test sea < 1.5x

---

## 🔗 Archivos Relevantes

- Métricas actuales: `models/xgb/completa_present_log_p2_1800_ELEGIDO/metrics.json`
- Meta del modelo: `models/xgb/completa_present_log_p2_1800_ELEGIDO/meta.json`
- Script de entrenamiento: `src/train_xgb_es.py` (usa early stopping)
- Diagnostico: `diagnostic_regression_vs_xgb.py`

---

## 📝 Notas

1. **El modelo NO es malo**: R² de 0.93 en test es bueno
2. **Pero hay potencial de mejora**: Reducir el gap train-test aumentaría confiabilidad
3. **Early stopping es tu amigo**: El script `train_xgb_es.py` ya lo implementa
4. **Considera el trade-off**: Mejor generalización vs. menor precisión en training

---

