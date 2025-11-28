#!/bin/bash

# Script para probar la Solución 1: Early Stopping más agresivo
# Reduce patience de 200 a 50 para detener el training más temprano

set -e

echo "=================================="
echo "  SOLUCIÓN 1: EARLY STOPPING"
echo "=================================="
echo ""
echo "Entrenando modelo con early stopping más agresivo..."
echo "patience: 50 (vs 200 anterior)"
echo ""

cd "$(dirname "$0")/.." || exit 1

# Crear directorio de salida
OUTPUT_DIR="models/xgb/completa_present_log_p2_early50"
mkdir -p "$OUTPUT_DIR"

# Ejecutar training
PYTHONPATH=. python3 src/train_xgb_es.py \
  --csv data/raw/df_final_regresion.csv \
  --target SalePrice_Present \
  --outdir "$OUTPUT_DIR" \
  --log_target \
  --patience 50

echo ""
echo "✅ Training completado"
echo ""
echo "Comparando resultados..."
echo ""

# Comparar métricas
echo "📊 MÉTRICA ORIGINAL (patience=200):"
python3 << 'EOF'
import json
with open("models/xgb/completa_present_log_p2_1800_ELEGIDO/metrics.json") as f:
    m = json.load(f)
    print(f"  MAPE test: {m['test']['MAPE_pct']:.2f}%")
    print(f"  MAE test:  ${m['test']['MAE']:,.0f}")
    print(f"  R² test:   {m['test']['R2']:.4f}")
EOF

echo ""
echo "📊 MÉTRICA NUEVA (patience=50):"
python3 << 'EOF'
import json
try:
    with open("models/xgb/completa_present_log_p2_early50/metrics.json") as f:
        m = json.load(f)
        print(f"  MAPE test: {m['test']['MAPE_pct']:.2f}%")
        print(f"  MAE test:  ${m['test']['MAE']:,.0f}")
        print(f"  R² test:   {m['test']['R2']:.4f}")
except FileNotFoundError:
    print("  ⚠️  No se encontró el archivo de métricas. Training aún en progreso...")
EOF

echo ""
echo "=================================="
echo "✅ Proceso completado"
echo "=================================="
