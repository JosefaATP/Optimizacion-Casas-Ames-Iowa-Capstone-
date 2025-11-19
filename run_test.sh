#!/bin/bash
# Script para ejecutar la optimización y guardar output

cd "/Users/josefaabettdelatorrep./Desktop/PUC/College/Semestre 8/Taller de Investigación Operativa (Capstone) (ICS2122-1)/Optimizacion-Casas-Ames-Iowa-Capstone-"

echo "Iniciando optimización..."
echo "PID: 526301100, Budget: 80000"
echo ""

./venv/bin/python -m optimization.remodel.run_opt --pid 526301100 --budget 80000

echo ""
echo "Optimización completada."
