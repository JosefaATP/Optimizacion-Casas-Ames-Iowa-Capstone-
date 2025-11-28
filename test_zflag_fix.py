#!/usr/bin/env python3
import sys
import os

# Set up path
base_dir = r"c:\Users\valen\OneDrive - uc.cl\UNIVERSIDAD\2024-1\Bases de datos\Optimizacion-Casas-Ames-Iowa-Capstone-"
sys.path.insert(0, base_dir)
os.chdir(base_dir)

# Import and run
from optimization.remodel import run_opt
run_opt.main()
