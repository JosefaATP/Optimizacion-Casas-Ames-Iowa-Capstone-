#!/usr/bin/env python3
"""
Auditoría simple: corre optimización en múltiples properties y extrae ROI.
"""

import sys
import subprocess
import pandas as pd

def run_optimization(pid, budget, time_limit=60):
    """Corre optimización usando CLI y retorna output"""
    print(f"\n[AUDIT] Optimizando PID {pid}...")
    cmd = [
        sys.executable, "-m", "optimization.remodel.run_opt",
        "--pid", str(pid),
        "--budget", str(budget),
        "--time-limit", str(time_limit)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=time_limit+30, 
                              encoding='utf-8', errors='ignore')
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"
    except Exception as e:
        return f"[ERROR] {e}"

def extract_roi_error(output):
    """Extrae ROI y y_log error del output"""
    roi = None
    ylog_error = None
    
    lines = output.split('\n')
    for line in lines:
        # Buscar objetivo corregido (línea después de re-optimización)
        if 'Objetivo corregido:' in line:
            try:
                roi = float(line.split(':')[-1].strip())
            except:
                pass
        
        # Buscar y_log error
        if 'error=' in line and 'y_log' in line:
            try:
                parts = line.split('error=')
                error_str = parts[-1].split()[0]
                ylog_error = float(error_str)
            except:
                pass
        
        # Si no está en formato "Objetivo corregido", buscar "Best objective"
        if 'Best objective' in line and roi is None:
            try:
                roi = float(line.split('Best objective')[-1].split(',')[0].strip())
            except:
                pass
    
    return roi, ylog_error

def main():
    # Propiedades a analizar
    properties = [
        (526351010, 500000),
        (528328100, 500000),
        (527328062, 400000),
    ]
    
    results = []
    
    print(f"{'='*80}")
    print("AUDITORÍA DE OPTIMIZACIONES")
    print(f"{'='*80}")
    
    for pid, budget in properties:
        output = run_optimization(pid, budget, time_limit=90)
        
        # Extraer métricas
        roi, ylog_error = extract_roi_error(output)
        
        results.append({
            'PID': pid,
            'Budget': f"${budget:,.0f}",
            'ROI (USD)': f"${roi:,.2f}" if roi else "N/A",
            'ROI %': f"{(roi/budget)*100:.1f}%" if roi and budget else "N/A",
            'y_log error': f"{ylog_error:.6f}" if ylog_error else "N/A"
        })
        
        print(f"\n[RESULT] {pid}")
        print(f"  ROI: ${roi:,.2f}" if roi else f"  ROI: N/A")
        print(f"  ROI %: {(roi/budget)*100:.1f}%" if roi and budget else "  ROI %: N/A")
        print(f"  y_log error: {ylog_error:.6f}" if ylog_error else "  y_log error: N/A")
    
    # Tabla final
    print(f"\n\n{'='*80}")
    print("RESUMEN")
    print(f"{'='*80}")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
