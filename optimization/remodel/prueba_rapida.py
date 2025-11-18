# optimization/remodel/prueba_rapida.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization.remodel.run_opt import ejecutar_modelo

def prueba_casa_individual():
    """Prueba con una casa individual para verificar que funcione"""
    
    print("🚀 PRUEBA RÁPIDA - CASA INDIVIDUAL")
    print("="*50)
    
    # Usar un PID que sabemos que existe
    pid = 526301100
    presupuestos = [20000, 40000, 60000, 80000]
    
    for presupuesto in presupuestos:
        print(f"\n💰 Probando con presupuesto: ${presupuesto:,}")
        try:
            resultado = ejecutar_modelo(pid=pid, budget=presupuesto)
            if resultado:
                print(f"✅ ÉXITO - Aumento utilidad: ${resultado.get('aumento_utilidad', 0):,.0f}")
            else:
                print("❌ FALLÓ - No se obtuvo resultado")
        except Exception as e:
            print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    prueba_casa_individual()