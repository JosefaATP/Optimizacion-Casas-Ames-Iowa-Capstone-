import argparse, subprocess, json
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='data/processed/base_completa_sin_nulos.csv')
    ap.add_argument('--xgbdir', default='models/xgb/with_es_test')
    ap.add_argument('--outcsv', default='bench_out/grid_full.csv')
    ap.add_argument('--budgets', default='200000,500000,700000')
    ap.add_argument('--profile', default='feasible')
    ap.add_argument('--fast', action='store_true', default=False)
    ap.add_argument('--limit-neigh', type=int, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, sep=None, engine='python')
    # Columna LotArea/Lot Area
    lot_col = 'Lot Area' if 'Lot Area' in df.columns else ('LotArea' if 'LotArea' in df.columns else None)
    if lot_col is None:
        raise ValueError('No se encontró columna Lot Area/LotArea en el CSV')
    # Quartiles globales
    q = df[lot_col].quantile([0.25, 0.5, 0.75]).tolist()
    lots = [int(round(v)) for v in q]
    # Vecindarios únicos
    ncol = 'Neighborhood' if 'Neighborhood' in df.columns else None
    if ncol is None:
        raise ValueError('No se encontró columna Neighborhood en el CSV')
    neighs = sorted(df[ncol].dropna().unique().tolist())
    if args.limit_neigh:
        neighs = neighs[:args.limit_neigh]
    budgets = [int(b) for b in args.budgets.split(',') if b.strip()]

    py = Path('.venv311/Scripts/python.exe') if Path('.venv311/Scripts/python.exe').exists() else 'python'
    Path(Path(args.outcsv).parent).mkdir(parents=True, exist_ok=True)
    for n in neighs:
        for b in budgets:
            for l in lots:
                cmd = [str(py), '-m', 'optimization.construction.run_opt',
                       '--neigh', str(n), '--lot', str(l), '--budget', str(b),
                       '--profile', args.profile, '--quiet', '--outcsv', args.outcsv,
                       '--xgbdir', args.xgbdir]
                if args.fast:
                    cmd.append('--fast')
                print('RUN', ' '.join(cmd))
                try:
                    subprocess.run(cmd, check=False)
                except Exception as e:
                    print('WARN: fallo corrida', n, l, b, e)

if __name__ == '__main__':
    main()
