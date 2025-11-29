import pandas as pd
import random

# Load dataset
df = pd.read_csv('data/processed/base_completa_sin_nulos.csv')
print(f'Total casas en dataset: {len(df)}')

# Get 10 random PIDs
pids = df['PID'].values.tolist()
selected_pids = random.sample(pids, min(10, len(pids)))

print('\n10 casas seleccionadas al azar:')
for i, pid in enumerate(selected_pids, 1):
    row = df[df['PID'] == pid].iloc[0]
    price = row.get('SalePrice', '?')
    if isinstance(price, (int, float)):
        print(f'{i}. PID: {pid:12} | SalePrice: ${price:>10,.0f}')
    else:
        print(f'{i}. PID: {pid:12} | SalePrice: {price}')

# Save to file for easy reference
with open('test_pids.txt', 'w') as f:
    for pid in selected_pids:
        f.write(f'{pid}\n')

print(f'\n✅ PIDs saved to test_pids.txt')
print(f'Comando para probar:')
print(f'python -m optimization.remodel.run_opt --pid <PID> --budget 500000')
