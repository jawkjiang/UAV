import pandas as pd

df = pd.read_csv('output/time_aware_metrics/overall_metrics_v2.csv')
print('MTBFA修复后的值:')
print('='*60)
for _, row in df.iterrows():
    mtbfa_min = row['mtbfa'] * 60
    mtbfa_sec = row['mtbfa'] * 3600
    print(f'{row["model"]:<12} MTBFA={row["mtbfa"]:.4f}h = {mtbfa_min:.2f}min = {mtbfa_sec:.1f}s')
