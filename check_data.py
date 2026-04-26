import pandas as pd
import os

print("="*80)
print("DATA VALIDATION CHECK")
print("="*80)

# 1. Overall Metrics
print("\n1. OVERALL METRICS (step6)")
print("-"*80)
df_overall = pd.read_csv('step6_newMetrcisWithTimegan/output/time_aware_metrics/overall_metrics.csv')
print(df_overall[['model', 'DR@1s', 'DR@2s', 'DR@5s', 'DR@10s', 'ADD', 'MTBFA']].to_string())

# 2. Traditional Metrics
print("\n\n2. TRADITIONAL METRICS (step5)")
print("-"*80)
models = ['cnn', 'lstm', 'bilstm', 'gru', 'cnn_lstm', 'tcn', 'transformer']
trad_data = []
for m in models:
    path = f'step5_timegan/output/test_metrics_{m}.json'
    if os.path.exists(path):
        import json
        with open(path) as f:
            metrics = json.load(f)
            trad_data.append({
                'model': m,
                'precision': metrics.get('precision', None),
                'recall': metrics.get('recall', None),
                'f1': metrics.get('f1', None)
            })
    else:
        trad_data.append({'model': m, 'precision': None, 'recall': None, 'f1': None})

df_trad = pd.DataFrame(trad_data)
print(df_trad.to_string())

# 3. Per-Attack Metrics
print("\n\n3. PER-ATTACK METRICS (step6)")
print("-"*80)
for m in models:
    path = f'step6_newMetrcisWithTimegan/output/time_aware_metrics/{m}_per_attack_metrics.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"\n{m.upper()}:")
        print(df.to_string())
    else:
        print(f"\n{m.upper()}: NOT FOUND")

# 4. Check for data consistency
print("\n\n4. DATA CONSISTENCY CHECK")
print("-"*80)
print("\nIssue 1: GRU DR@10s")
gru_row = df_overall[df_overall['model'] == 'gru'].iloc[0]
print(f"  GRU DR@5s: {gru_row['DR@5s']}")
print(f"  GRU DR@10s: {gru_row['DR@10s']}")
print(f"  Expected: DR@10s should be >= DR@5s ✓" if gru_row['DR@10s'] >= gru_row['DR@5s'] else "  ERROR: DR@10s < DR@5s")

print("\nIssue 2: Traditional metrics completeness")
missing_trad = df_trad[df_trad['precision'].isna()]
if len(missing_trad) > 0:
    print(f"  Missing traditional metrics for: {missing_trad['model'].tolist()}")
else:
    print("  All traditional metrics available ✓")

print("\nIssue 3: Per-attack metrics availability")
for m in models:
    path = f'step6_newMetrcisWithTimegan/output/time_aware_metrics/{m}_per_attack_metrics.csv'
    exists = "✓" if os.path.exists(path) else "✗ MISSING"
    print(f"  {m}: {exists}")
