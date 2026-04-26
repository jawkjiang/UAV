"""
快速分析：提取FP窗口数并与FP事件数对比

目标：展示Precision-MTBFA悖论
"""
import json
import pandas as pd

# 读取传统指标（窗口级FP）
models = ['cnn', 'lstm', 'bilstm', 'gru', 'cnn_lstm', 'tcn', 'transformer']

traditional_metrics = {}
for model in models:
    with open(f'step5_timegan/output/test_metrics_{model}.json', 'r') as f:
        data = json.load(f)
        traditional_metrics[model] = {
            'precision': data['precision'],
            'recall': data['recall'],
            'f1': data['f1'],
            'fp_windows': int(data['false_positives']),  # 窗口级FP
            'tp_windows': int(data['true_positives'])
        }

# 读取时间感知指标（事件级FP）
time_aware = pd.read_csv('step6_newMetrcisWithTimegan/output/time_aware_metrics/overall_metrics.csv')

# 合并数据
results = []
for model in models:
    ta_row = time_aware[time_aware['model'] == model].iloc[0]
    
    trad = traditional_metrics[model]
    fp_windows = trad['fp_windows']
    fp_events = int(ta_row['n_false_alarms'])
    
    # 计算聚合率
    aggregation_ratio = fp_events / fp_windows if fp_windows > 0 else 0
    
    results.append({
        'Model': model.upper(),
        'Precision': f"{trad['precision']:.4f}",
        'FP Windows': fp_windows,
        'FP Events': fp_events,
        'Aggregation': f"{aggregation_ratio:.2f}",
        'MTBFA (h)': f"{ta_row['MTBFA']:.2f}",
        'DR@5s': f"{ta_row['DR@5s']:.4f}"
    })

# 转换为DataFrame
df = pd.DataFrame(results)

# 按Precision排序
df_precision_order = df.sort_values('Precision', ascending=False).reset_index(drop=True)
df_precision_order['Precision Rank'] = range(1, len(df_precision_order) + 1)

# 按MTBFA排序
df_mtbfa = df.copy()
df_mtbfa['MTBFA_val'] = df_mtbfa['MTBFA (h)'].astype(float)
df_mtbfa_order = df_mtbfa.sort_values('MTBFA_val', ascending=False).reset_index(drop=True)
df_mtbfa_order['MTBFA Rank'] = range(1, len(df_mtbfa_order) + 1)

# 输出结果
print("="*80)
print("Precision vs MTBFA Paradox Analysis")
print("="*80)
print("\n1. FP Aggregation (Window-level vs Event-level)")
print("-"*80)
print(df.to_string(index=False))

print("\n\n2. Precision Ranking vs MTBFA Ranking")
print("-"*80)
comparison = df_precision_order[['Model', 'Precision Rank']].merge(
    df_mtbfa_order[['Model', 'MTBFA Rank']], on='Model'
)
comparison['Rank Diff'] = abs(comparison['Precision Rank'] - comparison['MTBFA Rank'])
print(comparison.to_string(index=False))

print("\n\n3. Key Paradox Examples")
print("-"*80)
# 找出排名差异最大的例子
paradox_examples = comparison.nlargest(3, 'Rank Diff')
for _, row in paradox_examples.iterrows():
    model = row['Model']
    model_data = df[df['Model'] == model].iloc[0]
    print(f"\n{model}:")
    print(f"  Precision: {model_data['Precision']} (Rank #{row['Precision Rank']})")
    print(f"  MTBFA: {model_data['MTBFA (h)']}h (Rank #{row['MTBFA Rank']})")
    print(f"  FP Windows: {model_data['FP Windows']} → FP Events: {model_data['FP Events']}")
    print(f"  Aggregation Ratio: {model_data['Aggregation']}")

# 保存结果
import os
os.makedirs('paper_analysis/output/tables', exist_ok=True)
df.to_csv('paper_analysis/output/tables/precision_mtbfa_comparison.csv', index=False)
comparison.to_csv('paper_analysis/output/tables/precision_mtbfa_ranking.csv', index=False)

print("\n\n✅ Results saved to paper_analysis/output/tables/")
print("="*80)
