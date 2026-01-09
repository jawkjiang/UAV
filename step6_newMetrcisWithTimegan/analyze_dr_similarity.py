"""
分析为什么所有模型的DR都相同
"""
import numpy as np
import pandas as pd
from pathlib import Path

print("="*80)
print("1. 检查不同模型的预测差异")
print("="*80)

# 加载不同模型的预测
models = ['lstm', 'cnn', 'tcn', 'gru', 'transformer']
predictions = {}

for model in models:
    data = np.load(f'output/{model}_test_predictions.npz')
    predictions[model] = {
        'y_pred': data['y_pred'],
        'y_true': data['y_true'],
        'y_prob': data['y_prob']
    }

# 检查预测分布
print("\n预测分布:")
for model in models:
    unique, counts = np.unique(predictions[model]['y_pred'], return_counts=True)
    print(f"{model.upper():12s}: ", end="")
    for val, count in zip(unique, counts):
        label = "攻击" if val == 1 else "正常"
        print(f"{label}={count:,} ({count/len(predictions[model]['y_pred'])*100:.2f}%)", end="  ")
    print()

# 检查模型之间的预测差异
print("\n" + "="*80)
print("2. 模型预测差异矩阵")
print("="*80)

print("\n窗口级预测不同的数量:")
for i, m1 in enumerate(models):
    for m2 in models[i+1:]:
        diff = (predictions[m1]['y_pred'] != predictions[m2]['y_pred']).sum()
        total = len(predictions[m1]['y_pred'])
        print(f"{m1.upper():12s} vs {m2.upper():12s}: {diff:4d} 个窗口不同 ({diff/total*100:.2f}%)")

# 检查攻击窗口的预测
print("\n" + "="*80)
print("3. 攻击窗口的预测情况")
print("="*80)

y_true = predictions['lstm']['y_true']
attack_mask = (y_true == 1)
attack_indices = np.where(attack_mask)[0]

print(f"\n总共{len(attack_indices)}个攻击窗口")
print(f"各模型在攻击窗口的预测:")

for model in models:
    attack_preds = predictions[model]['y_pred'][attack_mask]
    detected = (attack_preds == 1).sum()
    print(f"{model.upper():12s}: 检测到 {detected}/{len(attack_indices)} ({detected/len(attack_indices)*100:.1f}%)")

# 检查第一个检测的窗口
print("\n" + "="*80)
print("4. 检查每个模型首次检测到攻击的窗口位置")
print("="*80)

# 加载详细延迟信息
detailed_delays = pd.read_csv('output/time_aware_metrics/detailed_delays.csv')

print("\n每个攻击的首次检测延迟 (前20行):")
print(detailed_delays[['model', 'flight_id', 'attack_type', 'delay', 'detected']].head(20))

# 统计每个攻击被各模型检测的延迟
print("\n" + "="*80)
print("5. 同一攻击在不同模型中的检测延迟")
print("="*80)

# 按flight_id分组查看
unique_flights = detailed_delays['flight_id'].unique()[:5]  # 前5个攻击flights

for flight_id in unique_flights:
    attack_data = detailed_delays[detailed_delays['flight_id'] == flight_id]
    print(f"\nFlight {flight_id} ({attack_data.iloc[0]['attack_type']}):")
    for _, row in attack_data.iterrows():
        status = "✓检测到" if row['detected'] else "✗未检测"
        delay_str = f"{row['delay']:.3f}s" if row['detected'] else "N/A"
        print(f"  {row['model']:12s}: {status:6s}  延迟={delay_str}")

print("\n" + "="*80)
print("6. DR计算逻辑分析")
print("="*80)

# 加载总体指标
overall = pd.read_csv('output/time_aware_metrics/overall_metrics.csv')
print("\nDR@不同阈值的结果:")
print(overall[['model', 'DR@1s', 'DR@2s', 'DR@5s', 'ADD']].to_string(index=False))

# 计算每个模型有多少攻击在不同时间阈值内被检测
print("\n分析：")
for delta_t in [1, 2, 5]:
    print(f"\nΔt={delta_t}s:")
    for model in models[:3]:  # 只看前3个模型
        model_data = detailed_delays[detailed_delays['model'] == model]
        detected_within = model_data[(model_data['detected']) & (model_data['delay'] <= delta_t)]
        total_attacks = len(model_data)
        print(f"  {model.upper():12s}: {len(detected_within)}/{total_attacks} 攻击在{delta_t}s内检测到")
