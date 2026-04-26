"""
分析测试集规模和模型性能差异
"""
import pandas as pd
import numpy as np

# 加载数据
overall = pd.read_csv('output/time_aware_metrics/overall_metrics.csv')
detailed = pd.read_csv('output/time_aware_metrics/detailed_delays.csv')

print('='*80)
print('测试集规模分析')
print('='*80)
print(f'\n总攻击数: {overall["n_attacks"].iloc[0]}')
print('⚠️  这是一个非常小的测试集！')

print('\n每种攻击类型的样本数:')
attack_counts = detailed[detailed['model'] == 'cnn']['attack_type'].value_counts()
for attack_type, count in attack_counts.items():
    print(f'  {attack_type:10s}: {count:2d} 样本')

print('\n' + '='*80)
print('所有模型的DR@5s')
print('='*80)
for _, row in overall.iterrows():
    detected = int(row['n_detected'])
    total = int(row['n_attacks'])
    print(f'{row["model"].upper():12s}: {row["DR@5s"]:.3f} ({detected}/{total}) = {detected/total*100:.1f}%')

print('\n' + '='*80)
print('ADD (平均检测延迟) 差异')
print('='*80)
for _, row in overall.sort_values('ADD').iterrows():
    print(f'{row["model"].upper():12s}: {row["ADD"]:.4f}s')

print('\n' + '='*80)
print('检测延迟分布差异')
print('='*80)
for model in ['cnn', 'lstm', 'bilstm', 'gru', 'cnn_lstm', 'tcn', 'transformer']:
    model_delays = detailed[detailed['model'] == model]['delay'].dropna()
    if len(model_delays) > 0:
        print(f'{model.upper():12s}: mean={model_delays.mean():.3f}s, '
              f'std={model_delays.std():.3f}s, median={model_delays.median():.3f}s, '
              f'min={model_delays.min():.3f}s, max={model_delays.max():.3f}s')

print('\n' + '='*80)
print('DR@不同时间阈值的差异')
print('='*80)
print(f'{"Model":<12s} {"DR@1s":>8s} {"DR@2s":>8s} {"DR@5s":>8s} {"DR@10s":>8s}')
print('-'*80)
for _, row in overall.iterrows():
    print(f'{row["model"].upper():<12s} {row["DR@1s"]:>8.3f} {row["DR@2s"]:>8.3f} '
          f'{row["DR@5s"]:>8.3f} {row["DR@10s"]:>8.3f}')

print('\n' + '='*80)
print('问题诊断')
print('='*80)
print("""
1. ❌ 测试集只有51个攻击 - 样本量太小！
   - 每种攻击类型只有12-13个样本
   - 统计显著性不足
   
2. ❌ DR@5s离散化问题:
   - 所有模型都检测到48个攻击（在5秒内）
   - 48/51 = 0.941 (94.1%)
   - 无法区分模型性能差异
   
3. ✅ 可以区分的指标:
   - ADD (平均检测延迟): 有0.03-0.4秒的差异
   - DR@1s: 有15-35%的显著差异
   - DR@2s: 有62-78%的显著差异
   
4. 💡 建议:
   - 使用更细粒度的时间阈值 (DR@0.5s, DR@1.5s, DR@3s)
   - 强调ADD和延迟分布差异
   - 增加测试集大小（如果可能）
   - 分析per-attack-type的性能差异
""")

print('\n' + '='*80)
print('Per-Attack-Type 性能差异')
print('='*80)
for attack_type in ['step', 'drift', 'delay', 'takeover']:
    print(f'\n{attack_type.upper()}:')
    for model in ['cnn', 'lstm', 'tcn', 'transformer']:
        model_attack = detailed[(detailed['model'] == model) & 
                               (detailed['attack_type'] == attack_type)]
        delays = model_attack['delay'].dropna()
        if len(delays) > 0:
            detected = len(delays)
            total = len(model_attack)
            avg_delay = delays.mean()
            print(f'  {model.upper():12s}: {detected}/{total} detected, ADD={avg_delay:.3f}s')
