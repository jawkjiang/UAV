"""
Compare step3b (mixed attacks) vs step3 (single attacks) results
"""
import pandas as pd
import json
import os

print('='*80)
print('STEP3 vs STEP3B COMPARISON')
print('='*80)

# Check if step3 results exist
step3_path = '../step3_multiModel/output'
step3b_path = './output'

if not os.path.exists(step3_path):
    print(f'\n⚠️  Step3 results not found at {step3_path}')
    print('Cannot perform comparison.')
    exit(0)

# Load step3b summary
step3b_summary = pd.read_csv(f'{step3b_path}/overall_summary.csv')

print('\n' + '='*80)
print('STEP3B RESULTS (Mixed Attack Training)')
print('='*80)
print(step3b_summary.to_string(index=False))

print('\n' + '='*80)
print('STEP3B - PER-ATTACK BREAKDOWN (CNN model)')
print('='*80)

with open(f'{step3b_path}/cnn/per_attack_metrics.json', 'r') as f:
    cnn_per_attack = json.load(f)
    
print(f"{'Attack Type':<20} {'ROC-AUC':>10} {'F1':>10} {'Precision':>10} {'Recall':>10} {'N_samples':>10}")
print('-'*80)
for attack, metrics in cnn_per_attack.items():
    print(f"{attack:<20} {metrics['auc_roc']:>10.4f} {metrics['f1_score']:>10.4f} "
          f"{metrics['precision']:>10.4f} {metrics['recall']:>10.4f} {metrics['n_samples']:>10}")

print('\n' + '='*80)
print('TEST SET COMPOSITION')
print('='*80)

test_info = pd.read_csv(f'{step3b_path}/cnn/test_attack_info.csv')
attack_counts = test_info['attack_type'].value_counts()
print(attack_counts)

print(f'\n⚠️  CRITICAL ISSUE: Only {len(attack_counts)-1} attack types in test set!')
print(f'   Missing attack types: delay, takeover_step, takeover_ramp')
print(f'   Each attack type has only 1 flight - insufficient for robust evaluation!')

# Check step3 benchmark if available
step3_benchmark = f'{step3_path}/../benchmark_output/benchmark_summary.csv'
if os.path.exists(step3_benchmark):
    print('\n' + '='*80)
    print('STEP3 RESULTS (Single Attack Training - for reference)')
    print('='*80)
    
    step3_df = pd.read_csv(step3_benchmark)
    
    # Filter for the 6 attack types we're using
    attack_types = ['step', 'drift_ramp', 'drift_sigmoid', 'delay', 'takeover_step', 'takeover_ramp']
    
    for attack in attack_types:
        attack_results = step3_df[step3_df['attack_type'] == attack]
        if len(attack_results) > 0:
            print(f'\n{attack.upper()}:')
            print(attack_results[['model_type', 'test_auc_roc', 'test_f1']].to_string(index=False))
        else:
            print(f'\n{attack.upper()}: No results found')

print('\n' + '='*80)
print('ANALYSIS SUMMARY')
print('='*80)
print('''
Step3b训练策略：
- 每个模型训练时看到所有6种攻击类型
- 训练集：73个被攻击航班，50%比例，6种攻击均匀分布
- 目标：通过混合攻击训练提高泛化能力

当前问题：
1. ❌ 测试集不完整：只测试了3/6种攻击类型
2. ❌ 测试样本不足：每种攻击只有1个航班
3. ❌ 无法验证对delay和takeover系列的检测能力
4. ⚠️  完美的性能指标（ROC-AUC=1.0）缺乏说服力

建议：
1. 重新划分数据集，确保所有6种攻击在测试集都有>=3个航班
2. 增加测试集比例（从9.4%提高到15-20%）
3. 使用分层采样确保稀有攻击类型被充分测试
4. 与step3结果对比，验证混合训练是否真的提升了泛化能力
''')
