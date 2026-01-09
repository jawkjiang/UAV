"""
分析风险分数的合理性
"""

import pandas as pd
import numpy as np

print("="*80)
print("风险分数数据分析")
print("="*80)

# 1. 检查各攻击类型的实际参数
attack_types = ['drift_ramp', 'drift_sigmoid', 'delay', 'replay_same_hard', 
                'replay_other_soft', 'takeover_step', 'takeover_ramp', 'step']

for attack_type in attack_types:
    print(f"\n{'='*80}")
    print(f"攻击类型: {attack_type}")
    print('='*80)
    
    try:
        attack_info_path = f'../step3_multiModel/output/{attack_type}/cnn/test_attack_info.csv'
        df = pd.read_csv(attack_info_path)
        
        attacked_flights = df[df['attacked'] == True]
        print(f"总飞行数: {len(df)}")
        print(f"受攻击飞行数: {len(attacked_flights)}")
        
        if len(attacked_flights) > 0:
            print(f"\n示例攻击参数:")
            sample = attacked_flights.iloc[0]
            print(f"  Flight: {sample['flight']}")
            print(f"  Attack start: {sample['attack_start_time']}")
            print(f"  Attack params: {sample['attack_params'][:200]}...")
        else:
            print("  ⚠️ 无攻击样本！")
            
    except Exception as e:
        print(f"  ❌ 读取失败: {e}")

# 2. 检查生成的风险分数
print(f"\n{'='*80}")
print("生成的风险分数统计")
print('='*80)

risk_df = pd.read_csv('./output/power_grid_analysis/metrics6_advanced_risk.csv')

risk_summary = risk_df.groupby('attack_type').agg({
    'avg_risk_score': ['mean', 'min', 'max'],
    'total_attacks': 'mean'
}).round(2)

print("\n各攻击类型的风险分数:")
print(risk_summary)

# 3. 识别异常值
print(f"\n{'='*80}")
print("风险分数异常检测")
print('='*80)

for attack_type in risk_df['attack_type'].unique():
    attack_data = risk_df[risk_df['attack_type'] == attack_type]
    avg_risk = attack_data['avg_risk_score'].mean()
    
    status = "✅"
    if avg_risk > 1000:
        status = "❌ 过高"
    elif avg_risk == 0:
        status = "⚠️ 为零"
    elif avg_risk < 10 and attack_type not in ['drift_ramp', 'drift_sigmoid', 'takeover_step', 'takeover_ramp']:
        status = "⚠️ 过低"
    
    print(f"{status} {attack_type:20s}: {avg_risk:8.1f}")

print("\n" + "="*80)
