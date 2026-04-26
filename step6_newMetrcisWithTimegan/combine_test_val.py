"""
合并测试集和验证集以增加测试攻击数量
"""
import pandas as pd

# 加载测试集和验证集
test_df = pd.read_csv('../step5_timegan/output/test_with_attacks.csv', low_memory=False)
test_attack = pd.read_csv('../step5_timegan/output/test_attack_info.csv')

val_df = pd.read_csv('../step5_timegan/output/val_with_attacks.csv', low_memory=False)
val_attack = pd.read_csv('../step5_timegan/output/val_attack_info.csv')

print('='*80)
print('合并测试集和验证集')
print('='*80)

print('\n原始测试集:')
print(f'  飞行数: {test_df["flight"].nunique()}')
test_real_attacks = len(test_attack[test_attack['attack_type'] != 'none'])
print(f'  攻击数: {test_real_attacks}')

print('\n验证集:')
print(f'  飞行数: {val_df["flight"].nunique()}')
val_real_attacks = len(val_attack[val_attack['attack_type'] != 'none'])
print(f'  攻击数: {val_real_attacks}')

# 调整验证集的flight ID以避免冲突
max_test_flight_id = test_df['flight'].max()
print(f'\n调整验证集flight ID (偏移 {max_test_flight_id})...')
val_df['flight'] = val_df['flight'] + max_test_flight_id
val_attack['flight'] = val_attack['flight'] + max_test_flight_id

# 合并
combined_df = pd.concat([test_df, val_df], ignore_index=True)
combined_attack = pd.concat([test_attack, val_attack], ignore_index=True)

print('\n合并后的测试集:')
print(f'  总点数: {len(combined_df):,}')
print(f'  飞行数: {combined_df["flight"].nunique()}')
combined_real_attacks = len(combined_attack[combined_attack['attack_type'] != 'none'])
print(f'  攻击数: {combined_real_attacks}')
print(f'  攻击比例: {combined_real_attacks / len(combined_attack) * 100:.1f}%')

print('\n攻击类型分布:')
print(combined_attack['attack_type'].value_counts())

# 保存
combined_df.to_csv('../step5_timegan/output/test_with_attacks_combined.csv', index=False)
combined_attack.to_csv('../step5_timegan/output/test_attack_info_combined.csv', index=False)

print('\n✓ 保存合并后的数据:')
print('  ../step5_timegan/output/test_with_attacks_combined.csv')
print('  ../step5_timegan/output/test_attack_info_combined.csv')

print('\n使用方法:')
print('  修改 config.py 中的文件名为 test_with_attacks_combined.csv 和 test_attack_info_combined.csv')
