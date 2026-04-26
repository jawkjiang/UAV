"""
增强测试集：提高攻击注入比例
将部分正常飞行转换为攻击飞行，增加测试攻击数量
"""
import pandas as pd
import numpy as np
import os

# 配置
TARGET_ATTACK_COUNT = 120  # 目标攻击数量（当前51，提升到120）
ATTACK_TYPES = ['step', 'drift', 'delay', 'takeover']

def augment_test_set():
    """增强测试集的攻击数量"""
    
    # 加载原始数据
    test_attack_info = pd.read_csv('../step5_timegan/output/test_attack_info.csv')
    
    print('='*80)
    print('测试集攻击增强')
    print('='*80)
    
    # 当前状态
    current_attacks = test_attack_info[test_attack_info['attack_type'] != 'none']
    normal_flights = test_attack_info[test_attack_info['attack_type'] == 'none']
    
    print(f'\n当前状态:')
    print(f'  攻击飞行: {len(current_attacks)}')
    print(f'  正常飞行: {len(normal_flights)}')
    print(f'  攻击比例: {len(current_attacks) / len(test_attack_info) * 100:.1f}%')
    
    # 计算需要转换的飞行数
    needed_attacks = TARGET_ATTACK_COUNT - len(current_attacks)
    
    if needed_attacks <= 0:
        print(f'\n✓ 已经有足够的攻击（{len(current_attacks)}个）')
        return
    
    if needed_attacks > len(normal_flights):
        print(f'\n⚠️  正常飞行不足，只能增加{len(normal_flights)}个攻击')
        needed_attacks = len(normal_flights)
    
    print(f'\n目标:')
    print(f'  需要增加: {needed_attacks}个攻击')
    print(f'  目标总数: {len(current_attacks) + needed_attacks}')
    print(f'  新攻击比例: {(len(current_attacks) + needed_attacks) / len(test_attack_info) * 100:.1f}%')
    
    # 随机选择正常飞行转换为攻击
    np.random.seed(42)
    selected_normal = normal_flights.sample(n=needed_attacks, random_state=42)
    
    # 为选中的飞行分配攻击类型（平均分配）
    new_attacks = []
    attacks_per_type = needed_attacks // len(ATTACK_TYPES)
    remainder = needed_attacks % len(ATTACK_TYPES)
    
    idx = 0
    for attack_type in ATTACK_TYPES:
        count = attacks_per_type + (1 if remainder > 0 else 0)
        remainder -= 1
        
        for i in range(count):
            if idx >= len(selected_normal):
                break
            
            flight_row = selected_normal.iloc[idx].copy()
            # 修改攻击类型
            flight_row['attack_type'] = attack_type
            # 设置攻击开始时间（在飞行中间，随机位置）
            # 假设每个飞行约150个点 * 0.156秒 ≈ 23.4秒
            attack_start_time = np.random.uniform(5, 15)  # 5-15秒之间
            flight_row['attack_start_time'] = attack_start_time
            
            new_attacks.append(flight_row)
            idx += 1
    
    new_attacks_df = pd.DataFrame(new_attacks)
    
    print(f'\n新增攻击分布:')
    print(new_attacks_df['attack_type'].value_counts())
    
    # 创建新的attack_info
    # 移除被转换的正常飞行，添加新攻击
    remaining_normal = normal_flights[~normal_flights['flight'].isin(selected_normal['flight'])]
    augmented_attack_info = pd.concat([current_attacks, new_attacks_df, remaining_normal], ignore_index=True)
    
    # 按flight排序
    augmented_attack_info = augmented_attack_info.sort_values('flight').reset_index(drop=True)
    
    print(f'\n最终测试集:')
    print(f'  总飞行数: {len(augmented_attack_info)}')
    final_attacks = augmented_attack_info[augmented_attack_info['attack_type'] != 'none']
    print(f'  攻击飞行: {len(final_attacks)}')
    print(f'  正常飞行: {len(augmented_attack_info) - len(final_attacks)}')
    print(f'  攻击比例: {len(final_attacks) / len(augmented_attack_info) * 100:.1f}%')
    
    print(f'\n攻击类型分布:')
    print(augmented_attack_info['attack_type'].value_counts())
    
    # 保存增强后的attack_info
    output_path = '../step5_timegan/output/test_attack_info_augmented.csv'
    augmented_attack_info.to_csv(output_path, index=False)
    print(f'\n✓ 保存增强后的attack_info: {output_path}')
    
    # 同时备份原始文件
    backup_path = '../step5_timegan/output/test_attack_info_original.csv'
    if not os.path.exists(backup_path):
        test_attack_info.to_csv(backup_path, index=False)
        print(f'✓ 备份原始attack_info: {backup_path}')
    
    print(f'\n使用方法:')
    print(f'  修改 data_loader.py 中的路径指向新文件：')
    print(f'  return os.path.join(STEP5_OUTPUT_DIR, "test_attack_info_augmented.csv")')
    
    return augmented_attack_info


if __name__ == '__main__':
    augment_test_set()
