"""
步骤4：分层注入攻击
Train: 8%, Val: 15%, Test: 20%，确保测试集充分
"""
import sys
sys.path.append('../step3b_multiModelGeneral')

import pandas as pd
import numpy as np
import json
from pathlib import Path
import config
from multi_attack_injector import MultiAttackInjector

# 定义攻击类型 (映射到MultiAttackInjector的格式)
ATTACK_TYPES = ['step', 'drift', 'takeover', 'delay']

# 攻击类型到参数的映射
ATTACK_TYPE_PARAMS = {
    'step': {},
    'drift': {'profile': 'ramp'},
    'takeover': {'offset_profile': 'step'},
    'delay': {}
}

def inject_attacks_for_split(split_name, df, attack_ratio):
    """
    为单个split注入攻击
    
    Args:
        split_name: 'train', 'val', 或 'test'
        df: 数据DataFrame
        attack_ratio: 攻击注入比例
    """
    print(f"\n{'=' * 80}")
    print(f" " * 20 + f"为{split_name.upper()}注入攻击")
    print(f"{'=' * 80}")
    
    # 创建攻击注入器
    injector = MultiAttackInjector(random_seed=config.RANDOM_SEED + hash(split_name) % 10000)
    
    flight_ids = df['flight'].unique()
    n_flights = len(flight_ids)
    n_attacks = int(n_flights * attack_ratio)
    
    print(f"\n总flights: {n_flights}")
    print(f"注入比例: {attack_ratio:.1%}")
    print(f"攻击flights: {n_attacks}")
    print(f"正常flights: {n_flights - n_attacks}")
    
    # 特殊处理测试集：确保每种攻击至少有目标数量
    if split_name == 'test':
        min_per_type = config.TEST_ATTACKS_PER_TYPE
        required_attacks = len(ATTACK_TYPES) * min_per_type
        
        if n_attacks < required_attacks:
            print(f"\n⚠️  警告: 测试集攻击数({n_attacks})少于要求({required_attacks})")
            print(f"   调整为每种攻击{min_per_type}个")
            n_attacks = required_attacks
        
        print(f"\n测试集策略:")
        print(f"  目标: 每种攻击{min_per_type}个")
        print(f"  总攻击: {n_attacks}")
    
    # 随机选择要注入攻击的flights
    np.random.seed(config.RANDOM_SEED + hash(split_name) % 10000)
    attack_flights = np.random.choice(flight_ids, size=n_attacks, replace=False)
    
    # 为每种攻击分配flights
    n_types = len(ATTACK_TYPES)
    flights_per_type = n_attacks // n_types
    remainder = n_attacks % n_types
    
    attack_assignment = {}
    idx = 0
    for i, attack_type in enumerate(ATTACK_TYPES):
        # 前remainder个类型多分配1个
        count = flights_per_type + (1 if i < remainder else 0)
        attack_assignment[attack_type] = attack_flights[idx:idx + count]
        idx += count
    
    print(f"\n攻击类型分布:")
    for attack_type, flights in attack_assignment.items():
        print(f"  {attack_type:20s}: {len(flights):3d} flights")
    
    # 注入攻击
    print(f"\n开始注入攻击...")
    attack_info = []
    attacked_dfs = []
    
    for attack_type, flights in attack_assignment.items():
        for flight_id in flights:
            flight_df = df[df['flight'] == flight_id].copy()
            
            # 获取攻击参数
            attack_params = ATTACK_TYPE_PARAMS.get(attack_type, {}).copy()
            
            # Delay attack doesn't need direction_mode or consistency_mode
            if attack_type != 'delay':
                attack_params['direction_mode'] = np.random.choice([
                    'random_xy', 'along_track_xy', 'cross_track_xy'
                ])
                attack_params['consistency_mode'] = 'pos_vel_acc'
            
            if attack_type == 'step':
                # Step: magnitude (meters)
                attack_params['magnitude'] = np.random.uniform(5.0, 30.0)
            
            elif attack_type == 'drift':
                # Drift: M (magnitude), T_drift (duration)
                attack_params['M'] = np.random.uniform(5.0, 30.0)
                attack_params['T_drift'] = np.random.uniform(5.0, 20.0)
            
            elif attack_type == 'takeover':
                # Takeover: M (magnitude), takeover_alpha, T_takeover
                attack_params['M'] = np.random.uniform(5.0, 30.0)
                attack_params['takeover_alpha'] = np.random.uniform(0.3, 0.7)
                if attack_params['offset_profile'] in ['ramp', 'sigmoid']:
                    attack_params['T_takeover'] = np.random.uniform(0.5, 2.0)
            
            elif attack_type == 'delay':
                # Delay: delay_seconds (no direction or consistency needed)
                attack_params['delay_seconds'] = np.random.uniform(1.0, 5.0)
            
            # 注入攻击
            try:
                attacked_flight, attack_info_dict = injector.inject_attack_to_flight(
                    flight_df,
                    attack_type=attack_type,
                    attack_prob=1.0,  # 确保注入
                    **attack_params
                )
                
                attacked_dfs.append(attacked_flight)
                
                # 记录攻击信息
                if attack_info_dict['attacked']:
                    attack_info.append({
                        'flight': int(flight_id),
                        'attack_type': attack_type,
                        'attack_start_time': float(attack_info_dict['attack_start_time']),
                        'attack_params': str(attack_params)
                    })
                else:
                    # 攻击失败（可能flight太短）
                    print(f"  警告: Flight {flight_id} 攻击失败: {attack_info_dict.get('reason', 'unknown')}")
                    attack_info.append({
                        'flight': int(flight_id),
                        'attack_type': 'none',
                        'attack_start_time': np.nan,
                        'attack_params': str(attack_info_dict.get('reason', 'failed'))
                    })
                
            except Exception as e:
                print(f"  错误: Flight {flight_id} 注入异常: {e}")
                # 保留原始flight
                attacked_dfs.append(flight_df)
                attack_info.append({
                    'flight': int(flight_id),
                    'attack_type': 'none',
                    'attack_start_time': np.nan,
                    'attack_params': str(e)
                })
    
    # 添加未被攻击的正常flights
    normal_flights = set(flight_ids) - set(attack_flights)
    for flight_id in normal_flights:
        flight_df = df[df['flight'] == flight_id].copy()
        attacked_dfs.append(flight_df)
        
        # 记录为正常
        attack_info.append({
            'flight': int(flight_id),
            'attack_type': 'none',
            'attack_start_time': np.nan,
            'magnitude': 0.0
        })
    
    # 合并所有flights
    final_df = pd.concat(attacked_dfs, ignore_index=True)
    
    # 排序
    final_df = final_df.sort_values(['flight', 'time']).reset_index(drop=True)
    
    print(f"\n✓ 注入完成:")
    print(f"  总flights: {final_df['flight'].nunique()}")
    print(f"  总记录: {len(final_df):,}")
    print(f"  攻击flights: {len([x for x in attack_info if x['attack_type'] != 'none'])}")
    
    return final_df, attack_info


def inject_all_splits():
    """为所有splits注入攻击"""
    print("=" * 80)
    print(" " * 20 + "步骤4：分层注入攻击")
    print("=" * 80)
    
    output_dir = Path(config.OUTPUT_DIR)
    aug_dir = output_dir / 'augmented'
    
    # 检查输入文件
    required_files = [
        aug_dir / 'train_augmented.csv',
        aug_dir / 'val_augmented.csv',
        aug_dir / 'test_augmented.csv'
    ]
    
    for f in required_files:
        if not f.exists():
            print(f"\n错误: 未找到 {f}")
            print("请先运行 generate_per_split.py")
            return None
    
    results = {}
    
    # Train注入
    print("\n" + "=" * 80)
    print("1/3: Train集攻击注入")
    print("=" * 80)
    train_df = pd.read_csv(aug_dir / 'train_augmented.csv', low_memory=False)
    train_final, train_attack_info = inject_attacks_for_split(
        'train',
        train_df,
        config.ATTACK_INJECTION_RATIOS['train']
    )
    results['train'] = (train_final, train_attack_info)
    
    # Val注入
    print("\n" + "=" * 80)
    print("2/3: Val集攻击注入")
    print("=" * 80)
    val_df = pd.read_csv(aug_dir / 'val_augmented.csv', low_memory=False)
    val_final, val_attack_info = inject_attacks_for_split(
        'val',
        val_df,
        config.ATTACK_INJECTION_RATIOS['val']
    )
    results['val'] = (val_final, val_attack_info)
    
    # Test注入
    print("\n" + "=" * 80)
    print("3/3: Test集攻击注入")
    print("=" * 80)
    test_df = pd.read_csv(aug_dir / 'test_augmented.csv', low_memory=False)
    test_final, test_attack_info = inject_attacks_for_split(
        'test',
        test_df,
        config.ATTACK_INJECTION_RATIOS['test']
    )
    results['test'] = (test_final, test_attack_info)
    
    # 保存结果
    print("\n" + "=" * 80)
    print("保存最终数据")
    print("=" * 80)
    
    for split_name, (final_df, attack_info) in results.items():
        # 保存数据
        data_path = output_dir / f'{split_name}_with_attacks.csv'
        final_df.to_csv(data_path, index=False)
        print(f"\n✓ {data_path}")
        print(f"  Flights: {final_df['flight'].nunique()}")
        print(f"  Records: {len(final_df):,}")
        
        # 保存攻击信息
        info_df = pd.DataFrame(attack_info)
        
        # 添加'attacked'列（兼容labeling模块）
        info_df['attacked'] = info_df['attack_type'] != 'none'
        
        info_path = output_dir / f'{split_name}_attack_info.csv'
        info_df.to_csv(info_path, index=False)
        print(f"✓ {info_path}")
        
        # 统计
        n_attacks = len([x for x in attack_info if x['attack_type'] != 'none'])
        print(f"  攻击flights: {n_attacks}")
        for atype in ATTACK_TYPES:
            count = len([x for x in attack_info if x['attack_type'] == atype])
            if count > 0:
                print(f"    {atype:20s}: {count}")
    
    # 保存总结
    summary = {
        'train': {
            'total_flights': int(results['train'][0]['flight'].nunique()),
            'attack_flights': len([x for x in results['train'][1] if x['attack_type'] != 'none']),
            'attack_ratio': float(config.ATTACK_INJECTION_RATIOS['train'])
        },
        'val': {
            'total_flights': int(results['val'][0]['flight'].nunique()),
            'attack_flights': len([x for x in results['val'][1] if x['attack_type'] != 'none']),
            'attack_ratio': float(config.ATTACK_INJECTION_RATIOS['val'])
        },
        'test': {
            'total_flights': int(results['test'][0]['flight'].nunique()),
            'attack_flights': len([x for x in results['test'][1] if x['attack_type'] != 'none']),
            'attack_ratio': float(config.ATTACK_INJECTION_RATIOS['test']),
            'attacks_per_type': {
                atype: len([x for x in results['test'][1] if x['attack_type'] == atype])
                for atype in ATTACK_TYPES
            }
        }
    }
    
    summary_path = output_dir / 'attack_injection_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ 总结已保存: {summary_path}")
    
    print("\n" + "=" * 80)
    print("✅ 步骤4完成：攻击注入完成")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = inject_all_splits()
    
    if results is not None:
        print("\n✅ 数据准备完成！")
        print("\n下一步: 运行 main.py 训练模型")
