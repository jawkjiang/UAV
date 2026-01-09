"""
重新注入攻击并正确保存完整数据集
修复：确保保存包含正常+攻击的完整数据集
"""
import sys
sys.path.append('../step3b_multiModelGeneral')

import pandas as pd
import numpy as np
from pathlib import Path
import json
import config
from mixed_attack_injector import MixedAttackInjector


def reinject_attacks():
    """
    重新注入攻击，正确保存正常+攻击的完整数据集
    """
    print("=" * 80)
    print(" " * 20 + "重新注入攻击")
    print("=" * 80)
    
    output_path = Path(config.OUTPUT_DIR)
    
    # 加载划分
    with open(output_path / 'flight_splits.json', 'r') as f:
        splits = json.load(f)
    
    # 加载所有正常数据
    print("\n加载合并的正常数据...")
    merged_df = pd.read_csv(output_path / 'merged_normal_flights.csv', low_memory=False)
    print(f"  Total flights: {merged_df['flight'].nunique()}")
    print(f"  Total points: {len(merged_df):,}")
    
    # 初始化injector
    injector = MixedAttackInjector(random_seed=config.RANDOM_SEED)
    
    # 处理每个split
    for split_name, flight_ids, attack_ratio in [
        ('train', splits['train'], config.TRAIN_ATTACK_RATIO),
        ('val', splits['val'], config.VAL_ATTACK_RATIO),
        ('test', splits['test'], config.TEST_ATTACK_RATIO)
    ]:
        print("\n" + "=" * 80)
        print(f"{split_name.upper()} - Attack Ratio: {attack_ratio*100:.0f}%")
        print("=" * 80)
        
        # 获取该split的所有flights数据
        split_df = merged_df[merged_df['flight'].isin(flight_ids)].copy()
        n_total = split_df['flight'].nunique()
        print(f"  Total flights: {n_total}")
        print(f"  Total points: {len(split_df):,}")
        
        # 注入攻击
        if split_name == 'test':
            # 测试集使用保证覆盖的方法
            split_df_attacked, attack_info = injector.inject_mixed_attacks_with_guarantee(
                split_df,
                attack_ratio=attack_ratio,
                attack_types=config.ATTACK_TYPES,
                min_per_attack=config.MIN_TEST_FLIGHTS_PER_ATTACK
            )
        else:
            # 训练/验证集使用普通方法
            split_df_attacked, attack_info = injector.inject_mixed_attacks_to_dataset(
                split_df,
                attack_ratio=attack_ratio,
                attack_types=config.ATTACK_TYPES
            )
        
        # 统计
        n_attacked = attack_info['flight'].nunique()
        print(f"\n✓ {split_name.capitalize()} completed:")
        print(f"  Total flights: {n_total}")
        print(f"  Attacked flights: {n_attacked}")
        print(f"  Normal flights: {n_total - n_attacked}")
        print(f"  Actual attack ratio: {n_attacked/n_total*100:.1f}%")
        print(f"  Total points: {len(split_df_attacked):,}")
        
        # 验证数据
        assert len(split_df_attacked) == len(split_df), f"Data points mismatch for {split_name}!"
        assert split_df_attacked['flight'].nunique() == n_total, f"Flight count mismatch for {split_name}!"
        
        # 保存数据
        complete_path = output_path / f'{split_name}_complete.csv'
        split_df_attacked.to_csv(complete_path, index=False)
        print(f"  ✓ Saved: {complete_path}")
        
        # 保存攻击信息
        attack_info_path = output_path / f'{split_name}_attack_info.csv'
        attack_info.to_csv(attack_info_path, index=False)
        print(f"  ✓ Saved: {attack_info_path}")
        
        # 打印攻击类型分布
        if len(attack_info) > 0:
            print(f"\n  Attack type distribution:")
            type_counts = attack_info['attack_type'].value_counts()
            for attack_type, count in type_counts.items():
                print(f"    {attack_type:20s}: {count:3d} flights")
    
    print("\n" + "=" * 80)
    print("✓ 攻击注入完成!")
    print("=" * 80)
    
    print("\n下一步: 使用 *_complete.csv 文件进行训练")
    print("  训练集: train_complete.csv")
    print("  验证集: val_complete.csv")
    print("  测试集: test_complete.csv")


if __name__ == "__main__":
    reinject_attacks()
