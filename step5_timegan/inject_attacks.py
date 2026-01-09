"""
阶段5: 在扩充后的正常数据上注入攻击（低比例）
"""
import sys
sys.path.append('../step3b_multiModelGeneral')

import pandas as pd
import numpy as np
from pathlib import Path
import json
import config
from mixed_attack_injector import MixedAttackInjector

def inject_attacks():
    """
    在划分好的数据集上注入攻击
    使用低攻击比例，避免过拟合
    """
    print("=" * 80)
    print(" " * 25 + "注入攻击")
    print("=" * 80)
    
    output_path = Path(config.OUTPUT_DIR)
    
    # 加载划分
    splits_path = output_path / 'flight_splits.json'
    if not splits_path.exists():
        print(f"\n错误: 请先运行 merge_and_split.py")
        return None
    
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    
    # 加载合并的正常数据
    merged_path = output_path / 'merged_normal_flights.csv'
    if not merged_path.exists():
        print(f"\n错误: 未找到 {merged_path}")
        return None
    
    print(f"\n加载数据: {merged_path}")
    df = pd.read_csv(merged_path, low_memory=False)
    
    # 初始化injector
    injector = MixedAttackInjector(random_seed=config.RANDOM_SEED)
    
    # 辅助函数：获取子集
    def get_subset(flight_ids):
        return df[df['flight'].isin(flight_ids)].copy()
    
    # ========================================================================
    # 训练集
    # ========================================================================
    print("\n" + "=" * 80)
    print(f"训练集 - 注入攻击 (比例: {config.TRAIN_ATTACK_RATIO*100:.0f}%)")
    print("=" * 80)
    
    train_df = get_subset(splits['train'])
    print(f"正常飞行数: {train_df['flight'].nunique()}")
    
    train_df, train_attack_info = injector.inject_mixed_attacks_to_dataset(
        train_df,
        attack_ratio=config.TRAIN_ATTACK_RATIO,
        attack_types=config.ATTACK_TYPES
    )
    
    train_attack_info.to_csv(output_path / 'train_attack_info.csv', index=False)
    
    # 统计 - 从attack_info获取攻击数，从原始splits获取总数
    n_attacked = len(train_attack_info)
    n_total = len(splits['train'])
    print(f"\n✓ 训练集攻击注入完成:")
    print(f"  总飞行数: {n_total}")
    print(f"  攻击飞行数: {n_attacked}")
    print(f"  实际攻击比例: {n_attacked/n_total*100:.1f}%")
    
    # ========================================================================
    # 验证集
    # ========================================================================
    print("\n" + "=" * 80)
    print(f"验证集 - 注入攻击 (比例: {config.VAL_ATTACK_RATIO*100:.0f}%)")
    print("=" * 80)
    
    val_df = get_subset(splits['val'])
    print(f"正常飞行数: {val_df['flight'].nunique()}")
    
    val_df, val_attack_info = injector.inject_mixed_attacks_to_dataset(
        val_df,
        attack_ratio=config.VAL_ATTACK_RATIO,
        attack_types=config.ATTACK_TYPES
    )
    
    val_attack_info.to_csv(output_path / 'val_attack_info.csv', index=False)
    
    n_attacked = len(val_attack_info)
    n_total = len(splits['val'])
    print(f"\n✓ 验证集攻击注入完成:")
    print(f"  总飞行数: {n_total}")
    print(f"  攻击飞行数: {n_attacked}")
    print(f"  实际攻击比例: {n_attacked/n_total*100:.1f}%")
    
    # ========================================================================
    # 测试集（保证覆盖所有攻击类型）
    # ========================================================================
    print("\n" + "=" * 80)
    print(f"测试集 - 注入攻击 (比例: {config.TEST_ATTACK_RATIO*100:.0f}%)")
    print("=" * 80)
    
    test_df = get_subset(splits['test'])
    print(f"正常飞行数: {test_df['flight'].nunique()}")
    
    test_df, test_attack_info = injector.inject_mixed_attacks_with_guarantee(
        test_df,
        attack_ratio=config.TEST_ATTACK_RATIO,
        attack_types=config.ATTACK_TYPES,
        min_per_attack=config.MIN_TEST_FLIGHTS_PER_ATTACK
    )
    
    test_attack_info.to_csv(output_path / 'test_attack_info.csv', index=False)
    
    n_attacked = len(test_attack_info)
    n_total = len(splits['test'])
    print(f"\n✓ 测试集攻击注入完成:")
    print(f"  总飞行数: {n_total}")
    print(f"  攻击飞行数: {n_attacked}")
    print(f"  实际攻击比例: {n_attacked/n_total*100:.1f}%")
    
    # 保存带攻击的数据 
    print("\n保存数据集...")
    print("  保存训练集...")
    train_df.to_csv(output_path / 'train_with_attacks.csv', index=False)
    print("  保存验证集...")
    val_df.to_csv(output_path / 'val_with_attacks.csv', index=False)
    print("  保存测试集...")
    test_df.to_csv(output_path / 'test_with_attacks.csv', index=False)
    print("✓ 数据集已保存")
    
    print("\n" + "=" * 80)
    print("所有数据集攻击注入完成!")
    print("=" * 80)
    
    # 总结
    print("\n最终数据分布:")
    print("-" * 80)
    
    for name, info_df in [('训练集', train_attack_info), 
                          ('验证集', val_attack_info), 
                          ('测试集', test_attack_info)]:
        if len(info_df) > 0:
            print(f"\n{name}:")
            print(info_df['attack_type'].value_counts())
    
    return train_df, val_df, test_df, train_attack_info, val_attack_info, test_attack_info


if __name__ == "__main__":
    results = inject_attacks()
    
    if results is not None:
        print("\n" + "=" * 80)
        print("下一步: 运行 main.py 完成特征工程、窗口创建和模型训练")
        print("=" * 80)
