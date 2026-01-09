"""
阶段4: 合并原始和合成正常飞行，并划分数据集
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import config

def merge_and_split():
    """
    合并原始和合成正常飞行，划分为Train/Val/Test
    """
    print("=" * 80)
    print(" " * 20 + "合并数据并划分数据集")
    print("=" * 80)
    
    output_path = Path(config.OUTPUT_DIR)
    
    # 加载数据
    normal_path = output_path / 'normal_flights.csv'
    synthetic_path = output_path / 'synthetic_normal_flights.csv'
    
    if not normal_path.exists():
        print(f"\n错误: 未找到 {normal_path}")
        return None
    
    print(f"\n加载原始正常飞行: {normal_path}")
    normal_df = pd.read_csv(normal_path, low_memory=False)
    n_normal = normal_df['flight'].nunique()
    print(f"  飞行数: {n_normal}")
    print(f"  记录数: {len(normal_df):,}")
    
    if synthetic_path.exists():
        print(f"\n加载合成正常飞行: {synthetic_path}")
        synthetic_df = pd.read_csv(synthetic_path)
        n_synthetic = synthetic_df['flight'].nunique()
        print(f"  飞行数: {n_synthetic}")
        print(f"  记录数: {len(synthetic_df):,}")
        
        # 合并
        print("\n合并数据...")
        merged_df = pd.concat([normal_df, synthetic_df], ignore_index=True)
    else:
        print("\n警告: 未找到合成数据，仅使用原始数据")
        merged_df = normal_df.copy()
        n_synthetic = 0
    
    # 保存合并数据
    merged_path = output_path / 'merged_normal_flights.csv'
    merged_df.to_csv(merged_path, index=False)
    print(f"\n✓ 已保存合并数据: {merged_path}")
    
    # 统计
    total_flights = merged_df['flight'].nunique()
    print(f"\n总飞行数: {total_flights}")
    print(f"  原始: {n_normal} ({n_normal/total_flights*100:.1f}%)")
    print(f"  合成: {n_synthetic} ({n_synthetic/total_flights*100:.1f}%)")
    
    # 划分数据集
    print("\n" + "=" * 80)
    print("划分数据集")
    print("=" * 80)
    
    flight_ids = merged_df['flight'].unique()
    np.random.seed(config.RANDOM_SEED)
    np.random.shuffle(flight_ids)
    
    n_total = len(flight_ids)
    n_train = int(n_total * config.TRAIN_RATIO)
    n_val = int(n_total * config.VAL_RATIO)
    
    train_flights = flight_ids[:n_train]
    val_flights = flight_ids[n_train:n_train + n_val]
    test_flights = flight_ids[n_train + n_val:]
    
    print(f"\n数据集划分 ({config.TRAIN_RATIO:.0%}/{config.VAL_RATIO:.0%}/{config.TEST_RATIO:.0%}):")
    print(f"  训练集: {len(train_flights)} flights")
    print(f"  验证集: {len(val_flights)} flights")
    print(f"  测试集: {len(test_flights)} flights")
    
    # 保存划分
    splits = {
        'train': train_flights.tolist(),
        'val': val_flights.tolist(),
        'test': test_flights.tolist()
    }
    
    splits_path = output_path / 'flight_splits.json'
    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"\n✓ 已保存划分: {splits_path}")
    
    # 保存各集合的CSV
    train_df = merged_df[merged_df['flight'].isin(train_flights)]
    val_df = merged_df[merged_df['flight'].isin(val_flights)]
    test_df = merged_df[merged_df['flight'].isin(test_flights)]
    
    train_df.to_csv(output_path / 'train_normal.csv', index=False)
    val_df.to_csv(output_path / 'val_normal.csv', index=False)
    test_df.to_csv(output_path / 'test_normal.csv', index=False)
    
    print("\n✓ 已保存各数据集:")
    print(f"  train_normal.csv: {len(train_df):,} 记录")
    print(f"  val_normal.csv: {len(val_df):,} 记录")
    print(f"  test_normal.csv: {len(test_df):,} 记录")
    
    return splits, merged_df


if __name__ == "__main__":
    splits, merged_df = merge_and_split()
    
    if splits is not None:
        print("\n" + "=" * 80)
        print("下一步: 运行 inject_attacks.py 注入攻击")
        print("=" * 80)
