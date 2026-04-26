"""
步骤1：先划分原始数据
从flights.csv中划分train/val/test，确保无重叠
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
import config

def split_original_flights():
    """
    第一步：从原始数据划分train/val/test
    
    关键：这一步在TimeGAN之前，确保测试集完全独立
    """
    print("=" * 80)
    print(" " * 25 + "步骤1：划分原始数据")
    print("=" * 80)
    
    # 加载原始数据
    data_path = Path(config.DATA_PATH)
    print(f"\n加载原始数据: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    
    print(f"总记录数: {len(df):,}")
    print(f"总flights: {df['flight'].nunique()}")
    
    # 只保留正常flights（如果有攻击标签）
    if 'is_attack' in df.columns or 'label' in df.columns:
        label_col = 'is_attack' if 'is_attack' in df.columns else 'label'
        df = df[df[label_col] == 0].copy()
        print(f"正常记录数: {len(df):,}")
    
    # 添加delta_t
    if 'delta_t' not in df.columns:
        print("\n添加delta_t...")
        df['delta_t'] = df.groupby('flight')['time'].diff()
        df.loc[df.groupby('flight').head(1).index, 'delta_t'] = 0.0
    
    # 获取所有flight IDs
    all_flights = df['flight'].unique()
    n_total = len(all_flights)
    print(f"\n正常flights总数: {n_total}")
    
    # 划分比例
    train_ratio = config.SPLIT_RATIOS['train']
    val_ratio = config.SPLIT_RATIOS['val']
    test_ratio = config.SPLIT_RATIOS['test']
    
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val  # 确保总和正确
    
    print(f"\n划分比例: {train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%}")
    print(f"  Train: {n_train} flights")
    print(f"  Val:   {n_val} flights")
    print(f"  Test:  {n_test} flights")
    
    # 随机划分（固定种子确保可重复）
    np.random.seed(config.RANDOM_SEED)
    shuffled_flights = all_flights.copy()
    np.random.shuffle(shuffled_flights)
    
    train_flights = shuffled_flights[:n_train]
    val_flights = shuffled_flights[n_train:n_train + n_val]
    test_flights = shuffled_flights[n_train + n_val:]
    
    # 验证无重叠
    assert len(set(train_flights) & set(val_flights)) == 0, "Train-Val overlap!"
    assert len(set(train_flights) & set(test_flights)) == 0, "Train-Test overlap!"
    assert len(set(val_flights) & set(test_flights)) == 0, "Val-Test overlap!"
    print("\n✓ 验证通过：三个集合无重叠")
    
    # 创建输出目录
    output_dir = Path(config.OUTPUT_DIR) / 'split_flights'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存flight IDs
    with open(output_dir / 'train_flights.txt', 'w') as f:
        f.write('\n'.join(map(str, sorted(train_flights))))
    with open(output_dir / 'val_flights.txt', 'w') as f:
        f.write('\n'.join(map(str, sorted(val_flights))))
    with open(output_dir / 'test_flights.txt', 'w') as f:
        f.write('\n'.join(map(str, sorted(test_flights))))
    
    print(f"\n✓ Flight IDs已保存到: {output_dir}")
    
    # 提取各集合的完整数据
    train_df = df[df['flight'].isin(train_flights)].copy()
    val_df = df[df['flight'].isin(val_flights)].copy()
    test_df = df[df['flight'].isin(test_flights)].copy()
    
    # 保存CSV
    train_df.to_csv(output_dir / 'train_normal.csv', index=False)
    val_df.to_csv(output_dir / 'val_normal.csv', index=False)
    test_df.to_csv(output_dir / 'test_normal.csv', index=False)
    
    print(f"\n✓ 数据已保存:")
    print(f"  train_normal.csv: {len(train_df):,} 记录")
    print(f"  val_normal.csv:   {len(val_df):,} 记录")
    print(f"  test_normal.csv:  {len(test_df):,} 记录")
    
    # 保存划分信息
    split_info = {
        'total_flights': int(n_total),
        'train': {
            'n_flights': int(n_train),
            'n_records': int(len(train_df)),
            'flight_ids': sorted([int(x) for x in train_flights])
        },
        'val': {
            'n_flights': int(n_val),
            'n_records': int(len(val_df)),
            'flight_ids': sorted([int(x) for x in val_flights])
        },
        'test': {
            'n_flights': int(n_test),
            'n_records': int(len(test_df)),
            'flight_ids': sorted([int(x) for x in test_flights])
        },
        'random_seed': config.RANDOM_SEED,
        'split_ratios': config.SPLIT_RATIOS
    }
    
    split_info_path = Path(config.OUTPUT_DIR) / 'split_info.json'
    with open(split_info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n✓ 划分信息已保存: {split_info_path}")
    
    # 统计信息
    print("\n" + "=" * 80)
    print("划分统计")
    print("=" * 80)
    
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        flight_lengths = split_df.groupby('flight').size()
        print(f"\n{split_name}:")
        print(f"  Flights: {split_df['flight'].nunique()}")
        print(f"  Records: {len(split_df):,}")
        print(f"  Avg length: {flight_lengths.mean():.1f}")
        print(f"  Length range: [{flight_lengths.min()}, {flight_lengths.max()}]")
    
    print("\n" + "=" * 80)
    print("✅ 步骤1完成：原始数据已划分")
    print("=" * 80)
    
    return split_info


if __name__ == "__main__":
    split_info = split_original_flights()
    print("\n下一步: 运行 train_independent_timegans.py")
