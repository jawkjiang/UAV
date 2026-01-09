"""
阶段1: 从原始数据中提取正常飞行轨迹
用于TimeGAN训练
"""
import pandas as pd
import numpy as np
from pathlib import Path
import config

def extract_normal_flights():
    """
    从原始flights.csv中提取所有正常飞行
    
    Returns:
        normal_flights_df: 完整的正常飞行数据
    """
    print("=" * 80)
    print(" " * 25 + "提取正常飞行数据")
    print("=" * 80)
    
    # 加载原始数据
    print(f"\n加载数据: {config.DATA_PATH}")
    df = pd.read_csv(config.DATA_PATH, low_memory=False)
    print(f"总记录数: {len(df):,}")
    print(f"总飞行数: {df['flight'].nunique()}")
    
    # 检查是否有攻击标签
    if 'is_attack' in df.columns or 'label' in df.columns:
        print("\n警告: 数据中包含攻击标签，将只保留正常数据")
        label_col = 'is_attack' if 'is_attack' in df.columns else 'label'
        df = df[df[label_col] == 0].copy()
        print(f"正常记录数: {len(df):,}")
    
    # 统计信息
    flight_ids = df['flight'].unique()
    print(f"\n提取的正常飞行数: {len(flight_ids)}")
    
    # 添加delta_t（如果不存在）
    if 'delta_t' not in df.columns:
        print("\n添加delta_t列...")
        df['delta_t'] = df.groupby('flight')['time'].diff()
        df.loc[df.groupby('flight').head(1).index, 'delta_t'] = 0.0
        print(f"  ✓ delta_t已添加，均值: {df['delta_t'].mean():.4f}秒")
    
    # 按flight分析长度
    flight_lengths = df.groupby('flight').size()
    print(f"\n飞行长度统计:")
    print(f"  最短: {flight_lengths.min()}")
    print(f"  最长: {flight_lengths.max()}")
    print(f"  平均: {flight_lengths.mean():.1f}")
    print(f"  中位数: {flight_lengths.median():.1f}")
    
    # 保存
    output_path = Path(config.OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / 'normal_flights.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ 已保存: {output_file}")
    print(f"  记录数: {len(df):,}")
    print(f"  飞行数: {len(flight_ids)}")
    
    return df


def prepare_sequences_for_timegan(df, seq_length=150):
    """
    将飞行数据切分为固定长度的序列用于TimeGAN训练
    
    Args:
        df: 正常飞行数据
        seq_length: 序列长度
    
    Returns:
        sequences: (n_sequences, seq_length, n_features)
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "准备TimeGAN训练序列")
    print("=" * 80)
    
    # 选择特征 - 使用TimeGAN专用特征（10个核心物理特征）
    feature_cols = config.TIMEGAN_FEATURES
    
    print(f"\n特征列: {feature_cols}")
    print(f"特征数: {len(feature_cols)}")
    print(f"序列长度: {seq_length}")
    
    # 验证数据包含所有需要的特征
    missing_features = [f for f in feature_cols if f not in df.columns]
    if missing_features:
        raise ValueError(f"数据缺少以下特征: {missing_features}")
    
    sequences = []
    flight_ids = df['flight'].unique()
    
    print(f"\n切分飞行序列...")
    for flight_id in flight_ids:
        flight_df = df[df['flight'] == flight_id].copy()
        flight_data = flight_df[feature_cols].values
        
        # 滑动窗口切分
        for i in range(0, len(flight_data) - seq_length + 1, seq_length // 2):
            seq = flight_data[i:i + seq_length]
            if len(seq) == seq_length:
                sequences.append(seq)
    
    sequences = np.array(sequences)
    print(f"\n切分完成:")
    print(f"  序列数: {len(sequences)}")
    print(f"  形状: {sequences.shape}")
    print(f"  数据范围: [{sequences.min():.4f}, {sequences.max():.4f}]")
    
    # 保存
    output_path = Path(config.OUTPUT_DIR)
    output_file = output_path / 'timegan_training_sequences.npz'
    np.savez(output_file, 
             sequences=sequences,
             feature_cols=feature_cols,
             seq_length=seq_length)
    print(f"\n✓ 已保存: {output_file}")
    
    return sequences


if __name__ == "__main__":
    # 提取正常飞行
    normal_df = extract_normal_flights()
    
    # 准备TimeGAN训练序列
    sequences = prepare_sequences_for_timegan(
        normal_df, 
        seq_length=config.TIMEGAN_SEQUENCE_LENGTH
    )
    
    print("\n" + "=" * 80)
    print("下一步: 运行 train_timegan.py 训练TimeGAN模型")
    print("=" * 80)
