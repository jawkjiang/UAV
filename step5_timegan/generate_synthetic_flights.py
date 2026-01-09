"""
阶段3: 使用训练好的TimeGAN生成合成正常飞行
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import config
from timegan_flight import TimeGAN

def generate_synthetic_flights(expansion_factor=None):
    """
    生成合成正常飞行轨迹
    
    Args:
        expansion_factor: 扩充倍数（默认使用config中的值）
    """
    if expansion_factor is None:
        expansion_factor = config.TIMEGAN_EXPANSION_FACTOR
    
    print("=" * 80)
    print(" " * 20 + "生成合成正常飞行")
    print("=" * 80)
    print(f"扩充倍数: {expansion_factor}x")
    
    # 加载原始正常飞行数据（用于参考）
    normal_flights_path = Path(config.OUTPUT_DIR) / 'normal_flights.csv'
    if not normal_flights_path.exists():
        print(f"\n错误: 未找到 {normal_flights_path}")
        return None
    
    original_df = pd.read_csv(normal_flights_path, low_memory=False)
    n_original_flights = original_df['flight'].nunique()
    n_synthetic_flights = int(n_original_flights * expansion_factor)
    
    print(f"\n原始正常飞行数: {n_original_flights}")
    print(f"目标合成飞行数: {n_synthetic_flights}")
    
    # 加载TimeGAN模型
    model_dir = Path('timegan_models')
    model_path = model_dir / 'timegan_normal.pt'
    norm_path = model_dir / 'norm_params.npz'
    
    if not model_path.exists() or not norm_path.exists():
        print(f"\n错误: 请先运行 train_timegan.py 训练模型")
        return None
    
    print(f"\n加载TimeGAN模型: {model_path}")
    norm_data = np.load(norm_path, allow_pickle=True)
    mean = norm_data['mean']
    std = norm_data['std']
    feature_cols = list(norm_data['feature_cols'])
    
    # 优先使用CPU以避免内存问题
    device = 'cpu'  # 强制使用CPU，避免CUDA OOM
    print(f"使用设备: {device} (避免GPU内存问题)")
    
    timegan = TimeGAN(
        seq_len=config.TIMEGAN_SEQUENCE_LENGTH,
        feature_dim=len(feature_cols),
        hidden_dim=config.TIMEGAN_HIDDEN_DIM,
        num_layers=config.TIMEGAN_NUM_LAYERS,
        device=device
    )
    timegan.load(model_path)
    
    # 计算需要生成的序列数（每个飞行大约3-5个序列）
    sequences_per_flight = 4  # 每个合成飞行使用4个序列
    n_sequences_to_generate = n_synthetic_flights * sequences_per_flight
    
    print(f"\n生成 {n_sequences_to_generate} 个序列（每飞行{sequences_per_flight}个）...")
    
    # 分批生成以避免内存问题
    batch_size = 1000  # 每批生成1000个序列
    all_synthetic_sequences = []
    
    n_batches = (n_sequences_to_generate + batch_size - 1) // batch_size
    print(f"分{n_batches}批生成，每批{batch_size}个序列")
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_sequences_to_generate)
        batch_n = end_idx - start_idx
        
        print(f"  批次 {i+1}/{n_batches}: 生成 {batch_n} 个序列...", end='')
        
        # 生成序列
        synthetic_sequences_norm_batch = timegan.generate(batch_n)
        
        # 反归一化
        synthetic_sequences_batch = synthetic_sequences_norm_batch * std + mean
        
        all_synthetic_sequences.append(synthetic_sequences_batch)
        print(f" ✓")
    
    # 合并所有批次
    synthetic_sequences = np.concatenate(all_synthetic_sequences, axis=0)
    
    print(f"\n✓ 生成完成")
    print(f"  形状: {synthetic_sequences.shape}")
    print(f"  范围: [{synthetic_sequences.min():.4f}, {synthetic_sequences.max():.4f}]")
    
    # 重构为飞行格式
    print("\n重构为飞行格式...")
    synthetic_flights = []
    
    flight_id_start = original_df['flight'].max() + 1
    
    for i in range(n_synthetic_flights):
        # 每个飞行使用固定数量的序列
        start_idx = i * sequences_per_flight
        end_idx = start_idx + sequences_per_flight
        
        if start_idx >= len(synthetic_sequences):
            break
        
        # 拼接序列
        flight_sequences = synthetic_sequences[start_idx:end_idx]
        flight_data = flight_sequences.reshape(-1, len(feature_cols))
        
        # 创建DataFrame
        flight_df = pd.DataFrame(flight_data, columns=feature_cols)
        flight_df['flight'] = flight_id_start + i
        # 使用与原始数据相同的采样间隔（约0.156s，而不是config中的100Hz）
        flight_df['time'] = np.arange(len(flight_df)) * 0.156  # 匹配原始数据
        
        # 添加delta_t（时间差，feature_engineering需要）
        flight_df['delta_t'] = 0.156  # 固定采样间隔
        flight_df.loc[flight_df.index[0], 'delta_t'] = 0.0  # 第一个点delta_t为0
        
        # 后处理：计算衍生特征（使feature_engineering生成的16个特征完整）
        # 这些特征可以从TimeGAN生成的10个核心特征推导
        
        # 注意：不需要生成所有原始数据的列（如wind_speed等）
        # feature_engineering.py会使用这些核心特征计算一致性残差
        # 只需要提供足够的列即可
        
        synthetic_flights.append(flight_df)
    
    # 合并所有合成飞行
    synthetic_df = pd.concat(synthetic_flights, ignore_index=True)
    
    print(f"\n✓ 重构完成:")
    print(f"  合成飞行数: {synthetic_df['flight'].nunique()}")
    print(f"  总记录数: {len(synthetic_df):,}")
    
    # 保存
    output_path = Path(config.OUTPUT_DIR)
    output_file = output_path / 'synthetic_normal_flights.csv'
    synthetic_df.to_csv(output_file, index=False)
    print(f"\n✓ 已保存: {output_file}")
    
    # 统计
    print("\n" + "=" * 80)
    print("数据统计:")
    print("=" * 80)
    print(f"原始正常飞行: {n_original_flights}")
    print(f"合成正常飞行: {synthetic_df['flight'].nunique()}")
    print(f"总正常飞行: {n_original_flights + synthetic_df['flight'].nunique()}")
    print(f"扩充倍数: {(n_original_flights + synthetic_df['flight'].nunique()) / n_original_flights:.2f}x")
    
    return synthetic_df


if __name__ == "__main__":
    synthetic_df = generate_synthetic_flights()
    
    if synthetic_df is not None:
        print("\n下一步: 运行 merge_and_split.py 合并数据并划分训练/验证/测试集")
