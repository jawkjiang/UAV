"""
步骤2：为每个split独立训练TimeGAN
关键：3个完全独立的TimeGAN，避免数据泄漏
"""
import numpy as np
import torch
from pathlib import Path
import config
from timegan_flight import TimeGAN

def prepare_sequences(df, seq_length=150):
    """将数据切分为固定长度序列"""
    feature_cols = config.TIMEGAN_FEATURES
    
    sequences = []
    flight_ids = df['flight'].unique()
    
    for flight_id in flight_ids:
        flight_df = df[df['flight'] == flight_id].copy()
        flight_data = flight_df[feature_cols].values
        
        # 滑动窗口切分
        for i in range(0, len(flight_data) - seq_length + 1, seq_length // 2):
            seq = flight_data[i:i + seq_length]
            if len(seq) == seq_length:
                sequences.append(seq)
    
    return np.array(sequences)


def train_single_timegan(split_name, data_path, save_prefix):
    """
    训练单个TimeGAN
    
    Args:
        split_name: 'train', 'val', 或 'test'
        data_path: 数据CSV路径
        save_prefix: 模型保存前缀
    """
    print(f"\n{'=' * 80}")
    print(f" " * 25 + f"训练{split_name.upper()} TimeGAN")
    print(f"{'=' * 80}")
    
    import pandas as pd
    
    # 加载数据
    print(f"\n加载数据: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    
    n_flights = df['flight'].nunique()
    print(f"  Flights: {n_flights}")
    print(f"  Records: {len(df):,}")
    
    # 准备序列
    print("\n准备训练序列...")
    sequences = prepare_sequences(df, seq_length=config.TIMEGAN_SEQUENCE_LENGTH)
    
    print(f"  序列数: {len(sequences)}")
    print(f"  序列形状: {sequences.shape}")
    
    # 归一化
    print("\n归一化...")
    mean = sequences.mean(axis=(0, 1))
    std = sequences.std(axis=(0, 1)) + 1e-6
    sequences_norm = (sequences - mean) / std
    
    print(f"  归一化范围: [{sequences_norm.min():.4f}, {sequences_norm.max():.4f}]")
    
    # 初始化TimeGAN
    print("\n初始化TimeGAN...")
    device = config.TIMEGAN_DEVICE if torch.cuda.is_available() else 'cpu'
    print(f"  设备: {device}")
    
    timegan = TimeGAN(
        seq_len=sequences.shape[1],
        feature_dim=sequences.shape[2],
        hidden_dim=config.TIMEGAN_HIDDEN_DIM,
        num_layers=config.TIMEGAN_NUM_LAYERS,
        device=device
    )
    
    # 三阶段训练
    print("\n开始训练...")
    print(f"  批大小: {config.TIMEGAN_BATCH_SIZE}")
    print(f"  学习率: {config.TIMEGAN_LR}")
    
    print("\n[阶段1/3] 训练自编码器...")
    timegan.train_autoencoder(
        sequences_norm,
        epochs=config.TIMEGAN_EPOCHS_AE,
        batch_size=config.TIMEGAN_BATCH_SIZE,
        lr=config.TIMEGAN_LR
    )
    
    print("\n[阶段2/3] 训练监督器...")
    timegan.train_supervisor(
        sequences_norm,
        epochs=config.TIMEGAN_EPOCHS_SUP,
        batch_size=config.TIMEGAN_BATCH_SIZE,
        lr=config.TIMEGAN_LR
    )
    
    print("\n[阶段3/3] 训练GAN...")
    timegan.train_gan(
        sequences_norm,
        epochs=config.TIMEGAN_EPOCHS_GAN,
        batch_size=config.TIMEGAN_BATCH_SIZE,
        lr=config.TIMEGAN_LR
    )
    
    # 保存模型
    model_dir = Path('timegan_models')
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / f'{save_prefix}.pt'
    timegan.save(model_path)
    print(f"\n✓ 模型已保存: {model_path}")
    
    # 保存归一化参数
    norm_path = model_dir / f'norm_params_{split_name}.npz'
    np.savez(norm_path, 
             mean=mean, 
             std=std, 
             feature_cols=config.TIMEGAN_FEATURES)
    print(f"✓ 归一化参数已保存: {norm_path}")
    
    return timegan, mean, std


def train_all_independent_timegans():
    """训练3个独立的TimeGAN"""
    print("=" * 80)
    print(" " * 20 + "步骤2：训练独立TimeGAN模型")
    print("=" * 80)
    print("\n关键：每个split使用独立的TimeGAN，避免数据泄漏")
    
    output_dir = Path(config.OUTPUT_DIR)
    split_dir = output_dir / 'split_flights'
    
    # 检查输入文件
    required_files = [
        split_dir / 'train_normal.csv',
        split_dir / 'val_normal.csv',
        split_dir / 'test_normal.csv'
    ]
    
    for f in required_files:
        if not f.exists():
            print(f"\n错误: 未找到 {f}")
            print("请先运行 split_original_first.py")
            return None
    
    results = {}
    
    # 训练Train TimeGAN
    print("\n" + "=" * 80)
    print("1/3: Train TimeGAN")
    print("=" * 80)
    train_timegan, train_mean, train_std = train_single_timegan(
        'train',
        split_dir / 'train_normal.csv',
        'timegan_train'
    )
    results['train'] = {
        'model': train_timegan,
        'mean': train_mean,
        'std': train_std
    }
    
    # 训练Val TimeGAN
    print("\n" + "=" * 80)
    print("2/3: Val TimeGAN")
    print("=" * 80)
    val_timegan, val_mean, val_std = train_single_timegan(
        'val',
        split_dir / 'val_normal.csv',
        'timegan_val'
    )
    results['val'] = {
        'model': val_timegan,
        'mean': val_mean,
        'std': val_std
    }
    
    # 训练Test TimeGAN
    print("\n" + "=" * 80)
    print("3/3: Test TimeGAN")
    print("=" * 80)
    test_timegan, test_mean, test_std = train_single_timegan(
        'test',
        split_dir / 'test_normal.csv',
        'timegan_test'
    )
    results['test'] = {
        'model': test_timegan,
        'mean': test_mean,
        'std': test_std
    }
    
    print("\n" + "=" * 80)
    print("✅ 步骤2完成：3个独立TimeGAN已训练")
    print("=" * 80)
    print("\n关键验证：")
    print("  ✓ Train-TimeGAN: 只在train flights上训练")
    print("  ✓ Val-TimeGAN:   只在val flights上训练")
    print("  ✓ Test-TimeGAN:  只在test flights上训练")
    print("  ✓ 无数据泄漏！")
    
    return results


if __name__ == "__main__":
    results = train_all_independent_timegans()
    
    if results is not None:
        print("\n下一步: 运行 generate_per_split.py")
