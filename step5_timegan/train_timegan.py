"""
阶段2: 训练TimeGAN学习正常飞行模式
"""
import numpy as np
import torch
from pathlib import Path
import config
from timegan_flight import TimeGAN

def train_timegan():
    """训练TimeGAN模型"""
    print("=" * 80)
    print(" " * 25 + "训练TimeGAN模型")
    print("=" * 80)
    
    # 加载训练序列
    data_path = Path(config.OUTPUT_DIR) / 'timegan_training_sequences.npz'
    if not data_path.exists():
        print(f"\n错误: 请先运行 prepare_normal_data.py 提取数据")
        print(f"缺少文件: {data_path}")
        return None
    
    print(f"\n加载训练数据: {data_path}")
    data = np.load(data_path, allow_pickle=True)
    sequences = data['sequences']
    feature_cols = data['feature_cols']
    
    print(f"序列数: {len(sequences)}")
    print(f"序列形状: {sequences.shape}")
    print(f"特征: {list(feature_cols)}")
    
    # 归一化
    print("\n归一化数据...")
    mean = sequences.mean(axis=(0, 1))
    std = sequences.std(axis=(0, 1)) + 1e-6
    sequences_norm = (sequences - mean) / std
    
    print(f"归一化后范围: [{sequences_norm.min():.4f}, {sequences_norm.max():.4f}]")
    
    # 初始化TimeGAN
    print("\n初始化TimeGAN...")
    device = config.TIMEGAN_DEVICE if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    timegan = TimeGAN(
        seq_len=sequences.shape[1],
        feature_dim=sequences.shape[2],
        hidden_dim=config.TIMEGAN_HIDDEN_DIM,
        num_layers=config.TIMEGAN_NUM_LAYERS,
        device=device
    )
    
    # 三阶段训练
    print("\n开始训练...")
    print(f"批大小: {config.TIMEGAN_BATCH_SIZE}")
    print(f"学习率: {config.TIMEGAN_LR}")
    
    timegan.train_autoencoder(
        sequences_norm,
        epochs=config.TIMEGAN_EPOCHS_AE,
        batch_size=config.TIMEGAN_BATCH_SIZE,
        lr=config.TIMEGAN_LR
    )
    
    timegan.train_supervisor(
        sequences_norm,
        epochs=config.TIMEGAN_EPOCHS_SUP,
        batch_size=config.TIMEGAN_BATCH_SIZE,
        lr=config.TIMEGAN_LR
    )
    
    timegan.train_gan(
        sequences_norm,
        epochs=config.TIMEGAN_EPOCHS_GAN,
        batch_size=config.TIMEGAN_BATCH_SIZE,
        lr=config.TIMEGAN_LR
    )
    
    # 保存模型
    print("\n保存模型...")
    model_dir = Path('timegan_models')
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / 'timegan_normal.pt'
    timegan.save(model_path)
    
    # 保存归一化参数
    norm_path = model_dir / 'norm_params.npz'
    np.savez(norm_path, mean=mean, std=std, feature_cols=feature_cols)
    print(f"✓ 归一化参数已保存: {norm_path}")
    
    print("\n" + "=" * 80)
    print("训练完成!")
    print("=" * 80)
    
    return timegan, mean, std


if __name__ == "__main__":
    timegan, mean, std = train_timegan()
    
    if timegan is not None:
        print("\n下一步: 运行 generate_synthetic_flights.py 生成合成飞行")
