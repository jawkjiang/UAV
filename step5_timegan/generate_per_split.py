"""
步骤3：为每个split生成合成数据
使用各自的TimeGAN生成合成flights
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import config
from timegan_flight import TimeGAN

def generate_for_split(split_name, n_synthetic, timegan_model_path, norm_params_path, original_data_path):
    """
    为单个split生成合成数据
    
    Args:
        split_name: 'train', 'val', 或 'test'
        n_synthetic: 生成多少个合成flights
        timegan_model_path: TimeGAN模型路径
        norm_params_path: 归一化参数路径
        original_data_path: 原始数据路径（用作模板）
    """
    print(f"\n{'=' * 80}")
    print(f" " * 20 + f"为{split_name.upper()}生成合成数据")
    print(f"{'=' * 80}")
    
    # 加载TimeGAN模型
    print(f"\n加载TimeGAN模型: {timegan_model_path}")
    device = config.TIMEGAN_DEVICE if torch.cuda.is_available() else 'cpu'
    
    # 创建TimeGAN实例
    timegan = TimeGAN(
        seq_len=config.TIMEGAN_SEQUENCE_LENGTH,
        feature_dim=len(config.TIMEGAN_FEATURES),
        hidden_dim=config.TIMEGAN_HIDDEN_DIM,
        num_layers=config.TIMEGAN_NUM_LAYERS,
        device=device
    )
    
    # 加载模型权重
    timegan.load(timegan_model_path)
    
    # 加载归一化参数
    print(f"加载归一化参数: {norm_params_path}")
    norm_data = np.load(norm_params_path, allow_pickle=True)
    mean = norm_data['mean']
    std = norm_data['std']
    feature_cols = norm_data['feature_cols']
    
    # 加载原始数据作为模板
    print(f"加载原始数据: {original_data_path}")
    original_df = pd.read_csv(original_data_path, low_memory=False)
    
    # 获取合成flight ID范围
    id_start, id_end = config.SYNTHETIC_FLIGHT_ID_RANGES[split_name]
    print(f"\n生成{n_synthetic}个合成flights (ID: {id_start}-{id_end-1})")
    
    # 生成合成序列
    print("\n生成合成序列...")
    synthetic_sequences = timegan.generate(n_synthetic)
    
    # 反归一化
    synthetic_sequences = synthetic_sequences * std + mean
    
    print(f"  生成序列形状: {synthetic_sequences.shape}")
    print(f"  数据范围: [{synthetic_sequences.min():.4f}, {synthetic_sequences.max():.4f}]")
    
    # 转换为flights
    print("\n转换为flight格式...")
    synthetic_flights = []
    
    # 从原始数据中随机选择模板flights
    template_flights = original_df['flight'].unique()
    np.random.seed(config.RANDOM_SEED + hash(split_name) % 10000)
    
    for i, seq in enumerate(synthetic_sequences):
        flight_id = id_start + i
        
        # 随机选择一个模板flight的结构
        template_flight_id = np.random.choice(template_flights)
        template_df = original_df[original_df['flight'] == template_flight_id].copy()
        
        # 使用序列长度
        seq_len = len(seq)
        
        # 如果模板比序列长，截断；如果短，重复最后几行
        if len(template_df) > seq_len:
            template_df = template_df.iloc[:seq_len].copy()
        elif len(template_df) < seq_len:
            # 重复最后几行直到达到seq_len
            repeat_count = seq_len - len(template_df)
            last_rows = template_df.iloc[-repeat_count:].copy()
            template_df = pd.concat([template_df, last_rows], ignore_index=True)
        
        # 替换特征值
        template_df['flight'] = flight_id
        for j, col in enumerate(feature_cols):
            if col in template_df.columns:
                template_df[col] = seq[:, j]
        
        synthetic_flights.append(template_df)
    
    # 合并所有合成flights
    synthetic_df = pd.concat(synthetic_flights, ignore_index=True)
    
    print(f"✓ 生成完成:")
    print(f"  合成flights: {synthetic_df['flight'].nunique()}")
    print(f"  合成记录: {len(synthetic_df):,}")
    
    # 合并原始和合成数据
    print("\n合并原始和合成数据...")
    augmented_df = pd.concat([original_df, synthetic_df], ignore_index=True)
    
    print(f"  原始flights: {original_df['flight'].nunique()}")
    print(f"  合成flights: {synthetic_df['flight'].nunique()}")
    print(f"  总flights: {augmented_df['flight'].nunique()}")
    print(f"  总记录: {len(augmented_df):,}")
    
    # 保存
    output_dir = Path(config.OUTPUT_DIR) / 'augmented'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f'{split_name}_augmented.csv'
    augmented_df.to_csv(output_path, index=False)
    
    print(f"\n✓ 已保存: {output_path}")
    
    return augmented_df


def generate_all_splits():
    """为所有splits生成合成数据"""
    print("=" * 80)
    print(" " * 20 + "步骤3：生成合成数据")
    print("=" * 80)
    
    output_dir = Path(config.OUTPUT_DIR)
    split_dir = output_dir / 'split_flights'
    model_dir = Path('timegan_models')
    
    # 检查必要文件
    required_files = [
        model_dir / 'timegan_train.pt',
        model_dir / 'timegan_val.pt',
        model_dir / 'timegan_test.pt',
        model_dir / 'norm_params_train.npz',
        model_dir / 'norm_params_val.npz',
        model_dir / 'norm_params_test.npz',
        split_dir / 'train_normal.csv',
        split_dir / 'val_normal.csv',
        split_dir / 'test_normal.csv'
    ]
    
    for f in required_files:
        if not f.exists():
            print(f"\n错误: 未找到 {f}")
            print("请先运行前面的步骤")
            return None
    
    results = {}
    
    # 生成Train
    print("\n" + "=" * 80)
    print("1/3: 生成Train合成数据")
    print("=" * 80)
    train_mult = config.AUGMENTATION_MULTIPLIERS['train']
    
    # 计算需要生成的数量
    train_df = pd.read_csv(split_dir / 'train_normal.csv', low_memory=False)
    n_train_original = train_df['flight'].nunique()
    n_train_synthetic = n_train_original * train_mult
    
    train_augmented = generate_for_split(
        'train',
        n_train_synthetic,
        model_dir / 'timegan_train.pt',
        model_dir / 'norm_params_train.npz',
        split_dir / 'train_normal.csv'
    )
    results['train'] = train_augmented
    
    # 生成Val
    print("\n" + "=" * 80)
    print("2/3: 生成Val合成数据")
    print("=" * 80)
    val_mult = config.AUGMENTATION_MULTIPLIERS['val']
    
    val_df = pd.read_csv(split_dir / 'val_normal.csv', low_memory=False)
    n_val_original = val_df['flight'].nunique()
    n_val_synthetic = n_val_original * val_mult
    
    val_augmented = generate_for_split(
        'val',
        n_val_synthetic,
        model_dir / 'timegan_val.pt',
        model_dir / 'norm_params_val.npz',
        split_dir / 'val_normal.csv'
    )
    results['val'] = val_augmented
    
    # 生成Test
    print("\n" + "=" * 80)
    print("3/3: 生成Test合成数据")
    print("=" * 80)
    test_mult = config.AUGMENTATION_MULTIPLIERS['test']
    
    test_df = pd.read_csv(split_dir / 'test_normal.csv', low_memory=False)
    n_test_original = test_df['flight'].nunique()
    n_test_synthetic = n_test_original * test_mult
    
    test_augmented = generate_for_split(
        'test',
        n_test_synthetic,
        model_dir / 'timegan_test.pt',
        model_dir / 'norm_params_test.npz',
        split_dir / 'test_normal.csv'
    )
    results['test'] = test_augmented
    
    print("\n" + "=" * 80)
    print("✅ 步骤3完成：合成数据已生成")
    print("=" * 80)
    
    print("\n数据扩充总结:")
    print(f"  Train: {n_train_original} → {results['train']['flight'].nunique()} ({train_mult}x)")
    print(f"  Val:   {n_val_original} → {results['val']['flight'].nunique()} ({val_mult}x)")
    print(f"  Test:  {n_test_original} → {results['test']['flight'].nunique()} ({test_mult}x)")
    
    return results


if __name__ == "__main__":
    results = generate_all_splits()
    
    if results is not None:
        print("\n下一步: 运行 inject_attacks_stratified.py")
