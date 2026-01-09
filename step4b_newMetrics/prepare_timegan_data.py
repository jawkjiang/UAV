"""
从step3b训练数据中提取攻击窗口，准备TimeGAN训练
"""
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

def extract_attack_windows():
    """
    从step3b的训练数据中提取攻击窗口
    
    Returns:
        attack_windows_by_type: {attack_type: array of shape (n_samples, 50, 6)}
    """
    print("=" * 70)
    print("提取攻击窗口用于TimeGAN训练")
    print("=" * 70)
    
    # 加载训练数据
    data_path = Path('../step3b_multiModelGeneral/output')
    
    print("\n加载数据...")
    npz_file = data_path / 'train_data.npz'
    data = np.load(npz_file)
    
    X_train = data['X']  # (n_windows, 50, 6)
    y_train = data['y']  # (n_windows,)
    flight_ids = data['flight_ids']  # (n_windows,)
    
    print(f"训练窗口总数: {len(X_train)}")
    print(f"攻击窗口数: {y_train.sum()}")
    print(f"正常窗口数: {(y_train == 0).sum()}")
    print(f"攻击比例: {y_train.mean():.1%}")
    
    # 加载元数据以获取攻击类型
    metadata_file = data_path / 'window_metadata.csv'
    if not metadata_file.exists():
        print(f"\n警告: 未找到 {metadata_file}")
        print("将所有攻击窗口视为同一类型")
        attack_windows_by_type = {'all_attacks': X_train[y_train == 1]}
        return attack_windows_by_type
    
    metadata = pd.read_csv(metadata_file)
    print(f"\n元数据记录数: {len(metadata)}")
    
    # 按攻击类型分组
    attack_windows_by_type = defaultdict(list)
    
    for i, (x, y, flight_id) in enumerate(zip(X_train, y_train, flight_ids)):
        if y == 1:  # 攻击窗口
            # 从元数据中查找攻击类型
            flight_meta = metadata[metadata['flight_id'] == flight_id]
            
            if len(flight_meta) > 0:
                attack_type = flight_meta.iloc[0]['attack_type']
                attack_windows_by_type[attack_type].append(x)
            else:
                attack_windows_by_type['unknown'].append(x)
    
    # 转换为numpy数组
    for attack_type in attack_windows_by_type:
        attack_windows_by_type[attack_type] = np.array(attack_windows_by_type[attack_type])
    
    # 显示统计
    print("\n攻击窗口按类型分布:")
    print("-" * 70)
    total = 0
    for attack_type, windows in sorted(attack_windows_by_type.items()):
        print(f"  {attack_type:20s}: {len(windows):5d} 窗口")
        total += len(windows)
    print("-" * 70)
    print(f"  {'总计':20s}: {total:5d} 窗口")
    
    return dict(attack_windows_by_type)


def validate_extraction(attack_windows_by_type):
    """
    验证提取的数据质量
    """
    print("\n" + "=" * 70)
    print("数据质量验证")
    print("=" * 70)
    
    for attack_type, windows in attack_windows_by_type.items():
        print(f"\n{attack_type}:")
        print(f"  形状: {windows.shape}")
        print(f"  数据范围: [{windows.min():.4f}, {windows.max():.4f}]")
        print(f"  均值: {windows.mean():.4f}")
        print(f"  标准差: {windows.std():.4f}")
        
        # 检查NaN和Inf
        has_nan = np.isnan(windows).any()
        has_inf = np.isinf(windows).any()
        print(f"  包含NaN: {has_nan}")
        print(f"  包含Inf: {has_inf}")
        
        if has_nan or has_inf:
            print("  ⚠ 警告: 数据包含无效值！")


def save_attack_windows(attack_windows_by_type, save_path='timegan_data'):
    """
    保存提取的攻击窗口
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("保存攻击窗口数据")
    print("=" * 70)
    
    for attack_type, windows in attack_windows_by_type.items():
        filename = f'{save_path}/attack_windows_{attack_type}.npz'
        np.savez(filename, windows=windows)
        print(f"✓ 已保存: {filename} ({len(windows)} 窗口)")
    
    print(f"\n✓ 所有数据已保存到 {save_path}/")


if __name__ == "__main__":
    # 提取
    attack_windows_by_type = extract_attack_windows()
    
    # 验证
    validate_extraction(attack_windows_by_type)
    
    # 保存
    save_attack_windows(attack_windows_by_type)
    
    print("\n" + "=" * 70)
    print("下一步: 运行 train_timegan_all_attacks.py 训练TimeGAN模型")
    print("=" * 70)
