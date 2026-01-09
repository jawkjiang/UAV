"""
SMOTE用于时序数据的攻击样本增强
"""
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN

def smote_for_timeseries(windows, labels, target_ratio=0.3):
    """
    使用SMOTE增强时序数据
    
    注意：SMOTE原本用于表格数据，需要将时序展平
    
    Args:
        windows: (n_windows, window_size, n_features)
        labels: (n_windows,)
        target_ratio: 目标攻击比例
    
    Returns:
        augmented_windows, augmented_labels
    """
    # 展平时序维度
    n_windows, window_size, n_features = windows.shape
    windows_flat = windows.reshape(n_windows, -1)
    
    print(f"原始数据: {windows.shape}")
    print(f"  正常: {(labels==0).sum()}, 攻击: {(labels==1).sum()}")
    print(f"  攻击比例: {100*labels.sum()/len(labels):.1f}%")
    
    # 计算需要的样本数
    n_majority = (labels == 0).sum()
    n_minority = (labels == 1).sum()
    
    # 计算采样策略
    # target_ratio = n_minority_new / (n_majority + n_minority_new)
    # n_minority_new = target_ratio * n_majority / (1 - target_ratio)
    n_minority_target = int(target_ratio * n_majority / (1 - target_ratio))
    
    sampling_strategy = {
        0: n_majority,  # 保持多数类不变
        1: max(n_minority_target, n_minority)  # 增强少数类
    }
    
    print(f"\nSMOTE目标: 将攻击样本从 {n_minority} 增加到 {sampling_strategy[1]}")
    
    # 应用SMOTE
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    windows_resampled, labels_resampled = smote.fit_resample(windows_flat, labels)
    
    # 恢复时序形状
    windows_resampled = windows_resampled.reshape(-1, window_size, n_features)
    
    print(f"\n增强后数据: {windows_resampled.shape}")
    print(f"  正常: {(labels_resampled==0).sum()}, 攻击: {(labels_resampled==1).sum()}")
    print(f"  攻击比例: {100*labels_resampled.sum()/len(labels_resampled):.1f}%")
    
    return windows_resampled, labels_resampled


# 使用示例
if __name__ == "__main__":
    np.random.seed(42)
    
    # 模拟高度不平衡数据 (97% 正常, 3% 攻击)
    n_normal = 6800
    n_attack = 200
    window_size = 50
    n_features = 6
    
    normal_windows = np.random.randn(n_normal, window_size, n_features) * 0.5
    attack_windows = np.random.randn(n_attack, window_size, n_features) * 1.5 + 2.0
    
    windows = np.concatenate([normal_windows, attack_windows], axis=0)
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_attack)], axis=0)
    
    # 打乱
    shuffle_idx = np.random.permutation(len(windows))
    windows = windows[shuffle_idx]
    labels = labels[shuffle_idx]
    
    print("=" * 60)
    print("SMOTE时序数据增强")
    print("=" * 60)
    
    # 增强到30%攻击比例
    aug_windows, aug_labels = smote_for_timeseries(windows, labels, target_ratio=0.3)
    
    print("\n✓ SMOTE增强完成！")
    print("  建议用于重新训练，显著改善类别不平衡问题")
