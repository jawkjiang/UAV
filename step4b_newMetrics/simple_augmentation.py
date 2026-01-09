"""
简单时序数据增强 - 立即可用
适用于GPS欺骗检测的攻击窗口增强
"""
import numpy as np

class TimeSeriesAugmenter:
    """时序数据增强器"""
    
    @staticmethod
    def jittering(data, sigma=0.03):
        """
        添加高斯噪声（模拟传感器噪声）
        
        Args:
            data: (n_samples, n_features) 时序数据
            sigma: 噪声标准差（相对于数据标准差）
        
        Returns:
            增强后的数据
        """
        noise = np.random.normal(0, sigma, data.shape)
        return data + noise * np.std(data, axis=0)
    
    @staticmethod
    def scaling(data, sigma=0.1):
        """
        随机缩放（模拟信号强度变化）
        
        Args:
            data: (n_samples, n_features)
            sigma: 缩放因子的标准差
        
        Returns:
            缩放后的数据
        """
        factor = np.random.normal(1.0, sigma, (1, data.shape[1]))
        return data * factor
    
    @staticmethod
    def time_warp(data, sigma=0.2, knot=4):
        """
        时间扭曲（模拟采样率变化）
        
        Args:
            data: (n_samples, n_features)
            sigma: 扭曲强度
            knot: 扭曲节点数
        
        Returns:
            时间扭曲后的数据
        """
        from scipy.interpolate import CubicSpline
        
        orig_steps = np.arange(data.shape[0])
        
        # 生成随机扭曲曲线
        random_warps = np.random.normal(0, sigma, size=(knot+2, data.shape[1]))
        warp_steps = np.linspace(0, data.shape[0]-1, num=knot+2)
        
        # 使用三次样条插值
        warper = CubicSpline(warp_steps, random_warps)
        warp = np.cumsum(warper(orig_steps), axis=0)
        
        # 应用扭曲
        warped_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            warped_data[:, i] = np.interp(orig_steps, 
                                          orig_steps + warp[:, i], 
                                          data[:, i])
        
        return warped_data
    
    @staticmethod
    def magnitude_warp(data, sigma=0.2, knot=4):
        """
        幅度扭曲（模拟信号强度随时间变化）
        
        Args:
            data: (n_samples, n_features)
            sigma: 扭曲强度
            knot: 扭曲节点数
        
        Returns:
            幅度扭曲后的数据
        """
        from scipy.interpolate import CubicSpline
        
        orig_steps = np.arange(data.shape[0])
        
        # 生成随机幅度曲线
        random_warps = np.random.normal(1.0, sigma, size=(knot+2, data.shape[1]))
        warp_steps = np.linspace(0, data.shape[0]-1, num=knot+2)
        
        # 使用三次样条插值
        warper = CubicSpline(warp_steps, random_warps)
        warp = warper(orig_steps)
        
        return data * warp
    
    @staticmethod
    def window_slice(data, reduce_ratio=0.9):
        """
        窗口切片（提取子序列）
        
        Args:
            data: (n_samples, n_features)
            reduce_ratio: 保留比例
        
        Returns:
            切片后的数据（保持原长度，用插值填充）
        """
        target_len = int(data.shape[0] * reduce_ratio)
        if target_len >= data.shape[0]:
            return data
        
        start = np.random.randint(0, data.shape[0] - target_len)
        sliced = data[start:start+target_len, :]
        
        # 插值回原长度
        from scipy.interpolate import interp1d
        orig_idx = np.arange(target_len)
        new_idx = np.linspace(0, target_len-1, data.shape[0])
        
        result = np.zeros_like(data)
        for i in range(data.shape[1]):
            f = interp1d(orig_idx, sliced[:, i], kind='cubic', fill_value='extrapolate')
            result[:, i] = f(new_idx)
        
        return result


def augment_attack_windows(windows, labels, augment_factor=3):
    """
    增强攻击窗口
    
    Args:
        windows: (n_windows, window_size, n_features)
        labels: (n_windows,) 0=normal, 1=attack
        augment_factor: 每个攻击窗口生成几个增强样本
    
    Returns:
        augmented_windows, augmented_labels
    """
    augmenter = TimeSeriesAugmenter()
    
    # 分离正常和攻击样本
    attack_mask = labels == 1
    attack_windows = windows[attack_mask]
    normal_windows = windows[~attack_mask]
    
    print(f"原始数据: {len(attack_windows)} 攻击窗口, {len(normal_windows)} 正常窗口")
    
    # 增强攻击样本
    augmented_attack = [attack_windows]
    
    for i in range(augment_factor - 1):
        aug_type = i % 4  # 轮流使用不同增强方法
        
        if aug_type == 0:
            # Jittering
            aug = np.array([augmenter.jittering(w) for w in attack_windows])
        elif aug_type == 1:
            # Scaling
            aug = np.array([augmenter.scaling(w) for w in attack_windows])
        elif aug_type == 2:
            # Magnitude Warp
            aug = np.array([augmenter.magnitude_warp(w) for w in attack_windows])
        else:
            # 组合: Jittering + Scaling
            aug = np.array([augmenter.scaling(augmenter.jittering(w)) 
                           for w in attack_windows])
        
        augmented_attack.append(aug)
    
    # 合并
    all_attack_windows = np.concatenate(augmented_attack, axis=0)
    all_attack_labels = np.ones(len(all_attack_windows), dtype=int)
    
    # 与正常样本合并
    final_windows = np.concatenate([normal_windows, all_attack_windows], axis=0)
    final_labels = np.concatenate([labels[~attack_mask], all_attack_labels], axis=0)
    
    print(f"增强后数据: {len(all_attack_windows)} 攻击窗口, {len(normal_windows)} 正常窗口")
    print(f"总计: {len(final_windows)} 窗口, 攻击比例: {100*final_labels.sum()/len(final_labels):.1f}%")
    
    # 打乱顺序
    shuffle_idx = np.random.permutation(len(final_windows))
    return final_windows[shuffle_idx], final_labels[shuffle_idx]


# 使用示例
if __name__ == "__main__":
    # 模拟数据
    np.random.seed(42)
    n_windows = 1000
    window_size = 50
    n_features = 6
    
    # 90% 正常, 10% 攻击
    windows = np.random.randn(n_windows, window_size, n_features)
    labels = np.random.choice([0, 1], size=n_windows, p=[0.9, 0.1])
    
    print("=" * 60)
    print("时序数据增强示例")
    print("=" * 60)
    
    # 增强
    aug_windows, aug_labels = augment_attack_windows(windows, labels, augment_factor=5)
    
    print("\n✓ 增强完成！")
    print(f"  可用于重新训练模型，降低类别不平衡问题")
