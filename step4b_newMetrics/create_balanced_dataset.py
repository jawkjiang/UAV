"""
创建平衡的数据集（训练、验证、测试）
使用TimeGAN合成数据 + 原始数据，对齐正样本率
"""
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

def create_balanced_dataset(
    target_train_ratio=0.25,  # 训练集目标攻击比例
    target_val_ratio=0.15,    # 验证集目标攻击比例
    target_test_ratio=0.03,   # 测试集保持3%（真实场景）
    use_timegan=True
):
    """
    创建类别平衡的数据集
    
    策略B（推荐）: 渐进对齐
    - 训练集: 25% 攻击（使用TimeGAN扩充）
    - 验证集: 15% 攻击
    - 测试集: 3% 攻击（保持不变）
    
    Args:
        target_train_ratio: 训练集目标攻击比例
        target_val_ratio: 验证集目标攻击比例
        target_test_ratio: 测试集攻击比例（通常保持不变）
        use_timegan: 是否使用TimeGAN合成数据
    """
    print("=" * 80)
    print(" " * 25 + "创建平衡数据集")
    print("=" * 80)
    print(f"目标比例:")
    print(f"  训练集: {target_train_ratio:.1%} 攻击")
    print(f"  验证集: {target_val_ratio:.1%} 攻击")
    print(f"  测试集: {target_test_ratio:.1%} 攻击 (保持不变)")
    print(f"使用TimeGAN: {use_timegan}")
    print("=" * 80)
    
    # 1. 加载原始数据
    step3b_path = Path('../step3b_multiModelGeneral/output')
    
    print("\n[1] 加载原始数据...")
    train_data = np.load(step3b_path / 'train_data.npz')
    val_data = np.load(step3b_path / 'val_data.npz')
    test_data = np.load(step3b_path / 'test_data.npz')
    
    X_train_orig = train_data['X']
    y_train_orig = train_data['y']
    flight_ids_train = train_data['flight_ids']
    
    X_val_orig = val_data['X']
    y_val_orig = val_data['y']
    flight_ids_val = val_data['flight_ids']
    
    X_test = test_data['X']  # 测试集保持不变
    y_test = test_data['y']
    flight_ids_test = test_data['flight_ids']
    
    print(f"  训练集: {len(X_train_orig)} 窗口, {y_train_orig.mean():.1%} 攻击")
    print(f"  验证集: {len(X_val_orig)} 窗口, {y_val_orig.mean():.1%} 攻击")
    print(f"  测试集: {len(X_test)} 窗口, {y_test.mean():.1%} 攻击 (保持不变)")
    
    # 2. 加载元数据（用于追踪攻击类型）
    metadata = pd.read_csv(step3b_path / 'window_metadata.csv')
    
    # 3. 如果使用TimeGAN，加载合成数据
    synthetic_attacks = {}
    if use_timegan:
        print("\n[2] 加载TimeGAN合成数据...")
        synthetic_path = Path('timegan_synthetic')
        
        if not synthetic_path.exists():
            print("  ⚠ 未找到TimeGAN合成数据，将不使用TimeGAN")
            use_timegan = False
        else:
            synthetic_files = list(synthetic_path.glob('synthetic_*.npz'))
            for syn_file in synthetic_files:
                attack_type = syn_file.stem.replace('synthetic_', '')
                data = np.load(syn_file)
                synthetic_attacks[attack_type] = data['windows']
                print(f"  ✓ {attack_type}: {len(synthetic_attacks[attack_type])} 窗口")
    
    # 4. 重新平衡训练集
    print("\n[3] 重新平衡训练集...")
    
    # 分离原始训练集中的正常和攻击窗口
    X_train_normal = X_train_orig[y_train_orig == 0]
    X_train_attack = X_train_orig[y_train_orig == 1]
    
    print(f"  原始攻击窗口: {len(X_train_attack)}")
    
    # 如果使用TimeGAN，合并原始+合成攻击窗口
    if use_timegan and synthetic_attacks:
        # 合并所有合成攻击窗口
        all_synthetic = []
        for attack_type, syn_windows in synthetic_attacks.items():
            all_synthetic.append(syn_windows)
        
        if all_synthetic:
            all_synthetic = np.concatenate(all_synthetic, axis=0)
            print(f"  TimeGAN合成窗口: {len(all_synthetic)}")
            
            # 合并原始+合成
            X_train_attack_combined = np.concatenate([X_train_attack, all_synthetic], axis=0)
            print(f"  合并后攻击窗口: {len(X_train_attack_combined)}")
        else:
            X_train_attack_combined = X_train_attack
    else:
        X_train_attack_combined = X_train_attack
    
    # 计算需要的正常窗口数量
    n_attack = len(X_train_attack_combined)
    n_normal_needed = int(n_attack * (1 - target_train_ratio) / target_train_ratio)
    
    print(f"\n  目标比例: {target_train_ratio:.1%}")
    print(f"  攻击窗口: {n_attack}")
    print(f"  需要正常窗口: {n_normal_needed}")
    print(f"  可用正常窗口: {len(X_train_normal)}")
    
    # 如果正常窗口不够，需要扩充
    if n_normal_needed > len(X_train_normal):
        print(f"  ⚠ 正常窗口不足，需要扩充 {n_normal_needed - len(X_train_normal)} 个")
        
        # 简单扩充：添加随机噪声
        n_to_augment = n_normal_needed - len(X_train_normal)
        augmented_normal = []
        
        for i in range(n_to_augment):
            # 随机选择一个正常窗口
            idx = np.random.randint(0, len(X_train_normal))
            sample = X_train_normal[idx].copy()
            
            # 添加小噪声
            noise = np.random.randn(*sample.shape) * 0.01 * sample.std()
            sample_aug = sample + noise
            
            augmented_normal.append(sample_aug)
        
        augmented_normal = np.array(augmented_normal)
        X_train_normal_final = np.concatenate([X_train_normal, augmented_normal], axis=0)
        print(f"  ✓ 正常窗口扩充至: {len(X_train_normal_final)}")
    else:
        # 随机采样
        indices = np.random.choice(len(X_train_normal), n_normal_needed, replace=False)
        X_train_normal_final = X_train_normal[indices]
    
    # 合并训练集
    X_train_new = np.concatenate([X_train_normal_final, X_train_attack_combined], axis=0)
    y_train_new = np.concatenate([
        np.zeros(len(X_train_normal_final)),
        np.ones(len(X_train_attack_combined))
    ])
    
    # 打乱
    shuffle_idx = np.random.permutation(len(X_train_new))
    X_train_new = X_train_new[shuffle_idx]
    y_train_new = y_train_new[shuffle_idx]
    
    print(f"\n  新训练集: {len(X_train_new)} 窗口, {y_train_new.mean():.1%} 攻击")
    
    # 5. 重新平衡验证集
    print("\n[4] 重新平衡验证集...")
    
    X_val_normal = X_val_orig[y_val_orig == 0]
    X_val_attack = X_val_orig[y_val_orig == 1]
    
    n_attack_val = len(X_val_attack)
    n_normal_val_needed = int(n_attack_val * (1 - target_val_ratio) / target_val_ratio)
    
    print(f"  目标比例: {target_val_ratio:.1%}")
    print(f"  攻击窗口: {n_attack_val}")
    print(f"  需要正常窗口: {n_normal_val_needed}")
    print(f"  可用正常窗口: {len(X_val_normal)}")
    
    if n_normal_val_needed <= len(X_val_normal):
        indices = np.random.choice(len(X_val_normal), n_normal_val_needed, replace=False)
        X_val_normal_final = X_val_normal[indices]
    else:
        # 扩充
        n_to_augment = n_normal_val_needed - len(X_val_normal)
        augmented = []
        for i in range(n_to_augment):
            idx = np.random.randint(0, len(X_val_normal))
            sample = X_val_normal[idx].copy()
            noise = np.random.randn(*sample.shape) * 0.01 * sample.std()
            augmented.append(sample + noise)
        
        X_val_normal_final = np.concatenate([X_val_normal, np.array(augmented)], axis=0)
    
    X_val_new = np.concatenate([X_val_normal_final, X_val_attack], axis=0)
    y_val_new = np.concatenate([
        np.zeros(len(X_val_normal_final)),
        np.ones(len(X_val_attack))
    ])
    
    # 打乱
    shuffle_idx = np.random.permutation(len(X_val_new))
    X_val_new = X_val_new[shuffle_idx]
    y_val_new = y_val_new[shuffle_idx]
    
    print(f"  新验证集: {len(X_val_new)} 窗口, {y_val_new.mean():.1%} 攻击")
    
    # 6. 保存新数据集
    print("\n[5] 保存平衡数据集...")
    output_path = Path('balanced_dataset')
    output_path.mkdir(exist_ok=True)
    
    np.savez(
        output_path / 'train_data_balanced.npz',
        X=X_train_new,
        y=y_train_new
    )
    
    np.savez(
        output_path / 'val_data_balanced.npz',
        X=X_val_new,
        y=y_val_new
    )
    
    # 测试集保持不变
    np.savez(
        output_path / 'test_data_balanced.npz',
        X=X_test,
        y=y_test,
        flight_ids=flight_ids_test
    )
    
    print(f"  ✓ 训练集: balanced_dataset/train_data_balanced.npz")
    print(f"  ✓ 验证集: balanced_dataset/val_data_balanced.npz")
    print(f"  ✓ 测试集: balanced_dataset/test_data_balanced.npz")
    
    # 7. 打印总结
    print("\n" + "=" * 80)
    print(" " * 30 + "总结")
    print("=" * 80)
    print(f"{'数据集':<15} {'窗口数':<12} {'攻击比例':<12} {'正常:攻击':<15}")
    print("-" * 80)
    print(f"{'训练集(原)':<15} {len(X_train_orig):<12} {y_train_orig.mean():<12.1%} "
          f"{(1-y_train_orig.mean())/y_train_orig.mean():.1f}:1")
    print(f"{'训练集(新)':<15} {len(X_train_new):<12} {y_train_new.mean():<12.1%} "
          f"{(1-y_train_new.mean())/y_train_new.mean():.1f}:1")
    print()
    print(f"{'验证集(原)':<15} {len(X_val_orig):<12} {y_val_orig.mean():<12.1%} "
          f"{(1-y_val_orig.mean())/y_val_orig.mean():.1f}:1")
    print(f"{'验证集(新)':<15} {len(X_val_new):<12} {y_val_new.mean():<12.1%} "
          f"{(1-y_val_new.mean())/y_val_new.mean():.1f}:1")
    print()
    print(f"{'测试集':<15} {len(X_test):<12} {y_test.mean():<12.1%} "
          f"{(1-y_test.mean())/y_test.mean():.1f}:1")
    print("=" * 80)
    
    print("\n✓ 完成! 现在可以用新数据集重新训练模型了")
    print("\n建议:")
    print("  1. 将 balanced_dataset/ 中的文件复制到 step3b_multiModelGeneral/output/")
    print("  2. 备份原始数据")
    print("  3. 重新运行训练: cd step3b_multiModelGeneral && python main.py")
    print("  4. 重新评估: cd step4b_newMetrics && python main_v2.py")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='创建平衡数据集')
    parser.add_argument('--train-ratio', type=float, default=0.25,
                        help='训练集目标攻击比例 (默认0.25 = 25%%)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='验证集目标攻击比例 (默认0.15 = 15%%)')
    parser.add_argument('--no-timegan', action='store_true',
                        help='不使用TimeGAN合成数据')
    
    args = parser.parse_args()
    
    create_balanced_dataset(
        target_train_ratio=args.train_ratio,
        target_val_ratio=args.val_ratio,
        use_timegan=not args.no_timegan
    )
