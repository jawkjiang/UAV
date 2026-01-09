"""
为所有攻击类型训练TimeGAN模型
一键训练脚本
"""
import numpy as np
from pathlib import Path
import time
from timegan_simple import train_timegan_for_attack_type, generate_synthetic_attacks

def train_all_attack_types(expansion_factor=3):
    """
    为所有攻击类型训练TimeGAN并生成合成数据
    
    Args:
        expansion_factor: 扩充倍数（默认3倍，即为2000个原始样本生成6000个合成样本）
    
    Returns:
        synthetic_data_by_type: {attack_type: synthetic_windows}
    """
    print("=" * 80)
    print(" " * 25 + "TimeGAN 批量训练")
    print("=" * 80)
    print(f"扩充倍数: {expansion_factor}x")
    print(f"推荐配置: window级别, 3-5倍扩充")
    print("=" * 80)
    
    # 加载准备好的攻击窗口
    data_path = Path('timegan_data')
    if not data_path.exists():
        print("\n错误: 请先运行 prepare_timegan_data.py 提取攻击窗口")
        return None
    
    # 查找所有攻击类型数据
    attack_files = list(data_path.glob('attack_windows_*.npz'))
    print(f"\n发现 {len(attack_files)} 个攻击类型")
    
    synthetic_data_by_type = {}
    training_summary = []
    
    total_start = time.time()
    
    for i, attack_file in enumerate(attack_files, 1):
        # 提取攻击类型名称
        attack_type = attack_file.stem.replace('attack_windows_', '')
        
        print(f"\n[{i}/{len(attack_files)}] 处理攻击类型: {attack_type}")
        print("-" * 80)
        
        # 加载数据
        data = np.load(attack_file)
        attack_windows = data['windows']
        n_original = len(attack_windows)
        n_synthetic = int(n_original * expansion_factor)
        
        print(f"原始样本: {n_original}")
        print(f"生成目标: {n_synthetic} ({expansion_factor}x)")
        
        # 如果样本太少，跳过
        if n_original < 100:
            print(f"⚠ 样本数太少 ({n_original} < 100)，跳过此攻击类型")
            continue
        
        # 训练TimeGAN
        start_time = time.time()
        try:
            timegan, mean, std = train_timegan_for_attack_type(
                attack_windows, 
                attack_type,
                save_path='timegan_models'
            )
            
            # 生成合成数据
            print(f"\n生成 {n_synthetic} 个合成样本...")
            synthetic_windows = generate_synthetic_attacks(timegan, n_synthetic, mean, std)
            synthetic_data_by_type[attack_type] = synthetic_windows
            
            training_time = time.time() - start_time
            
            print(f"✓ 完成! 训练时间: {training_time:.1f}秒")
            
            training_summary.append({
                'attack_type': attack_type,
                'n_original': n_original,
                'n_synthetic': n_synthetic,
                'training_time': training_time
            })
            
        except Exception as e:
            print(f"✗ 训练失败: {e}")
            continue
    
    total_time = time.time() - total_start
    
    # 打印总结
    print("\n" + "=" * 80)
    print(" " * 30 + "训练总结")
    print("=" * 80)
    print(f"{'攻击类型':<20} {'原始':<10} {'合成':<10} {'训练时间':<15}")
    print("-" * 80)
    
    total_original = 0
    total_synthetic = 0
    
    for summary in training_summary:
        print(f"{summary['attack_type']:<20} {summary['n_original']:<10} "
              f"{summary['n_synthetic']:<10} {summary['training_time']:<15.1f}秒")
        total_original += summary['n_original']
        total_synthetic += summary['n_synthetic']
    
    print("-" * 80)
    print(f"{'总计':<20} {total_original:<10} {total_synthetic:<10} {total_time:<15.1f}秒")
    print("=" * 80)
    
    # 保存所有合成数据
    print("\n保存合成数据...")
    import os
    os.makedirs('timegan_synthetic', exist_ok=True)
    
    for attack_type, synthetic_windows in synthetic_data_by_type.items():
        filename = f'timegan_synthetic/synthetic_{attack_type}.npz'
        np.savez(filename, windows=synthetic_windows)
        print(f"✓ {filename} ({len(synthetic_windows)} 窗口)")
    
    print("\n✓ 全部完成!")
    return synthetic_data_by_type


def quick_quality_check(synthetic_data_by_type):
    """
    快速质量检查
    """
    print("\n" + "=" * 80)
    print(" " * 28 + "质量检查")
    print("=" * 80)
    
    # 加载原始数据进行对比
    data_path = Path('timegan_data')
    
    for attack_type, synthetic_windows in synthetic_data_by_type.items():
        print(f"\n{attack_type}:")
        
        # 加载原始数据
        original_file = data_path / f'attack_windows_{attack_type}.npz'
        if original_file.exists():
            original_data = np.load(original_file)
            original_windows = original_data['windows']
            
            print(f"  原始数据统计:")
            print(f"    均值: {original_windows.mean():.4f}, 标准差: {original_windows.std():.4f}")
            print(f"    范围: [{original_windows.min():.4f}, {original_windows.max():.4f}]")
            
            print(f"  合成数据统计:")
            print(f"    均值: {synthetic_windows.mean():.4f}, 标准差: {synthetic_windows.std():.4f}")
            print(f"    范围: [{synthetic_windows.min():.4f}, {synthetic_windows.max():.4f}]")
            
            # 计算分布差异
            mean_diff = abs(original_windows.mean() - synthetic_windows.mean())
            std_diff = abs(original_windows.std() - synthetic_windows.std())
            
            print(f"  分布差异:")
            print(f"    均值差: {mean_diff:.4f} ({mean_diff/abs(original_windows.mean())*100:.1f}%)")
            print(f"    标准差差: {std_diff:.4f} ({std_diff/abs(original_windows.std())*100:.1f}%)")
            
            if mean_diff / abs(original_windows.mean()) < 0.1 and std_diff / abs(original_windows.std()) < 0.2:
                print("  ✓ 质量良好!")
            else:
                print("  ⚠ 分布差异较大，建议检查")
        else:
            print("  ⚠ 未找到原始数据，跳过对比")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='训练TimeGAN for所有攻击类型')
    parser.add_argument('--expansion', type=int, default=3, 
                        help='扩充倍数 (默认3倍)')
    args = parser.parse_args()
    
    # 训练
    synthetic_data_by_type = train_all_attack_types(expansion_factor=args.expansion)
    
    if synthetic_data_by_type:
        # 质量检查
        quick_quality_check(synthetic_data_by_type)
        
        print("\n" + "=" * 80)
        print(" " * 25 + "下一步操作")
        print("=" * 80)
        print("1. 检查 timegan_models/ 中的模型文件")
        print("2. 检查 timegan_synthetic/ 中的合成数据")
        print("3. 运行 create_balanced_dataset.py 创建平衡数据集")
        print("4. 用新数据集重新训练模型")
        print("=" * 80)
