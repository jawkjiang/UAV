"""
修复训练数据：将正常飞行和攻击飞行正确合并
问题：当前train_with_attacks.csv只包含攻击飞行，缺少正常飞行
解决：从merged_normal_flights.csv中提取未攻击的飞行，合并到训练集
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
sys.path.append('../step3b_multiModelGeneral')

def fix_training_data():
    """修复训练、验证、测试数据"""
    
    output_dir = Path('./output')
    
    print("="*80)
    print(" "*20 + "修复训练数据")
    print("="*80)
    
    # 1. 加载flight splits
    print("\n[1/4] 加载flight splits...")
    with open(output_dir / 'flight_splits.json', 'r') as f:
        splits = json.load(f)
    
    print(f"  Train flights: {len(splits['train'])}")
    print(f"  Val flights: {len(splits['val'])}")
    print(f"  Test flights: {len(splits['test'])}")
    
    # 2. 加载所有正常飞行数据
    print("\n[2/4] 加载正常飞行数据...")
    merged_normal = pd.read_csv(output_dir / 'merged_normal_flights.csv', low_memory=False)
    print(f"  Total normal flights: {merged_normal['flight'].nunique()}")
    print(f"  Total normal points: {len(merged_normal):,}")
    
    # 3. 加载攻击信息
    print("\n[3/4] 加载攻击信息...")
    train_attack_info = pd.read_csv(output_dir / 'train_attack_info.csv')
    val_attack_info = pd.read_csv(output_dir / 'val_attack_info.csv')
    test_attack_info = pd.read_csv(output_dir / 'test_attack_info.csv')
    
    train_attacked_flights = set(train_attack_info['flight'].unique())
    val_attacked_flights = set(val_attack_info['flight'].unique())
    test_attacked_flights = set(test_attack_info['flight'].unique())
    
    print(f"  Train attacked flights: {len(train_attacked_flights)}")
    print(f"  Val attacked flights: {len(val_attacked_flights)}")
    print(f"  Test attacked flights: {len(test_attacked_flights)}")
    
    # 4. 为每个split构建完整数据集
    print("\n[4/4] 构建完整数据集...")
    
    for split_name, flight_ids, attacked_flights in [
        ('train', splits['train'], train_attacked_flights),
        ('val', splits['val'], val_attacked_flights),
        ('test', splits['test'], test_attacked_flights)
    ]:
        print(f"\n  处理{split_name}集...")
        
        # 获取该split的所有飞行
        split_df = merged_normal[merged_normal['flight'].isin(flight_ids)].copy()
        print(f"    Total flights: {split_df['flight'].nunique()}")
        print(f"    Total points: {len(split_df):,}")
        
        # 分离攻击和正常飞行
        normal_flights = [f for f in flight_ids if f not in attacked_flights]
        
        split_normal = split_df[split_df['flight'].isin(normal_flights)].copy()
        split_attacked = split_df[split_df['flight'].isin(attacked_flights)].copy()
        
        print(f"    Normal flights: {len(normal_flights)} ({len(split_normal):,} points)")
        print(f"    Attacked flights: {len(attacked_flights)} ({len(split_attacked):,} points)")
        
        # 保存正常飞行
        normal_path = output_dir / f'{split_name}_normal.csv'
        split_normal.to_csv(normal_path, index=False)
        print(f"    ✓ 保存正常飞行: {normal_path}")
        
        # 加载带攻击的飞行（已有攻击注入）
        attacked_path = output_dir / f'{split_name}_with_attacks.csv'
        if attacked_path.exists():
            split_attacked_injected = pd.read_csv(attacked_path, low_memory=False)
            print(f"    ✓ 加载攻击飞行: {attacked_path} ({len(split_attacked_injected):,} points)")
        else:
            print(f"    ✗ 未找到攻击文件: {attacked_path}")
            continue
        
        # 合并正常和攻击飞行
        split_complete = pd.concat([split_normal, split_attacked_injected], ignore_index=True)
        split_complete = split_complete.sort_values(['flight', 'timestamp']).reset_index(drop=True)
        
        print(f"    Complete dataset: {split_complete['flight'].nunique()} flights, {len(split_complete):,} points")
        
        # 保存完整数据集
        complete_path = output_dir / f'{split_name}_complete.csv'
        split_complete.to_csv(complete_path, index=False)
        print(f"    ✓ 保存完整数据集: {complete_path}")
        
        # 统计
        attack_ratio_flights = len(attacked_flights) / len(flight_ids)
        print(f"    Attack ratio (flights): {attack_ratio_flights*100:.1f}%")
    
    print("\n" + "="*80)
    print("✓ 训练数据修复完成!")
    print("="*80)
    
    print("\n下一步：使用*_complete.csv进行训练")
    print("  - train_complete.csv: 包含正常+攻击飞行")
    print("  - val_complete.csv: 包含正常+攻击飞行")
    print("  - test_complete.csv: 包含正常+攻击飞行")


if __name__ == "__main__":
    fix_training_data()
