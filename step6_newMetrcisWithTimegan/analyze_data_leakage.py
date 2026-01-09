"""
深度分析：检测TimeGAN流程中的数据泄漏
重点检查：
1. 训练集/测试集是否来自相同的原始flights
2. TimeGAN是否在训练时看到了测试集flights
3. 归一化参数是否在测试集上计算
4. 特征工程是否使用了测试集信息
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
sys.path.append('../step5_timegan')
sys.path.append('../step3b_multiModelGeneral')

print("="*80)
print("数据泄漏深度分析")
print("="*80)

# ============================================================================
# 1. 检查数据集划分
# ============================================================================
print("\n" + "="*80)
print("1. 检查数据集划分")
print("="*80)

step5_output = Path('../step5_timegan/output')

# 加载划分信息
splits_file = step5_output / 'flight_splits.json'
if splits_file.exists():
    with open(splits_file) as f:
        splits = json.load(f)
    
    train_flights = set(splits['train'])
    val_flights = set(splits['val'])
    test_flights = set(splits['test'])
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_flights)} flights")
    print(f"  验证集: {len(val_flights)} flights")
    print(f"  测试集: {len(test_flights)} flights")
    
    # 检查是否有重叠
    train_val_overlap = train_flights & val_flights
    train_test_overlap = train_flights & test_flights
    val_test_overlap = val_flights & test_flights
    
    print(f"\n检查重叠:")
    print(f"  Train ∩ Val: {len(train_val_overlap)}")
    print(f"  Train ∩ Test: {len(train_test_overlap)}")
    print(f"  Val ∩ Test: {len(val_test_overlap)}")
    
    if train_test_overlap:
        print(f"\n⚠️ 警告: 训练集和测试集有重叠flights: {train_test_overlap}")
    else:
        print(f"\n✓ 训练集和测试集没有重叠")

# ============================================================================
# 2. 检查TimeGAN训练数据来源
# ============================================================================
print("\n" + "="*80)
print("2. 检查TimeGAN训练数据来源")
print("="*80)

normal_flights_file = step5_output / 'normal_flights.csv'
if normal_flights_file.exists():
    print(f"\n加载TimeGAN训练数据源: {normal_flights_file}")
    normal_df = pd.read_csv(normal_flights_file, low_memory=False)
    
    timegan_source_flights = set(normal_df['flight'].unique())
    print(f"TimeGAN训练使用的flights数: {len(timegan_source_flights)}")
    
    # 检查TimeGAN训练数据是否包含测试集flights
    timegan_test_overlap = timegan_source_flights & test_flights
    
    print(f"\n关键检查: TimeGAN训练数据是否包含测试集flights?")
    print(f"  TimeGAN ∩ Test flights: {len(timegan_test_overlap)}")
    
    if timegan_test_overlap:
        print(f"\n🚨 严重警告: TimeGAN在训练时看到了测试集flights!")
        print(f"  重叠flights: {sorted(list(timegan_test_overlap))[:20]}...")
        print(f"  重叠比例: {len(timegan_test_overlap)/len(test_flights)*100:.1f}%")
        print(f"\n这是数据泄漏! TimeGAN学习了测试集的模式。")
    else:
        print(f"\n✓ TimeGAN训练数据不包含测试集flights")

# ============================================================================
# 3. 检查原始数据来源
# ============================================================================
print("\n" + "="*80)
print("3. 检查原始数据来源")
print("="*80)

# 加载原始flights.csv
original_data_path = Path('../data/src/flights.csv')
if original_data_path.exists():
    print(f"\n加载原始数据: {original_data_path}")
    original_df = pd.read_csv(original_data_path, low_memory=False)
    
    original_flights = set(original_df['flight'].unique())
    print(f"原始数据总flights: {len(original_flights)}")
    
    # 检查训练集、测试集是否都来自原始数据
    print(f"\n检查数据来源:")
    print(f"  训练集全部来自原始数据: {train_flights.issubset(original_flights)}")
    print(f"  测试集全部来自原始数据: {test_flights.issubset(original_flights)}")
    
    # 检查是否所有flights都被使用
    all_used_flights = train_flights | val_flights | test_flights
    unused_flights = original_flights - all_used_flights
    
    print(f"\n原始数据使用情况:")
    print(f"  已使用: {len(all_used_flights)} / {len(original_flights)}")
    print(f"  未使用: {len(unused_flights)}")

# ============================================================================
# 4. 检查合成数据
# ============================================================================
print("\n" + "="*80)
print("4. 检查合成数据（TimeGAN生成）")
print("="*80)

synthetic_file = step5_output / 'synthetic_normal_flights.csv'
if synthetic_file.exists():
    print(f"\n加载合成数据: {synthetic_file}")
    synthetic_df = pd.read_csv(synthetic_file)
    
    synthetic_flights = set(synthetic_df['flight'].unique())
    print(f"合成flights数: {len(synthetic_flights)}")
    
    # 检查合成flights的编号范围
    synthetic_ids = list(synthetic_flights)
    print(f"  编号范围: {min(synthetic_ids)} ~ {max(synthetic_ids)}")
    
    # 检查合成flights是否与原始flights重叠
    synthetic_original_overlap = synthetic_flights & original_flights
    
    if synthetic_original_overlap:
        print(f"\n⚠️ 警告: 合成flights与原始flights编号重叠!")
        print(f"  重叠数: {len(synthetic_original_overlap)}")
    else:
        print(f"\n✓ 合成flights与原始flights编号不重叠")

# ============================================================================
# 5. 检查最终训练/测试数据
# ============================================================================
print("\n" + "="*80)
print("5. 检查最终训练/测试数据（带攻击）")
print("="*80)

train_final = step5_output / 'train_with_attacks.csv'
test_final = step5_output / 'test_with_attacks.csv'

if train_final.exists() and test_final.exists():
    print(f"\n加载最终数据:")
    train_df = pd.read_csv(train_final)
    test_df = pd.read_csv(test_final)
    
    train_final_flights = set(train_df['flight'].unique())
    test_final_flights = set(test_df['flight'].unique())
    
    print(f"  训练集flights: {len(train_final_flights)}")
    print(f"  测试集flights: {len(test_final_flights)}")
    
    # 检查最终数据的重叠
    final_overlap = train_final_flights & test_final_flights
    
    if final_overlap:
        print(f"\n🚨 严重警告: 最终训练集和测试集有重叠!")
        print(f"  重叠flights: {final_overlap}")
    else:
        print(f"\n✓ 最终训练集和测试集没有重叠")
    
    # 检查攻击注入情况
    print(f"\n攻击注入情况:")
    
    # 加载攻击信息
    train_attack_info = pd.read_csv(step5_output / 'train_attack_info.csv')
    test_attack_info = pd.read_csv(step5_output / 'test_attack_info.csv')
    
    # 检查列名
    flight_col = 'flight_id' if 'flight_id' in train_attack_info.columns else 'flight'
    
    train_attacked_flights = set(train_attack_info[flight_col].unique())
    test_attacked_flights = set(test_attack_info[flight_col].unique())
    
    print(f"  训练集攻击flights: {len(train_attacked_flights)}")
    print(f"  测试集攻击flights: {len(test_attacked_flights)}")
    
    # 检查攻击flights是否重叠
    attack_overlap = train_attacked_flights & test_attacked_flights
    
    if attack_overlap:
        print(f"\n⚠️ 警告: 训练集和测试集的攻击注入到了相同的flights!")
        print(f"  重叠flights: {attack_overlap}")
    else:
        print(f"\n✓ 训练集和测试集的攻击flights不重叠")

# ============================================================================
# 6. 总结与结论
# ============================================================================
print("\n" + "="*80)
print("总结与结论")
print("="*80)

leakage_found = []

if 'timegan_test_overlap' in locals() and timegan_test_overlap:
    leakage_found.append(
        f"TimeGAN训练数据包含测试集flights ({len(timegan_test_overlap)} flights)"
    )

if 'final_overlap' in locals() and final_overlap:
    leakage_found.append(
        f"最终训练集和测试集有重叠flights ({len(final_overlap)} flights)"
    )

if 'attack_overlap' in locals() and attack_overlap:
    leakage_found.append(
        f"相同flights在训练集和测试集都被注入了攻击 ({len(attack_overlap)} flights)"
    )

if leakage_found:
    print("\n🚨 发现数据泄漏:")
    for i, issue in enumerate(leakage_found, 1):
        print(f"  {i}. {issue}")
    
    print("\n影响:")
    print("  - 模型可能记住了测试集的模式")
    print("  - 测试性能被人为提高")
    print("  - 所有模型在相同位置检测到攻击可能因为都'记住'了这些patterns")
    
    print("\n建议:")
    print("  1. 重新生成数据集，确保TimeGAN只在训练flights上训练")
    print("  2. 严格分离训练集和测试集的flights")
    print("  3. 使用新的测试集flights（从未在训练中见过）")
else:
    print("\n✓ 未发现明显的数据泄漏")
    print("\n但所有模型预测高度一致仍然值得关注:")
    print("  - 可能攻击模式过于简单")
    print("  - 可能TimeGAN生成的数据质量过高（完美复制）")
    print("  - 需要更难的测试场景")
