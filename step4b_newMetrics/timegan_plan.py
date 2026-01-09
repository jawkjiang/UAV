"""
TimeGAN数据扩充方案分析
基于当前数据集规模和硬件条件
"""
import numpy as np
import pandas as pd

print("=" * 80)
print("TimeGAN数据扩充方案设计")
print("=" * 80)

# 1. 分析当前数据规模
print("\n【1. 当前数据规模分析】")

# 从测试集推断
test_data = np.load('../step3b_multiModelGeneral/output/cnn/test_predictions.npz')
test_windows = len(test_data['labels'])
test_attacks = (test_data['labels'] == 1).sum()
test_flights = len(np.unique(test_data['flight_ids']))

print(f"\n测试集:")
print(f"  飞行数: {test_flights}")
print(f"  总窗口: {test_windows}")
print(f"  攻击窗口: {test_attacks} ({100*test_attacks/test_windows:.1f}%)")
print(f"  正常窗口: {test_windows - test_attacks} ({100*(test_windows-test_attacks)/test_windows:.1f}%)")

# 推断训练集规模（从README配置）
train_flights = 125  # 从step3b README得知
val_flights = 41
total_flights = 209

# 假设每个飞行平均窗口数相似
avg_windows_per_flight = test_windows / test_flights
train_windows_est = int(train_flights * avg_windows_per_flight)
val_windows_est = int(val_flights * avg_windows_per_flight)

print(f"\n训练集（估算）:")
print(f"  飞行数: {train_flights}")
print(f"  估算总窗口: {train_windows_est}")
print(f"  攻击比例: 60% (配置值)")
print(f"  攻击窗口: {int(train_windows_est * 0.6)}")

print(f"\n验证集（估算）:")
print(f"  飞行数: {val_flights}")
print(f"  估算总窗口: {val_windows_est}")
print(f"  攻击比例: 50% (配置值)")
print(f"  攻击窗口: {int(val_windows_est * 0.5)}")

print("\n【2. TimeGAN扩充级别选择】")

print("\n方案对比:")
print("┌─────────────┬──────────────────┬──────────────────┬─────────────┐")
print("│   级别      │   序列长度       │   训练难度       │   推荐度    │")
print("├─────────────┼──────────────────┼──────────────────┼─────────────┤")
print("│ Flight级别  │   几百到几千点   │   高（慢）       │   ⭐⭐     │")
print("│             │   (整个飞行)     │   需要GPU        │   (不推荐)  │")
print("├─────────────┼──────────────────┼──────────────────┼─────────────┤")
print("│ Window级别  │   50点           │   低（快）       │   ⭐⭐⭐⭐ │")
print("│             │   (单个窗口)     │   CPU即可        │   (强烈推荐)│")
print("├─────────────┼──────────────────┼──────────────────┼─────────────┤")
print("│ Segment级别 │   200-500点      │   中等           │   ⭐⭐⭐   │")
print("│             │   (攻击片段)     │   建议GPU        │   (可考虑)  │")
print("└─────────────┴──────────────────┴──────────────────┴─────────────┘")

print("\n✓ 推荐：Window级别")
print("  理由：")
print("  1. 窗口大小50点，TimeGAN训练快（2-5分钟）")
print("  2. 符合你的模型输入（窗口级检测）")
print("  3. 可以针对不同攻击类型分别生成")
print("  4. 容易控制生成样本的数量和质量")

print("\n【3. 扩充倍数设计】")

# 计算不同扩充倍数下的效果
print("\n当前训练集攻击窗口估算:", int(train_windows_est * 0.6))

scenarios = [
    {"name": "保守扩充", "factor": 2, "desc": "2倍扩充"},
    {"name": "中等扩充", "factor": 3, "desc": "3倍扩充"},
    {"name": "激进扩充", "factor": 5, "desc": "5倍扩充"},
]

print("\n扩充方案对比:")
print(f"{'方案':<12} {'扩充倍数':>8} {'原攻击窗口':>12} {'生成窗口':>10} {'总攻击窗口':>12} {'训练时间(估)':>12}")
print("-" * 80)

original_attack = int(train_windows_est * 0.6)
for scenario in scenarios:
    factor = scenario['factor']
    generated = original_attack * (factor - 1)
    total = original_attack * factor
    training_time = f"{factor * 2}-{factor * 3}分钟"
    
    print(f"{scenario['name']:<12} {factor:>8}x {original_attack:>12} {generated:>10} {total:>12} {training_time:>12}")

print("\n✓ 推荐：3倍扩充（中等方案）")
print("  - 生成窗口数: ~24,600个")
print("  - TimeGAN训练时间: 6-9分钟（单攻击类型）")
print("  - 数据多样性: 较好")
print("  - 过拟合风险: 中等")

print("\n【4. 正样本率对齐方案】")

print("\n目标: 解决训练-测试分布不匹配问题")
print("\n当前问题:")
print(f"  训练集: 60% 攻击 ❌")
print(f"  验证集: 50% 攻击 ❌")
print(f"  测试集:  3% 攻击 ✓ (真实场景)")
print(f"  → 分布严重不匹配，导致高FPR")

print("\n方案A: 完全对齐（不推荐）")
print("  训练集: 3% 攻击")
print("  验证集: 3% 攻击")
print("  测试集: 3% 攻击")
print("  ❌ 问题: 攻击样本太少，模型学不到攻击特征")

print("\n方案B: 渐进对齐（推荐）")
print("  训练集: 20-25% 攻击 ← 用TimeGAN扩充正常样本或减少攻击比例")
print("  验证集: 10-15% 攻击 ← 接近测试集但保留足够样本")
print("  测试集:  3% 攻击 ← 保持真实分布")
print("  ✓ 优点: 既有足够训练样本，又接近真实分布")

print("\n方案C: 混合策略（最佳，论文级）")
print("  训练集: 两部分")
print("    - 高攻击集: 40% 攻击（学习攻击特征）")
print("    - 低攻击集: 5% 攻击（学习真实分布）")
print("    - 混合训练或Curriculum Learning")
print("  验证集: 10% 攻击")
print("  测试集: 3% 攻击")
print("  ✓ 优点: 最佳性能，可作为论文创新点")

print("\n【5. 具体实施方案】")

print("\n✓ Window级别 + 3倍扩充 + 方案B对齐")

print("\n步骤1: 准备原始攻击窗口")
print("  - 从训练集提取所有攻击窗口")
print("  - 按攻击类型分组（6种攻击）")
print("  - 每种攻击约2000-2500个窗口")

print("\n步骤2: 训练TimeGAN（每种攻击类型独立训练）")
print("  攻击类型:")
attack_types = ['step', 'drift_ramp', 'drift_sigmoid', 'delay', 'takeover_step', 'takeover_ramp']
for i, att in enumerate(attack_types, 1):
    print(f"  {i}. {att:15s} - TimeGAN训练 3-5分钟")

print(f"\n  总训练时间: {len(attack_types) * 4}分钟 (并行: {4}分钟)")

print("\n步骤3: 生成合成样本")
print("  - 每种攻击类型生成原始数量的2倍")
print("  - 总生成窗口: ~24,600")
print("  - 验证质量: 可视化检查 + 判别器测试")

print("\n步骤4: 重组训练集（正样本率对齐）")
print("  原始:")
print(f"    攻击: {original_attack} (60%)")
print(f"    正常: {int(train_windows_est * 0.4)} (40%)")
print("\n  策略1: 扩充正常样本（推荐）")
target_normal = int(original_attack * 3 * (1 - 0.25) / 0.25)  # 目标25%攻击
print(f"    攻击: {original_attack * 3} (25%，含TimeGAN)")
print(f"    正常: {target_normal} (75%，简单增强扩充)")
print("\n  策略2: 仅用TimeGAN攻击，减少原始攻击")
keep_attack = int(original_attack * 0.5)
timegan_attack = int(original_attack * 0.5)
print(f"    攻击: {keep_attack} 原始 + {timegan_attack} TimeGAN = {keep_attack + timegan_attack} (25%)")
print(f"    正常: {int((keep_attack + timegan_attack) * 3)} (75%)")

print("\n步骤5: 重新训练模型")
print("  - 使用新的平衡数据集")
print("  - 训练时间: 2-3分钟（和现在差不多）")

print("\n【6. 预期效果】")

print("\n性能提升预测:")
print("┌──────────────┬─────────────┬─────────────┬──────────────┐")
print("│   指标       │   当前      │   预期      │   改善       │")
print("├──────────────┼─────────────┼─────────────┼──────────────┤")
print("│ TPR          │   99.05%    │   95-97%    │   轻微下降   │")
print("│ FPR          │   5.51%     │   2-3%      │   降低50%    │")
print("│ MTBFA        │   18秒      │   60-90秒   │   3-5倍      │")
print("│ Precision    │   35.67%    │   55-65%    │   大幅提升   │")
print("│ F1 Score     │   52.45%    │   70-75%    │   显著提升   │")
print("└──────────────┴─────────────┴─────────────┴──────────────┘")

print("\n论文贡献点:")
print("  1. TimeGAN用于GPS欺骗攻击合成（创新点）")
print("  2. 类别平衡策略降低误报（实用性）")
print("  3. 训练-测试分布对齐（方法论）")
print("  4. 消融实验: 真实vs增强vs TimeGAN（完整性）")

print("\n【7. 时间成本估算】")

print("\n总时间投入（你的硬件条件）:")
print("  1. 数据准备: 30分钟")
print("  2. TimeGAN训练（6种攻击，并行）: 10-15分钟")
print("  3. 样本生成和质量检查: 20分钟")
print("  4. 重组数据集: 10分钟")
print("  5. 重新训练7个模型: 15-20分钟")
print("  6. 评估和可视化: 10分钟")
print("\n  总计: ~2小时（一次性投入）")
print("  后续微调: 每次30分钟")

print("\n✓ 对于发论文，这个时间投入是完全值得的！")

print("\n" + "=" * 80)
