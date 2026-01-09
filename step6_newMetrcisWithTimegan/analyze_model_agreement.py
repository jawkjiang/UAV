"""
分析为什么所有模型都在相同位置检测到攻击
"""
import numpy as np
import pandas as pd
import sys
sys.path.append('../step5_timegan')
sys.path.append('../step3b_multiModelGeneral')

from data_loader import load_test_data, prepare_test_windows

print("="*80)
print("为什么所有模型都在相同窗口检测到攻击？")
print("="*80)

# 加载测试数据
test_df, attack_info = load_test_data()
X_test, y_test, timestamps, flight_ids, attack_types, attack_segments_info = prepare_test_windows(test_df, attack_info)

# 加载模型预测
models = ['lstm', 'cnn', 'tcn', 'bilstm', 'cnn_lstm', 'gru', 'transformer']
predictions = {}
pred_probs = {}  # 预测概率

for model in models:
    data = np.load(f'output/{model}_test_predictions.npz')
    predictions[model] = data['y_pred']
    if 'y_pred_proba' in data:
        pred_probs[model] = data['y_pred_proba']

print("\n1. 检查Flight 160的详细预测情况")
print("="*80)

flight_160_mask = flight_ids == 160
flight_160_indices = np.where(flight_160_mask)[0]
flight_160_timestamps = timestamps[flight_160_mask]
flight_160_y_true = y_test[flight_160_mask]

# 找到攻击窗口
attack_window_mask = flight_160_y_true == 1
attack_window_indices = np.where(attack_window_mask)[0]

first_attack_idx = attack_window_indices[0]
print(f"\n第一个攻击窗口: 局部索引{first_attack_idx}, 时间{flight_160_timestamps[first_attack_idx]:.3f}s")

# 检查首个攻击窗口及其前后的预测
print(f"\n各模型在首个攻击窗口前后的二值预测:")
print(f"{'窗口':>4s} {'时间':>8s} {'真实':>4s}", end='')
for model in models:
    print(f" {model.upper():>6s}", end='')
print()
print("-" * (24 + 7 * len(models)))

for i in range(max(0, first_attack_idx-5), min(len(flight_160_y_true), first_attack_idx+10)):
    print(f"{i:4d} {flight_160_timestamps[i]:8.2f} {flight_160_y_true[i]:4d}", end='')
    global_idx = flight_160_indices[i]
    for model in models:
        pred = predictions[model][global_idx]
        print(f" {pred:6d}", end='')
    marker = " <-- 首个攻击窗口" if i == first_attack_idx else ""
    print(marker)

# 检查预测概率
if pred_probs:
    print(f"\n各模型在首个攻击窗口前后的预测概率:")
    print(f"{'窗口':>4s} {'时间':>8s} {'真实':>4s}", end='')
    for model in models:
        print(f" {model.upper():>8s}", end='')
    print()
    print("-" * (24 + 9 * len(models)))
    
    for i in range(max(0, first_attack_idx-5), min(len(flight_160_y_true), first_attack_idx+10)):
        print(f"{i:4d} {flight_160_timestamps[i]:8.2f} {flight_160_y_true[i]:4d}", end='')
        global_idx = flight_160_indices[i]
        for model in models:
            if model in pred_probs:
                prob = pred_probs[model][global_idx]
                print(f" {prob:8.4f}", end='')
            else:
                print(f" {'N/A':>8s}", end='')
        marker = " <-- 首个攻击窗口" if i == first_attack_idx else ""
        print(marker)

print("\n2. 分析所有攻击窗口的模型一致性")
print("="*80)

# 对所有攻击窗口，检查模型预测的一致性
all_attack_indices = np.where(y_test == 1)[0]
print(f"\n总攻击窗口数: {len(all_attack_indices)}")

# 计算模型预测的一致性
agreement_counts = np.zeros(len(models) + 1, dtype=int)  # 0到7个模型同意

for idx in all_attack_indices:
    # 计算有多少模型预测为1
    agree_count = sum(predictions[model][idx] == 1 for model in models)
    agreement_counts[agree_count] += 1

print(f"\n攻击窗口预测一致性分布:")
print(f"{'同意数':>6s} {'窗口数':>8s} {'百分比':>8s}")
print("-" * 24)
for i, count in enumerate(agreement_counts):
    if count > 0:
        pct = count / len(all_attack_indices) * 100
        print(f"{i:6d} {count:8d} {pct:7.2f}%")

# 找出所有模型都预测为1的攻击窗口
all_agree_attack = np.array([
    idx for idx in all_attack_indices 
    if all(predictions[model][idx] == 1 for model in models)
])

print(f"\n所有7个模型都预测为1的攻击窗口: {len(all_agree_attack)} / {len(all_attack_indices)} ({len(all_agree_attack)/len(all_attack_indices)*100:.1f}%)")

# 找出所有模型都预测为0的攻击窗口
all_disagree_attack = np.array([
    idx for idx in all_attack_indices 
    if all(predictions[model][idx] == 0 for model in models)
])

print(f"所有7个模型都预测为0的攻击窗口: {len(all_disagree_attack)} / {len(all_attack_indices)} ({len(all_disagree_attack)/len(all_attack_indices)*100:.1f}%)")

print("\n3. 检查正常窗口的预测一致性")
print("="*80)

all_normal_indices = np.where(y_test == 0)[0]
print(f"\n总正常窗口数: {len(all_normal_indices)}")

# 计算正常窗口的模型预测一致性
normal_agreement = np.zeros(len(models) + 1, dtype=int)

for idx in all_normal_indices:
    agree_count = sum(predictions[model][idx] == 0 for model in models)
    normal_agreement[agree_count] += 1

print(f"\n正常窗口预测一致性分布 (预测为0的模型数):")
print(f"{'同意数':>6s} {'窗口数':>8s} {'百分比':>8s}")
print("-" * 24)
for i, count in enumerate(normal_agreement):
    if count > 0:
        pct = count / len(all_normal_indices) * 100
        print(f"{i:6d} {count:8d} {pct:7.2f}%")

print("\n" + "="*80)
print("关键结论")
print("="*80)
print("\n如果:")
print("  - 大部分攻击窗口所有模型都预测为1")
print("  - 大部分正常窗口所有模型都预测为0")
print("\n则说明:")
print("  - 所有模型学到了几乎相同的决策边界")
print("  - 数据中的模式非常明显，不同架构都能轻易识别")
print("  - 这会导致所有模型在相同位置首次检测到攻击")
