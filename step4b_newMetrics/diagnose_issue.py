"""
诊断脚本：分析时间戳和检测延迟的问题
"""
import numpy as np
import pandas as pd
import sys

print("="*80)
print("问题诊断")
print("="*80)

# 1. 加载数据
pred_file = '../step3b_multiModelGeneral/output/tcn/test_predictions.npz'
info_file = '../step3b_multiModelGeneral/output/tcn/test_attack_info.csv'

data = np.load(pred_file)
attack_info = pd.read_csv(info_file)

y_true = data['labels']
y_pred_prob = data['probabilities']  # 注意：这是概率，不是二值化预测！
y_pred_binary = (y_pred_prob > 0.5).astype(int)

print(f"\n1. 数据基本信息:")
print(f"   窗口数量: {len(y_true)}")
print(f"   飞行数量: {len(attack_info)}")
print(f"   真实标签 - Attack窗口: {(y_true == 1).sum()}, Normal窗口: {(y_true == 0).sum()}")
print(f"   预测标签 - Attack窗口: {(y_pred_binary == 1).sum()}, Normal窗口: {(y_pred_binary == 0).sum()}")

# 2. 时间戳计算问题
WINDOW_SIZE = 50
STEP_SIZE = 5
TIME_PER_SAMPLE = 0.01  # 100Hz采样率，每个采样点0.01s

print(f"\n2. 当前时间戳计算（错误的）:")
print(f"   TIME_PER_SAMPLE = {TIME_PER_SAMPLE}s (每个采样点)")
print(f"   STEP_SIZE = {STEP_SIZE}个采样点")
print(f"   窗口间时间间隔 = STEP_SIZE * TIME_PER_SAMPLE = {STEP_SIZE * TIME_PER_SAMPLE}s")
print(f"   问题：这是窗口之间的步长时间，不是窗口持续时间！")

print(f"\n3. 正确的时间戳计算应该是:")
window_duration = WINDOW_SIZE * TIME_PER_SAMPLE
window_step_time = STEP_SIZE * TIME_PER_SAMPLE
print(f"   窗口大小 = {WINDOW_SIZE}个采样点")
print(f"   窗口持续时间 = WINDOW_SIZE * TIME_PER_SAMPLE = {window_duration}s")
print(f"   窗口步长时间 = STEP_SIZE * TIME_PER_SAMPLE = {window_step_time}s")

# 4. 查看attack_info中的实际时间
print(f"\n4. attack_info中的攻击开始时间 (前5个被攻击的飞行):")
attacked_flights = attack_info[attack_info['attacked'] == True].head(5)
for idx, row in attacked_flights.iterrows():
    print(f"   Flight {row['flight']}: attack_start_time = {row['attack_start_time']:.2f}s")

print(f"\n5. 关键问题分析:")
print(f"   ❌ 错误1: 使用 'predictions' 键，但这个键存储的是概率值，不是二值化预测！")
print(f"      - 应该使用 y_pred = (probabilities > 0.5).astype(int)")
print(f"   ")
print(f"   ❌ 错误2: 时间戳重建不正确")
print(f"      - 当前: timestamps = np.arange(n_windows) * 0.05s")
print(f"      - 这只能提供0.05s的粒度，无法反映真实的攻击时间")
print(f"      - 窗口0: t=0s, 窗口1: t=0.05s, 窗口2: t=0.1s ... ")
print(f"   ")
print(f"   ❌ 错误3: 攻击片段识别问题")
print(f"      - attack_info包含真实的attack_start_time（如95.8s, 165.5s等）")
print(f"      - 但我们的时间戳只是简单的0, 0.05, 0.1...序列")
print(f"      - 完全无法对应！")

# 6. 检查为什么DR不随Δt变化
print(f"\n6. 检查首次检测的时间分布:")
current_timestamps = np.arange(len(y_true)) * STEP_SIZE * TIME_PER_SAMPLE

# 模拟当前的攻击片段识别
in_attack = False
attack_segments = []
for i in range(len(y_true)):
    if y_true[i] == 1 and not in_attack:
        attack_segments.append({
            't_start': current_timestamps[i],
            'start_idx': i
        })
        in_attack = True
    elif y_true[i] == 0 and in_attack:
        in_attack = False

print(f"   识别到的攻击片段数: {len(attack_segments)}")
if len(attack_segments) > 0:
    print(f"   前5个攻击片段的开始时间:")
    for seg in attack_segments[:5]:
        print(f"      - Attack starts at window {seg['start_idx']}, t={seg['t_start']:.3f}s")
        # 查找首次检测
        for j in range(seg['start_idx'], min(seg['start_idx']+100, len(y_pred_binary))):
            if y_pred_binary[j] == 1:
                delay = current_timestamps[j] - seg['t_start']
                print(f"        First detection at window {j}, delay={delay:.3f}s")
                break

print(f"\n7. 根本原因:")
print(f"   - 我们没有真实的时间戳数据！")
print(f"   - test_predictions.npz只包含：labels, predictions/probabilities")
print(f"   - test_attack_info.csv包含：flight, attacked, attack_type, attack_start_time")
print(f"   - 缺失：每个窗口对应的真实时间戳")
print(f"   ")
print(f"   解决方案需要:")
print(f"   1. 使用probabilities进行二值化（而不是直接用predictions键）")
print(f"   2. 需要从step3b重新获取或重建窗口级的时间戳映射")
print(f"   3. 或者基于窗口索引计算相对延迟（单位：窗口数，然后转换为秒）")

print("="*80)
