"""
深入分析DR相同的真正原因
重点关注：检测延迟的计算逻辑
"""
import numpy as np
import pandas as pd
import sys
sys.path.append('../step5_timegan')
sys.path.append('../step3b_multiModelGeneral')

from data_loader import load_test_data, prepare_test_windows

print("="*80)
print("分析：检测延迟计算逻辑与DR相同的原因")
print("="*80)

# 加载测试数据和窗口
test_df, attack_info = load_test_data()
X_test, y_test, timestamps, flight_ids, attack_types, attack_segments_info = prepare_test_windows(test_df, attack_info)

# 加载模型预测
models = ['lstm', 'cnn', 'tcn', 'bilstm', 'cnn_lstm', 'gru', 'transformer']
predictions = {}
for model in models:
    data = np.load(f'output/{model}_test_predictions.npz')
    predictions[model] = data['y_pred']

print(f"\n加载了{len(models)}个模型的预测结果")
print(f"测试窗口数: {len(y_test)}")
print(f"攻击片段数: {len(attack_segments_info)}")

# 分析attack_segments_info
print("\n检查attack_segments_info的内容:")
print(f"总共{len(attack_segments_info)}个攻击片段")
print(f"\n前3个攻击片段示例:")
for i, seg in enumerate(attack_segments_info[:3]):
    print(f"\n片段{i}:")
    for key, value in seg.items():
        print(f"  {key}: {value}")

# 手动模拟detection delay的计算
print("\n" + "="*80)
print("手动计算检测延迟 (模拟time_aware_metrics的逻辑)")
print("="*80)

# 找到Flight 160的攻击信息
flight_160_info = None
for seg in attack_segments_info:
    if seg['flight_id'] == 160:
        flight_160_info = seg
        break

if flight_160_info:
    print(f"\nFlight 160 攻击信息:")
    print(f"  flight_id: {flight_160_info['flight_id']}")
    print(f"  attack_type: {flight_160_info['attack_type']}")
    print(f"  attack_start_time: {flight_160_info['attack_start_time']:.3f}s")
    
    # 找到Flight 160的所有窗口
    flight_160_mask = flight_ids == 160
    flight_160_indices = np.where(flight_160_mask)[0]
    flight_160_timestamps = timestamps[flight_160_mask]
    flight_160_y_true = y_test[flight_160_mask]
    
    # 找到攻击窗口
    attack_window_mask = flight_160_y_true == 1
    attack_window_indices = np.where(attack_window_mask)[0]  # 局部索引
    attack_window_indices_global = flight_160_indices[attack_window_indices]  # 全局索引
    
    print(f"\n  窗口信息:")
    print(f"    总窗口数: {len(flight_160_indices)}")
    print(f"    攻击窗口数: {len(attack_window_indices)}")
    print(f"    第一个攻击窗口时间: {flight_160_timestamps[attack_window_indices[0]]:.3f}s")
    
    # 对每个模型，找到首次检测的窗口
    print(f"\n  各模型检测延迟:")
    print(f"  {'模型':12s} {'首次检测窗口':>12s} {'检测时间(s)':>12s} {'延迟(s)':>10s}")
    print(f"  {'-'*50}")
    
    t_attack = flight_160_info['attack_start_time']
    
    for model in models:
        # 获取该flight的预测
        flight_preds = predictions[model][flight_160_mask]
        
        # 在攻击窗口中找首次检测
        first_detection = None
        for local_idx in attack_window_indices:
            if flight_preds[local_idx] == 1:
                first_detection = local_idx
                break
        
        if first_detection is not None:
            t_detect = flight_160_timestamps[first_detection]
            delay = t_detect - t_attack
            if delay < 0:
                delay = 0.0
            print(f"  {model.upper():12s} {first_detection:12d} {t_detect:12.3f} {delay:10.3f}")
        else:
            print(f"  {model.upper():12s} {'未检测':>12s}")

print("\n" + "="*80)
print("关键发现")
print("="*80)
print("\n检测延迟 = 首次检测窗口时间 - 攻击真实开始时间")
print("\n如果所有模型的首次检测窗口时间相同，则延迟相同")
print("如果延迟相同，则DR@Δt也会相同！")

# 统计所有攻击的情况
print("\n" + "="*80)
print("统计所有12个测试攻击的首次检测时间")
print("="*80)

same_detection_count = 0
diff_detection_count = 0

for seg in attack_segments_info[:12]:  # 只看前12个
    fid = seg['flight_id']
    t_attack = seg['attack_start_time']
    
    # 找到该flight的攻击窗口
    flight_mask = flight_ids == fid
    if flight_mask.sum() == 0:
        continue
    
    flight_indices = np.where(flight_mask)[0]
    flight_timestamps = timestamps[flight_mask]
    flight_y_true = y_test[flight_mask]
    
    attack_window_mask = flight_y_true == 1
    attack_window_indices = np.where(attack_window_mask)[0]
    
    if len(attack_window_indices) == 0:
        continue
    
    # 对每个模型找首次检测时间
    detection_times = {}
    for model in models:
        flight_preds = predictions[model][flight_mask]
        
        first_detection = None
        for local_idx in attack_window_indices:
            if flight_preds[local_idx] == 1:
                first_detection = local_idx
                break
        
        if first_detection is not None:
            detection_times[model] = flight_timestamps[first_detection]
        else:
            detection_times[model] = None
    
    # 检查是否所有模型的检测时间相同
    valid_times = [t for t in detection_times.values() if t is not None]
    
    if len(valid_times) > 0:
        if len(set(valid_times)) == 1:
            same_detection_count += 1
        else:
            diff_detection_count += 1
            print(f"\nFlight {fid}: 检测时间不同!")
            for model, t in detection_times.items():
                if t is not None:
                    delay = t - t_attack
                    if delay < 0:
                        delay = 0.0
                    print(f"  {model.upper():12s}: {t:.3f}s (delay={delay:.3f}s)")

print(f"\n统计结果:")
print(f"  所有模型首次检测时间相同: {same_detection_count}")
print(f"  模型首次检测时间不同: {diff_detection_count}")

if same_detection_count == 12:
    print(f"\n结论: 所有12个攻击，所有模型的首次检测时间都相同！")
    print(f"       这就是为什么所有模型的DR@Δt完全相同的原因。")
