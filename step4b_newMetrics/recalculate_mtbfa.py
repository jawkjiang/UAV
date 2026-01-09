"""
重新计算正确的MTBFA
"""
import numpy as np
import pandas as pd

print("=" * 80)
print("MTBFA正确计算")
print("=" * 80)

# 加载数据
data_file = '../step3b_multiModelGeneral/output/cnn/test_predictions.npz'
data = np.load(data_file)
y_true = data['labels']
y_pred = (data['probabilities'] > 0.5).astype(int)
flight_ids = data['flight_ids']
window_indices = data['window_indices']
sampling_rate = int(data['sampling_rate'])
window_size = int(data['window_size'])

sampling_interval = 1.0 / sampling_rate
window_duration = window_size * sampling_interval
timestamps = window_indices * sampling_interval + window_duration / 2

print("\n【方法1：按飞行计算MTBFA】\n")

unique_flights = np.unique(flight_ids)
print(f"测试集飞行数: {len(unique_flights)}")

total_normal_time = 0  # 秒
total_false_alarms = 0

flight_details = []

for flight_id in unique_flights:
    # 获取这个飞行的所有窗口
    flight_mask = (flight_ids == flight_id)
    flight_timestamps = timestamps[flight_mask]
    flight_y_true = y_true[flight_mask]
    flight_y_pred = y_pred[flight_mask]
    
    # 飞行总时长（从第一个窗口到最后一个窗口）
    flight_duration = flight_timestamps.max() - flight_timestamps.min()
    
    # 统计正常窗口和攻击窗口
    n_normal = (flight_y_true == 0).sum()
    n_attack = (flight_y_true == 1).sum()
    
    # 估算正常时间（正常窗口数 * 窗口步长）
    # 注意：这是粗略估计，假设窗口均匀分布
    step_size = 5  # 从config中获取
    time_per_window = step_size / sampling_rate
    normal_time = n_normal * time_per_window
    
    # 统计误报（在正常窗口上预测为攻击）
    false_alarms_mask = (flight_y_true == 0) & (flight_y_pred == 1)
    
    # 连续误报算作一次
    fp_events = 0
    in_fp = False
    for i in range(len(flight_y_pred)):
        if flight_y_true[i] == 0 and flight_y_pred[i] == 1:
            if not in_fp:
                fp_events += 1
                in_fp = True
        elif flight_y_true[i] == 0:
            in_fp = False
        else:  # attack window
            in_fp = False
    
    total_normal_time += normal_time
    total_false_alarms += fp_events
    
    flight_details.append({
        'flight_id': flight_id,
        'duration': flight_duration,
        'n_windows': len(flight_timestamps),
        'n_normal': n_normal,
        'n_attack': n_attack,
        'normal_time': normal_time,
        'fp_events': fp_events
    })

df = pd.DataFrame(flight_details)

print(f"总正常时间: {total_normal_time:.2f}秒 = {total_normal_time/60:.2f}分钟 = {total_normal_time/3600:.2f}小时")
print(f"总误报事件: {total_false_alarms}")

if total_false_alarms > 0:
    mtbfa_hours = (total_normal_time / 3600) / total_false_alarms
    print(f"\nMTBFA = {total_normal_time/3600:.2f}h / {total_false_alarms} = {mtbfa_hours:.4f}小时")
    print(f"      = {mtbfa_hours * 60:.4f}分钟")
    print(f"      = {mtbfa_hours * 3600:.2f}秒")
    
    print(f"\n解释: 平均每 {mtbfa_hours*60:.2f}分钟 发生一次误报")
else:
    print(f"\nMTBFA = inf (无误报)")

print(f"\n有误报的飞行: {(df['fp_events'] > 0).sum()}/{len(df)}")
print(f"误报最严重的5个飞行:")
print(df.nlargest(5, 'fp_events')[['flight_id', 'normal_time', 'fp_events']].to_string(index=False))

print("\n" + "=" * 80)
print("\n【方法2：基于实际飞行时长估算】\n")

# 从数据集推断：43个飞行，总窗口7053个
# 平均每个飞行: 7053/43 ≈ 164个窗口
# 如果步长=5，采样率=100Hz，每个窗口代表0.05秒
# 平均飞行时长 ≈ 164 * 0.05 = 8.2秒

avg_windows_per_flight = len(timestamps) / len(unique_flights)
avg_flight_duration = avg_windows_per_flight * time_per_window

print(f"平均每个飞行窗口数: {avg_windows_per_flight:.1f}")
print(f"估算平均飞行时长: {avg_flight_duration:.2f}秒")
print(f"估算总飞行时长: {avg_flight_duration * len(unique_flights):.2f}秒 = {avg_flight_duration * len(unique_flights)/60:.2f}分钟")

# 如果70%是正常的
estimated_normal_ratio = (y_true == 0).sum() / len(y_true)
estimated_total_normal = avg_flight_duration * len(unique_flights) * estimated_normal_ratio

print(f"\n正常窗口比例: {estimated_normal_ratio*100:.1f}%")
print(f"估算总正常时间: {estimated_total_normal:.2f}秒 = {estimated_total_normal/60:.2f}分钟 = {estimated_total_normal/3600:.2f}小时")

if total_false_alarms > 0:
    mtbfa_v2 = (estimated_total_normal / 3600) / total_false_alarms
    print(f"\nMTBFA(方法2) = {estimated_total_normal/3600:.2f}h / {total_false_alarms} = {mtbfa_v2:.4f}小时 = {mtbfa_v2*60:.2f}分钟")

print("\n" + "=" * 80)
