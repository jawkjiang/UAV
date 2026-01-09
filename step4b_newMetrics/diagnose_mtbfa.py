"""
诊断MTBFA计算问题
"""
import numpy as np
import pandas as pd

print("=" * 80)
print("MTBFA诊断分析")
print("=" * 80)

# 加载数据
data_file = '../step3b_multiModelGeneral/output/cnn/test_predictions.npz'
meta_file = '../step3b_multiModelGeneral/output/cnn/window_metadata.csv'

print("\n【1. 加载数据】")
data = np.load(data_file)
y_true = data['labels']
y_pred = (data['probabilities'] > 0.5).astype(int)
flight_ids = data['flight_ids']
window_indices = data['window_indices']
sampling_rate = int(data['sampling_rate'])
window_size = int(data['window_size'])

# 计算时间戳
sampling_interval = 1.0 / sampling_rate
window_duration = window_size * sampling_interval
timestamps = window_indices * sampling_interval + window_duration / 2

print(f"  总窗口数: {len(y_true)}")
print(f"  飞行数: {len(np.unique(flight_ids))}")
print(f"  采样率: {sampling_rate}Hz")
print(f"  时间戳范围: {timestamps.min():.2f}s - {timestamps.max():.2f}s")

# 分析正常窗口
normal_mask = (y_true == 0)
print(f"\n【2. 正常窗口分析】")
print(f"  正常窗口数: {normal_mask.sum()} ({100*normal_mask.sum()/len(y_true):.1f}%)")
print(f"  攻击窗口数: {(~normal_mask).sum()} ({100*(~normal_mask).sum()/len(y_true):.1f}%)")

# 分析预测
print(f"\n【3. 预测分析】")
print(f"  预测为攻击的窗口: {(y_pred==1).sum()}")
print(f"  其中正确预测(TP): {((y_pred==1) & (y_true==1)).sum()}")
print(f"  其中误报(FP): {((y_pred==1) & (y_true==0)).sum()}")

# 手动计算MTBFA
print(f"\n【4. MTBFA手动计算】")

# 识别正常段
normal_segments = []
current_seg = None

for i in range(len(y_true)):
    if y_true[i] == 0:  # Normal
        if current_seg is None:
            current_seg = {'start': i, 'flight_id': flight_ids[i]}
        elif flight_ids[i] != current_seg['flight_id']:
            current_seg['end'] = i - 1
            normal_segments.append(current_seg)
            current_seg = {'start': i, 'flight_id': flight_ids[i]}
    else:  # Attack
        if current_seg is not None:
            current_seg['end'] = i - 1
            normal_segments.append(current_seg)
            current_seg = None

if current_seg is not None:
    current_seg['end'] = len(y_true) - 1
    normal_segments.append(current_seg)

print(f"  正常段数量: {len(normal_segments)}")

# 计算总正常时间和误报
total_normal_time = 0  # seconds
false_alarms = 0
false_alarm_details = []

for seg in normal_segments:
    duration = timestamps[seg['end']] - timestamps[seg['start']]
    total_normal_time += duration
    
    # 统计此段的误报
    seg_fp = 0
    in_false_alarm = False
    for i in range(seg['start'], seg['end'] + 1):
        if y_pred[i] == 1:
            if not in_false_alarm:
                false_alarms += 1
                seg_fp += 1
                in_false_alarm = True
        else:
            in_false_alarm = False
    
    if seg_fp > 0:
        false_alarm_details.append({
            'segment': len(false_alarm_details),
            'flight_id': seg['flight_id'],
            'duration_s': duration,
            'fp_count': seg_fp,
            'start_idx': seg['start'],
            'end_idx': seg['end']
        })

print(f"\n  总正常时间: {total_normal_time:.2f}秒 = {total_normal_time/60:.2f}分钟 = {total_normal_time/3600:.4f}小时")
print(f"  误报事件数: {false_alarms}")
print(f"  有误报的正常段: {len(false_alarm_details)}/{len(normal_segments)}")

if false_alarms > 0:
    mtbfa_seconds = total_normal_time / false_alarms
    mtbfa_minutes = mtbfa_seconds / 60
    mtbfa_hours = mtbfa_seconds / 3600
    
    print(f"\n  MTBFA = {total_normal_time:.2f}s / {false_alarms} = {mtbfa_seconds:.4f}秒")
    print(f"        = {mtbfa_minutes:.4f}分钟")
    print(f"        = {mtbfa_hours:.6f}小时")
    print(f"        = {mtbfa_hours:.2e}小时 (科学计数法)")
else:
    print(f"\n  MTBFA = inf (无误报)")

# 分析问题
print(f"\n【5. 问题诊断】")

# 检查时间计算
print(f"\n(a) 时间戳检查:")
print(f"  第一个窗口时间: {timestamps[0]:.2f}s")
print(f"  最后一个窗口时间: {timestamps[-1]:.2f}s")
print(f"  窗口间平均间隔: {np.mean(np.diff(timestamps)):.4f}s")

# 检查正常段的时间
if len(normal_segments) > 0:
    seg_durations = [timestamps[seg['end']] - timestamps[seg['start']] for seg in normal_segments[:5]]
    print(f"\n(b) 前5个正常段的持续时间:")
    for i, dur in enumerate(seg_durations):
        seg = normal_segments[i]
        print(f"  段{i}: {dur:.2f}s (索引 {seg['start']}-{seg['end']}, 窗口数={seg['end']-seg['start']+1})")

# 显示误报最多的段
if false_alarm_details:
    print(f"\n(c) 误报最多的5个正常段:")
    fa_df = pd.DataFrame(false_alarm_details)
    fa_df_sorted = fa_df.sort_values('fp_count', ascending=False).head(5)
    print(fa_df_sorted.to_string(index=False))

# 检查是否窗口连续性问题
print(f"\n(d) 窗口连续性检查:")
gaps = np.diff(window_indices)
print(f"  窗口索引差值统计:")
print(f"    最小: {gaps.min()}")
print(f"    最大: {gaps.max()}")
print(f"    平均: {gaps.mean():.1f}")
print(f"    中位数: {np.median(gaps):.1f}")
print(f"  大间隙(>1000)数量: {sum(gaps > 1000)}")

# 可能的问题
print(f"\n【6. 可能的问题】")
if total_normal_time < 60:
    print("  ⚠️  总正常时间过短(<1分钟)!")
    print("     → 可能原因: 窗口间的时间间隔计算有误")
    print("     → 检查: timestamps[end] - timestamps[start] 是否正确反映实际时间")

if false_alarms > 100:
    print(f"  ⚠️  误报数量过多({false_alarms}个)!")
    print("     → 可能原因: 模型阈值设置不当，或数据不平衡")

if total_normal_time / 3600 < 1 and false_alarms > 10:
    print(f"  ⚠️  MTBFA极低的根本原因:")
    print(f"     正常时间仅 {total_normal_time/3600:.4f}小时")
    print(f"     误报却有 {false_alarms} 个")
    print(f"     → 平均每 {total_normal_time/false_alarms:.2f}秒 就有一次误报!")

print("\n" + "=" * 80)
