"""
验证MTBFA计算的正确性

关键问题：
1. 测试集总飞行架次258次，总时长应该至少10h以上
2. TCN的总误报events数为9，但平均误报时间却只有0.3h
"""
import sys
sys.path.append('../step5_timegan')
sys.path.append('../step3b_multiModelGeneral')

import numpy as np
import pandas as pd
import config
import data_loader

print("="*80)
print("MTBFA计算验证")
print("="*80)

# 加载测试数据
print("\n[1] 加载测试数据...")
test_df, attack_info = data_loader.load_test_data()

X_test, y_test, timestamps, flight_ids, attack_types, attack_segments_info = \
    data_loader.prepare_test_windows(test_df, attack_info)

print("\n" + "="*80)
print("数据规模分析")
print("="*80)

# 基本统计
total_windows = len(y_test)
unique_flights = len(np.unique(flight_ids))
normal_windows = (y_test == 0).sum()
attack_windows = (y_test == 1).sum()

print(f"\n窗口统计:")
print(f"  总窗口数: {total_windows:,}")
print(f"  正常窗口: {normal_windows:,} ({normal_windows/total_windows*100:.1f}%)")
print(f"  攻击窗口: {attack_windows:,} ({attack_windows/total_windows*100:.1f}%)")
print(f"  飞行架次: {unique_flights}")

# 时间计算 - 关键点！
print(f"\n时间参数:")
print(f"  TIME_PER_POINT: {config.TIME_PER_POINT}秒/点")
print(f"  WINDOW_SIZE: {config.WINDOW_SIZE}点")
print(f"  STEP_SIZE: {config.STEP_SIZE}点")

step_time = config.STEP_SIZE * config.TIME_PER_POINT
window_span = config.WINDOW_SIZE * config.TIME_PER_POINT

print(f"\n时间跨度:")
print(f"  窗口时间跨度: {config.WINDOW_SIZE} × {config.TIME_PER_POINT}s = {window_span}s")
print(f"  窗口步长时间: {config.STEP_SIZE} × {config.TIME_PER_POINT}s = {step_time}s")

# 总时间计算
total_time_seconds = total_windows * step_time
total_time_hours = total_time_seconds / 3600
normal_time_seconds = normal_windows * step_time
normal_time_hours = normal_time_seconds / 3600

print(f"\n总时间计算 (使用步长):")
print(f"  总时间: {total_windows:,} × {step_time}s = {total_time_seconds:.1f}秒")
print(f"         = {total_time_seconds/60:.1f}分钟")
print(f"         = {total_time_hours:.2f}小时")
print(f"  正常时间: {normal_windows:,} × {step_time}s = {normal_time_seconds:.1f}秒")
print(f"           = {normal_time_seconds/60:.1f}分钟")
print(f"           = {normal_time_hours:.2f}小时")

# 每个飞行的平均时长
avg_windows_per_flight = total_windows / unique_flights
avg_time_per_flight = avg_windows_per_flight * step_time

print(f"\n每个飞行平均:")
print(f"  窗口数: {avg_windows_per_flight:.1f}")
print(f"  时长: {avg_time_per_flight:.1f}秒 = {avg_time_per_flight/60:.2f}分钟")

# 验证：如果有258个飞行，每个平均时长应该是多少才能达到10小时？
required_time_per_flight = 10 * 3600 / unique_flights
print(f"\n如果总时长要达到10小时:")
print(f"  需要每个飞行: {required_time_per_flight:.1f}秒 = {required_time_per_flight/60:.1f}分钟")
print(f"  实际每个飞行: {avg_time_per_flight:.1f}秒 = {avg_time_per_flight/60:.2f}分钟")
print(f"  差距: {required_time_per_flight / avg_time_per_flight:.1f}倍")

print("\n" + "="*80)
print("TCN模型MTBFA验证")
print("="*80)

# 加载TCN预测
data = np.load('output/tcn_test_predictions.npz')
y_pred = data['y_pred']

# 统计误报
false_alarms_windows = ((y_test == 0) & (y_pred == 1)).sum()

print(f"\nTCN误报统计:")
print(f"  误报窗口数: {false_alarms_windows}")

# 计算误报事件数（连续误报算一次）
false_alarm_events = 0
in_false_alarm = False

for flight_id in np.unique(flight_ids):
    flight_mask = (flight_ids == flight_id)
    flight_y_true = y_test[flight_mask]
    flight_y_pred = y_pred[flight_mask]
    
    for i in range(len(flight_y_true)):
        if flight_y_true[i] == 0 and flight_y_pred[i] == 1:  # False Positive
            if not in_false_alarm:
                false_alarm_events += 1
                in_false_alarm = True
        elif flight_y_true[i] == 0:  # Normal correctly predicted
            in_false_alarm = False
        else:  # Attack window
            in_false_alarm = False

print(f"  误报事件数: {false_alarm_events}")

# MTBFA计算
if false_alarm_events > 0:
    mtbfa_hours = normal_time_hours / false_alarm_events
    mtbfa_minutes = mtbfa_hours * 60
else:
    mtbfa_hours = float('inf')
    mtbfa_minutes = float('inf')

print(f"\nMTBFA计算:")
print(f"  公式: MTBFA = 总正常时间 / 误报事件数")
print(f"  计算: {normal_time_hours:.4f}h / {false_alarm_events} = {mtbfa_hours:.4f}h")
print(f"       = {mtbfa_minutes:.2f}分钟")

# 从CSV读取的值
overall_metrics = pd.read_csv('output/time_aware_metrics/overall_metrics.csv')
tcn_row = overall_metrics[overall_metrics['model'] == 'tcn'].iloc[0]
csv_mtbfa = tcn_row['MTBFA']
csv_fa = int(tcn_row['n_false_alarms'])

print(f"\nCSV中的值:")
print(f"  MTBFA: {csv_mtbfa:.4f}h = {csv_mtbfa*60:.2f}分钟")
print(f"  False Alarms: {csv_fa}")

print(f"\n验证:")
if abs(mtbfa_hours - csv_mtbfa) < 0.001:
    print(f"  ✅ 计算一致")
else:
    print(f"  ❌ 计算不一致")
    print(f"     期望: {mtbfa_hours:.4f}h")
    print(f"     实际: {csv_mtbfa:.4f}h")

if false_alarm_events == csv_fa:
    print(f"  ✅ 误报事件数一致")
else:
    print(f"  ❌ 误报事件数不一致")
    print(f"     期望: {false_alarm_events}")
    print(f"     实际: {csv_fa}")

print("\n" + "="*80)
print("结论")
print("="*80)

print(f"\n1. 测试集实际规模:")
print(f"   - 飞行架次: {unique_flights}")
print(f"   - 总时长: {total_time_hours:.2f}小时 (不是10小时以上)")
print(f"   - 每个飞行平均: {avg_time_per_flight/60:.2f}分钟")

print(f"\n2. MTBFA计算是否正确:")
if abs(mtbfa_hours - csv_mtbfa) < 0.001 and false_alarm_events == csv_fa:
    print(f"   ✅ MTBFA算法正确")
    print(f"   - 正常时间: {normal_time_hours:.2f}h")
    print(f"   - 误报事件: {false_alarm_events}")
    print(f"   - MTBFA: {mtbfa_hours:.4f}h = {mtbfa_minutes:.2f}分钟")
else:
    print(f"   ❌ MTBFA算法可能有问题")

print(f"\n3. 为什么MTBFA这么低?")
print(f"   - 测试集总时长只有{total_time_hours:.2f}小时，不是10小时以上")
print(f"   - 正常时间只有{normal_time_hours:.2f}小时")
print(f"   - {false_alarm_events}个误报事件分布在{normal_time_hours:.2f}小时内")
print(f"   - 所以MTBFA = {normal_time_hours:.2f}h / {false_alarm_events} = {mtbfa_hours:.4f}h")

print(f"\n4. 窗口化方法是否站得住脚?")
if total_time_hours < 1:
    print(f"   ⚠️  测试集总时长只有{total_time_hours:.2f}小时 ({total_time_seconds/60:.1f}分钟)")
    print(f"   ⚠️  这个规模可能不足以充分验证窗口化方法")
    print(f"   建议: 增加测试集规模或使用更长的飞行数据")
else:
    print(f"   ✅ 测试集总时长{total_time_hours:.2f}小时，规模合理")

print("\n" + "="*80)
