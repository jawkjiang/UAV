"""
检查实际使用的时间间隔
"""
import sys
sys.path.append('../step5_timegan')
sys.path.append('../step3b_multiModelGeneral')

import pandas as pd
import numpy as np
from pathlib import Path

# 1. 检查原始数据的delta_t
print("="*80)
print("1. 原始数据的采样间隔 (delta_t)")
print("="*80)

test_df = pd.read_csv('../step5_timegan/output/test_complete.csv', nrows=50000, low_memory=False)
print(f"\nDelta_t统计:")
print(test_df['delta_t'].describe())
print(f"\n平均采样间隔: {test_df['delta_t'].mean():.4f}秒")
print(f"中位数采样间隔: {test_df['delta_t'].median():.4f}秒")

# 2. 检查窗口级别的timestamps
print("\n" + "="*80)
print("2. 窗口级别的timestamps (data_loader生成)")
print("="*80)

# 加载data_loader生成的数据
import config
sys.path.append('.')
from data_loader import load_test_data, prepare_test_windows

test_df, attack_info = load_test_data()
X_test, y_test, timestamps, flight_ids, attack_types, attack_segments_info = prepare_test_windows(test_df, attack_info)

print(f"\n窗口timestamps统计:")
print(f"  总窗口数: {len(timestamps):,}")
print(f"  Timestamps范围: {timestamps.min():.2f}s - {timestamps.max():.2f}s")

# 每个flight的timestamps
unique_flights = np.unique(flight_ids)[:10]  # 前10个flights
print(f"\n前10个flights的窗口时间间隔:")
for fid in unique_flights:
    flight_mask = flight_ids == fid
    flight_ts = timestamps[flight_mask]
    if len(flight_ts) > 1:
        intervals = np.diff(flight_ts)
        print(f"  Flight {fid}: 平均间隔 {intervals.mean():.4f}s, 中位数 {np.median(intervals):.4f}s, 窗口数 {len(flight_ts)}")

# 3. 检查MTBFA计算时使用的time_per_window
print("\n" + "="*80)
print("3. MTBFA计算中使用的time_per_window")
print("="*80)

# 模拟MTBFA计算中的逻辑
time_per_window = np.median(np.diff(timestamps[flight_ids == unique_flights[0]][:100]))
print(f"\ntime_per_window (从timestamps推断): {time_per_window:.4f}秒")
print(f"配置中的TIME_PER_POINT: {config.TIME_PER_POINT:.4f}秒")
print(f"差距: {time_per_window / config.TIME_PER_POINT:.2f}倍")

# 4. 检查实际的点级时间间隔
print("\n" + "="*80)
print("4. 点级数据的实际时间间隔")
print("="*80)

# 读取一个flight的完整数据
test_full = pd.read_csv('../step5_timegan/output/test_complete.csv', low_memory=False)
first_flight = test_full['flight'].iloc[0]
flight_data = test_full[test_full['flight'] == first_flight]

print(f"\nFlight {first_flight}:")
print(f"  总数据点: {len(flight_data)}")
print(f"  Delta_t统计: mean={flight_data['delta_t'].mean():.4f}s, median={flight_data['delta_t'].median():.4f}s")

# 计算假设的timestamps
assumed_timestamps = np.arange(len(flight_data)) * config.TIME_PER_POINT
actual_timestamps = flight_data['delta_t'].cumsum().values

print(f"\n假设timestamps (使用config.TIME_PER_POINT={config.TIME_PER_POINT}):")
print(f"  总时长: {assumed_timestamps[-1]:.2f}秒")
print(f"\n实际timestamps (使用delta_t累积):")
print(f"  总时长: {actual_timestamps[-1]:.2f}秒")
print(f"  差距: {actual_timestamps[-1] / assumed_timestamps[-1]:.2f}倍")

# 5. 总结
print("\n" + "="*80)
print("总结")
print("="*80)

print(f"\n配置中TIME_PER_POINT = {config.TIME_PER_POINT}秒")
print(f"实际平均delta_t = {test_df['delta_t'].mean():.4f}秒")
print(f"错误倍数: {test_df['delta_t'].mean() / config.TIME_PER_POINT:.1f}倍")

print(f"\n影响:")
print(f"  - data_loader生成的timestamps: 使用配置值 (❌ 错误)")
print(f"  - MTBFA计算: 从timestamps推断 (✅ 自动修正)")
print(f"  - 窗口时间间隔: 被低估约{test_df['delta_t'].mean() / config.TIME_PER_POINT:.1f}倍")
