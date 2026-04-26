"""
诊断指标虚高问题
检查可能导致高性能的数据泄漏或评估问题
"""
import pandas as pd
import numpy as np
import sys
sys.path.append('../step3b_multiModelGeneral')

import config
from labeling import generate_point_labels
from feature_engineering import get_feature_columns
from window_creation import create_windows_from_dataset


def diagnose_test_data():
    """诊断测试数据"""
    print("="*80)
    print("DIAGNOSIS 1: Test Data Integrity")
    print("="*80)
    
    # 加载原始测试数据
    test_df = pd.read_csv('../step5_timegan/output/test_with_attacks.csv', low_memory=False)
    attack_info = pd.read_csv('../step5_timegan/output/test_attack_info.csv')
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total points: {len(test_df):,}")
    print(f"  Flights: {test_df['flight'].nunique()}")
    print(f"  Attack flights: {len(attack_info)}")
    print(f"  Has 'label' column: {'label' in test_df.columns}")
    print(f"  Has 'attack_start_time' column: {'attack_start_time' in test_df.columns}")
    
    # 生成标签
    from labeling import generate_point_labels
    test_df = generate_point_labels(test_df, attack_info)
    
    print(f"\n🏷️  Label Distribution (after generation):")
    label_counts = test_df['label'].value_counts()
    for label, count in label_counts.items():
        print(f"  Label {label}: {count:,} ({count/len(test_df)*100:.2f}%)")
    
    # 检查前5个攻击飞行
    print(f"\n🔍 Sample Attack Flights:")
    for idx, row in attack_info.head(5).iterrows():
        fid = int(row['flight'])
        flight_data = test_df[test_df['flight'] == fid].reset_index(drop=True)
        attack_data = flight_data[flight_data['label'] == 1]
        
        print(f"\n  Flight {fid} ({row['attack_type']}):")
        print(f"    Total points: {len(flight_data)}")
        print(f"    Attack points: {len(attack_data)} ({len(attack_data)/len(flight_data)*100:.1f}%)")
        print(f"    Attack start time (metadata): {row['attack_start_time']:.3f}s")
        
        if len(attack_data) > 0:
            # 找到第一个标签为1的点
            first_attack_idx = attack_data.index[0]
            first_attack_time = first_attack_idx * config.TIME_PER_POINT
            print(f"    First labeled attack point: index {first_attack_idx}, time {first_attack_time:.3f}s")
            print(f"    ⚠️  Discrepancy: {abs(first_attack_time - row['attack_start_time']):.3f}s")


def diagnose_window_creation():
    """诊断窗口创建过程"""
    print("\n" + "="*80)
    print("DIAGNOSIS 2: Window Creation Process")
    print("="*80)
    
    # 加载数据
    test_df = pd.read_csv('../step5_timegan/output/test_with_attacks.csv', low_memory=False)
    attack_info = pd.read_csv('../step5_timegan/output/test_attack_info.csv')
    
    # 生成点级标签
    print("\n[1] Generating point-level labels...")
    test_df = generate_point_labels(test_df, attack_info)
    
    # 特征工程
    print("\n[2] Feature engineering...")
    from feature_engineering import compute_all_features
    test_df = compute_all_features(test_df)
    
    feature_cols = get_feature_columns()
    
    # 创建窗口
    print("\n[3] Creating windows...")
    X_test, y_test, flight_ids = create_windows_from_dataset(
        test_df, feature_cols,
        window_size=config.WINDOW_SIZE,
        step_size=config.STEP_SIZE
    )
    
    print(f"\n📊 Window Statistics:")
    print(f"  Total windows: {len(X_test):,}")
    print(f"  Positive windows: {y_test.sum():,} ({y_test.mean()*100:.2f}%)")
    
    # 检查第一个攻击飞行的窗口
    first_attack_flight = int(attack_info.iloc[0]['flight'])
    flight_windows_mask = (flight_ids == first_attack_flight)
    flight_windows = np.where(flight_windows_mask)[0]
    flight_labels = y_test[flight_windows_mask]
    
    print(f"\n🔍 First Attack Flight (ID={first_attack_flight}):")
    print(f"  Windows created: {len(flight_windows)}")
    print(f"  Attack windows: {flight_labels.sum()} ({flight_labels.mean()*100:.1f}%)")
    if flight_labels.sum() > 0:
        first_attack_window_local = np.where(flight_labels == 1)[0][0]
        print(f"  First attack window index (local): {first_attack_window_local}")
    
    # 分析窗口与点的对应关系
    flight_data = test_df[test_df['flight'] == first_attack_flight].reset_index(drop=True)
    attack_points = flight_data[flight_data['label'] == 1]
    if len(attack_points) > 0:
        first_attack_point_idx = attack_points.index[0]
        
        print(f"\n  Original data:")
        print(f"    Total points: {len(flight_data)}")
        print(f"    First attack point index: {first_attack_point_idx}")
        print(f"    First attack point time: {first_attack_point_idx * config.TIME_PER_POINT:.3f}s")
        
        # 计算第一个攻击窗口应该在哪里
        # 窗口i: 包含点[i*step:(i*step+window_size)]
        # 最后一个点的索引: i*step+window_size-1
        # 如果最后一个点 >= first_attack_point_idx，则窗口会被标记为attack
        expected_first_attack_window = max(0, (first_attack_point_idx - config.WINDOW_SIZE + 1) // config.STEP_SIZE)
        
        print(f"\n  Expected first attack window (if using LAST point labeling):")
        print(f"    Window index (local): ~{expected_first_attack_window}")
        print(f"    Window start point: {expected_first_attack_window * config.STEP_SIZE}")
        print(f"    Window end point: {expected_first_attack_window * config.STEP_SIZE + config.WINDOW_SIZE - 1}")
        print(f"    Window center point: {expected_first_attack_window * config.STEP_SIZE + config.WINDOW_SIZE // 2}")


def diagnose_timestamp_calculation():
    """诊断时间戳计算"""
    print("\n" + "="*80)
    print("DIAGNOSIS 3: Timestamp Calculation")
    print("="*80)
    
    # 加载数据
    test_df = pd.read_csv('../step5_timegan/output/test_with_attacks.csv')
    attack_info = pd.read_csv('../step5_timegan/output/test_attack_info.csv')
    
    # 准备数据
    from data_loader import prepare_test_windows
    X_test, y_test, timestamps, flight_ids, attack_types, attack_segments_info = \
        prepare_test_windows(test_df, attack_info)
    
    # 检查第一个攻击
    first_attack = attack_segments_info[0]
    fid = first_attack['flight_id']
    
    # 找到该飞行的窗口
    flight_mask = flight_ids == fid
    flight_windows = np.where(flight_mask)[0]
    flight_timestamps = timestamps[flight_mask]
    flight_labels = y_test[flight_mask]
    
    attack_windows = np.where(flight_labels == 1)[0]
    
    print(f"\n🔍 Flight {fid} ({first_attack['attack_type']}):")
    print(f"  Real attack start time: {first_attack['attack_start_time']:.3f}s")
    print(f"  Total windows: {len(flight_windows)}")
    print(f"  Attack windows: {len(attack_windows)}")
    
    if len(attack_windows) > 0:
        first_attack_window_idx = attack_windows[0]
        first_window_timestamp = flight_timestamps[first_attack_window_idx]
        
        print(f"\n  First attack window:")
        print(f"    Local index: {first_attack_window_idx}")
        print(f"    Timestamp: {first_window_timestamp:.3f}s")
        print(f"    ⚠️  Discrepancy: {abs(first_window_timestamp - first_attack['attack_start_time']):.3f}s")
        
        # 如果时间戳早于或等于攻击时间，检测延迟将是0！
        if first_window_timestamp <= first_attack['attack_start_time']:
            print(f"\n  ❌ PROBLEM FOUND: Window timestamp ({first_window_timestamp:.3f}s) <= "
                  f"attack time ({first_attack['attack_start_time']:.3f}s)")
            print(f"     This will result in 0 detection delay!")
            print(f"     Window center is BEFORE the attack actually starts!")


def diagnose_window_labeling():
    """诊断窗口标签策略"""
    print("\n" + "="*80)
    print("DIAGNOSIS 4: Window Labeling Strategy")
    print("="*80)
    
    print("\n📋 Current Labeling Strategy:")
    print("  From labeling.py -> get_window_label():")
    print("  - Uses the label of the LAST point in the window")
    print("  - If last point is attack (1), window is labeled as attack")
    
    print("\n⚠️  Potential Issue:")
    print("  If a window's LAST point is labeled as attack,")
    print("  but the window CENTER is BEFORE the actual attack start,")
    print("  this creates a temporal inconsistency!")
    
    print("\n  Example:")
    print("    Window: points [40-89], center at point 64")
    print("    Attack starts: at point 45")
    print("    Last point (89): attack label = 1")
    print("    → Window labeled as attack")
    print("    → Window center time < Attack start time")
    print("    → Detection delay = 0 (or negative, capped to 0)")
    
    print("\n  This explains why 90% of delays are 0!")


if __name__ == '__main__':
    diagnose_test_data()
    diagnose_window_creation()
    diagnose_timestamp_calculation()
    diagnose_window_labeling()
    
    print("\n" + "="*80)
    print("SUMMARY OF FINDINGS")
    print("="*80)
    print("""
The high metrics (90% zero delays) are caused by:

1. Window labeling uses the LAST point's label
2. Window timestamp uses the CENTER point
3. When attack starts in the middle of a window:
   - Last point may be attack → window labeled as attack
   - But center is BEFORE attack start
   - Result: timestamp <= attack_start_time → delay = 0

This is a TEMPORAL INCONSISTENCY in the evaluation methodology!

SOLUTIONS:
A) Use window's LAST point time (instead of center) for timestamp
B) Use FIRST attack point in window for labeling (not last)
C) Require majority of window to be attack (not just last point)
D) Use a lookahead-free labeling strategy
    """)
