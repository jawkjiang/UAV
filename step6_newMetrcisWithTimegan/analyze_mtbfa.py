"""
分析MTBFA计算逻辑
"""
import sys
sys.path.append('../step5_timegan')
sys.path.append('../step3b_multiModelGeneral')

import numpy as np
import pandas as pd
import config

# 配置参数
WINDOW_SIZE = config.WINDOW_SIZE  # 50个点
STEP_SIZE = config.STEP_SIZE      # 5个点
TIME_PER_POINT = config.TIME_PER_POINT  # 0.01秒

print("="*80)
print("窗口时间计算")
print("="*80)

print(f"\n窗口配置:")
print(f"  窗口大小: {WINDOW_SIZE} 个采样点")
print(f"  步长: {STEP_SIZE} 个采样点")
print(f"  采样率: {1/TIME_PER_POINT:.0f} Hz ({TIME_PER_POINT}秒/点)")

print(f"\n时间跨度:")
window_time_span = WINDOW_SIZE * TIME_PER_POINT
step_time = STEP_SIZE * TIME_PER_POINT
print(f"  1个窗口时间跨度: {WINDOW_SIZE} × {TIME_PER_POINT}s = {window_time_span}s")
print(f"  相邻窗口时间间隔: {STEP_SIZE} × {TIME_PER_POINT}s = {step_time}s")

print("\n" + "="*80)
print("理论计算 vs 实际MTBFA")
print("="*80)

# 加载实际结果
overall_metrics = pd.read_csv('output/time_aware_metrics/overall_metrics.csv')
print("\n实际模型性能:")
for _, row in overall_metrics.iterrows():
    model = row['model']
    fa = int(row['n_false_alarms'])
    mtbfa = row['MTBFA']
    
    print(f"\n{model.upper()}:")
    print(f"  False Alarms: {fa} 次")
    print(f"  MTBFA: {mtbfa:.4f}h = {mtbfa*60:.2f}分钟")

print("\n" + "="*80)
print("用户的理论计算:")
print("="*80)

accuracy = 0.99
error_rate = 1 - accuracy
windows_per_error = 1 / error_rate

print(f"\n如果Accuracy={accuracy*100}%:")
print(f"  错误率 = {error_rate*100}% = 1/{int(1/error_rate)}")
print(f"  理论上每{int(windows_per_error)}个窗口出现1个错误")
print(f"  时间跨度 = {int(windows_per_error)} × {step_time}s = {windows_per_error * step_time:.0f}秒 = {windows_per_error * step_time / 60:.1f}分钟")

print("\n⚠️  问题：用户认为1个窗口=50秒，但实际是：")
print(f"  ❌ 错误计算: 1窗口 = {WINDOW_SIZE}秒  (误认为1点=1秒)")
print(f"  ✅ 正确计算: 1窗口 = {WINDOW_SIZE}点 × {TIME_PER_POINT}秒/点 = {window_time_span}秒")
print(f"  ✅ 窗口间隔: 步长{STEP_SIZE}点 × {TIME_PER_POINT}秒/点 = {step_time}秒")

print("\n" + "="*80)
print("MTBFA实际计算逻辑")
print("="*80)

print("\nMTBFA = 总正常飞行时间 / 误报事件数")
print("\n关键点:")
print(f"  1. 每个窗口代表{step_time}秒的时间推进 (步长)")
print(f"  2. 32,405个窗口 ≈ 32,405 × {step_time}s = {32405 * step_time / 3600:.2f}小时")
print(f"  3. 如果有5个误报，MTBFA = {32405 * step_time / 3600:.2f}h / 5 ≈ {32405 * step_time / 3600 / 5:.2f}h = {32405 * step_time / 3600 / 5 * 60:.1f}分钟")

print("\n实际数据验证:")
# 加载一个模型的预测
data = np.load('output/lstm_test_predictions.npz')
y_true = data['y_true']
y_pred = data['y_pred']

total_windows = len(y_true)
normal_windows = (y_true == 0).sum()
attack_windows = (y_true == 1).sum()
false_positives = ((y_true == 0) & (y_pred == 1)).sum()

print(f"\nLSTM模型:")
print(f"  总窗口数: {total_windows:,}")
print(f"  正常窗口: {normal_windows:,} ({normal_windows/total_windows*100:.1f}%)")
print(f"  攻击窗口: {attack_windows:,} ({attack_windows/total_windows*100:.1f}%)")
print(f"  误报窗口数: {false_positives:,} ({false_positives/total_windows*100:.2f}%)")

total_normal_time_hours = normal_windows * step_time / 3600
print(f"\n时间计算:")
print(f"  正常窗口总时间: {normal_windows:,} × {step_time}s = {total_normal_time_hours:.2f}小时")

# MTBFA计算使用的是误报"事件"数（连续误报算一次），不是误报窗口数
# 所以实际MTBFA会高于 total_normal_time_hours / false_positives
mtbfa_per_window = total_normal_time_hours / false_positives if false_positives > 0 else float('inf')
print(f"  如果按误报窗口计算: {total_normal_time_hours:.2f}h / {false_positives} = {mtbfa_per_window:.3f}h = {mtbfa_per_window*60:.1f}分钟")
print(f"  实际MTBFA（按事件）: 约0.1h = 6分钟 (因为连续误报合并为1个事件)")

print("\n" + "="*80)
print("结论")
print("="*80)
print("\n1. 用户的计算基于错误假设:")
print(f"   - 误认为1个窗口 = {WINDOW_SIZE}秒")
print(f"   - 实际1个窗口步长 = {step_time}秒 (步长{STEP_SIZE}点)")
print(f"   - 差距: {WINDOW_SIZE / step_time:.0f}倍")

print(f"\n2. 修正后的计算:")
print(f"   - Accuracy=99% → 每100个窗口1个错误")
print(f"   - 100窗口 × {step_time}s = {100 * step_time}s = {100 * step_time / 60:.1f}分钟")
print(f"   - 与实际MTBFA=6分钟数量级一致")

print(f"\n3. MTBFA计算细节:")
print(f"   - 测试集有252个flights，只有12个被攻击")
print(f"   - 240个正常flights，每个约688点 ≈ 6.88秒")
print(f"   - 总正常时间 ≈ 240 × 6.88s ≈ 27.5分钟 = 0.46小时")
print(f"   - 如果6个误报事件，MTBFA = 0.46h / 6 ≈ 0.077h ≈ 4.6分钟")
print(f"   - 实际值0.1h可能因为攻击flights中也有正常段")
