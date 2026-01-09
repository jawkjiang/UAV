"""
确定正确的TIME_PER_POINT值
"""
import pandas as pd
import numpy as np

df = pd.read_csv('../step5_timegan/output/test_complete.csv', low_memory=False)

print("="*80)
print("分析delta_t以确定正确的TIME_PER_POINT")
print("="*80)

# 每个flight的delta_t统计
stats = df.groupby('flight')['delta_t'].agg(['mean', 'median', 'std', 'count'])

print("\n每个flight的delta_t统计:")
print(stats.describe())

print(f"\n推荐的TIME_PER_POINT值:")
print(f"  所有点的平均delta_t: {df['delta_t'].mean():.6f}秒")
print(f"  所有点的中位数delta_t: {df['delta_t'].median():.6f}秒")
print(f"  每个flight平均值的均值: {stats['mean'].mean():.6f}秒")
print(f"  每个flight中位数的中位数: {stats['median'].median():.6f}秒")

# 检查是否有异常值
print(f"\nDelta_t范围检查:")
print(f"  最小值: {df['delta_t'].min():.6f}秒")
print(f"  第1百分位: {df['delta_t'].quantile(0.01):.6f}秒")
print(f"  第99百分位: {df['delta_t'].quantile(0.99):.6f}秒")
print(f"  最大值: {df['delta_t'].max():.6f}秒")

# 检查不同分组
print(f"\n不同label的delta_t:")
for label in df['label'].unique():
    subset = df[df['label'] == label]
    print(f"  Label {label}: mean={subset['delta_t'].mean():.6f}s, median={subset['delta_t'].median():.6f}s, count={len(subset)}")

print(f"\n建议:")
median_value = df['delta_t'].median()
print(f"  使用中位数作为TIME_PER_POINT: {median_value:.6f}秒")
print(f"  或使用0.167秒（用户提供的值）")
