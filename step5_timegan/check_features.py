"""
检查归一化后的特征分布
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
sys.path.append('../step3b_multiModelGeneral')

from labeling import generate_point_labels
from feature_engineering import compute_all_features, get_feature_columns

output_dir = Path('output')

# 加载数据
test_df = pd.read_csv(output_dir / 'test_complete.csv', low_memory=False).head(10000)
attack_info = pd.read_csv(output_dir / 'test_attack_info.csv')

test_df = generate_point_labels(test_df, attack_info)
test_df = compute_all_features(test_df)

feature_cols = get_feature_columns()

print("归一化前的特征统计:")
print(test_df[feature_cols].describe())

# 归一化
with open(output_dir / 'normalization_stats.json', 'r') as f:
    norm_stats = json.load(f)

for col in feature_cols:
    if col in norm_stats:
        mean = norm_stats[col]['mean']
        std = norm_stats[col]['std']
        test_df[col] = (test_df[col] - mean) / (std + 1e-8)

print("\n归一化后的特征统计:")
print(test_df[feature_cols].describe())

print("\n检查是否存在常数列:")
for col in feature_cols:
    unique_vals = test_df[col].nunique()
    if unique_vals < 10:
        print(f"  {col}: {unique_vals} unique values (可能是常数列)")
        print(f"    值分布: {test_df[col].value_counts().head()}")
