"""
测试数据结构
"""

import os
import numpy as np
import pandas as pd

# 加载一个实验的数据
exp_dir = '../step3_multiModel/output/step/cnn'

# 加载预测结果
predictions_path = os.path.join(exp_dir, 'test_predictions.npz')
predictions_data = np.load(predictions_path)

# 加载攻击信息
attack_info_path = os.path.join(exp_dir, 'test_attack_info.csv')
attack_info = pd.read_csv(attack_info_path)

print("=== 数据结构检查 ===\n")

print(f"predictions_data keys: {list(predictions_data.keys())}")
print(f"predictions shape: {predictions_data['predictions'].shape}")
print(f"targets shape: {predictions_data['targets'].shape}")
print(f"flight_ids shape: {predictions_data['flight_ids'].shape}")
print(f"flight_ids unique: {len(np.unique(predictions_data['flight_ids']))}")
print(f"flight_ids sample: {predictions_data['flight_ids'][:10]}")

print(f"\nattack_info shape: {attack_info.shape}")
print(f"attack_info columns: {list(attack_info.columns)}")
print(f"\nattack_info.head():")
print(attack_info.head())

print(f"\nattack_info['flight'].unique(): {sorted(attack_info['flight'].unique())}")
print(f"Number of flights in attack_info: {len(attack_info)}")
print(f"Number of samples in predictions: {len(predictions_data['predictions'])}")
