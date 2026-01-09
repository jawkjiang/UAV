"""
全面诊断：找出为什么所有模型都预测攻击
"""
import sys
sys.path.append('../step3b_multiModelGeneral')

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json

from labeling import generate_point_labels
from feature_engineering import compute_all_features, get_feature_columns, normalize_features
from window_creation import create_windows_from_dataset
from model import create_model

output_dir = Path('output')

print("="*80)
print("诊断1: 检查归一化参数")
print("="*80)

with open(output_dir / 'normalization_stats.json', 'r') as f:
    norm_stats = json.load(f)

print("\n归一化统计 (前5个特征):")
for col in list(norm_stats['mean'].keys())[:5]:
    print(f"  {col:25s}  mean={norm_stats['mean'][col]:10.4f}, std={norm_stats['std'][col]:10.4f}")

print("\n" + "="*80)
print("诊断2: 加载并归一化测试数据")
print("="*80)

test_df = pd.read_csv(output_dir / 'test_complete.csv', low_memory=False).head(5000)
attack_info = pd.read_csv(output_dir / 'test_attack_info.csv')

test_df = generate_point_labels(test_df, attack_info)
print(f"标签分布: {test_df['label'].value_counts().to_dict()}")

test_df = compute_all_features(test_df)
feature_cols = get_feature_columns()

print(f"\n归一化前 (前3个特征):")
for col in feature_cols[:3]:
    print(f"  {col:25s}  mean={test_df[col].mean():10.4f}, std={test_df[col].std():10.4f}")

# 手动归一化
for col in feature_cols:
    mean = norm_stats['mean'][col]
    std = norm_stats['std'][col]
    test_df[col] = (test_df[col] - mean) / (std + 1e-8)

print(f"\n归一化后 (前3个特征):")
for col in feature_cols[:3]:
    print(f"  {col:25s}  mean={test_df[col].mean():10.4f}, std={test_df[col].std():10.4f}")

print("\n" + "="*80)
print("诊断3: 创建窗口并检查特征值范围")
print("="*80)

X_test, y_test, _ = create_windows_from_dataset(test_df, feature_cols)
print(f"窗口数: {len(X_test)}, 正样本: {y_test.sum()}/{len(y_test)} ({y_test.mean()*100:.1f}%)")

print(f"\n特征统计:")
print(f"  Shape: {X_test.shape}")  # (N, window_size, n_features)
print(f"  Min: {X_test.min():.4f}")
print(f"  Max: {X_test.max():.4f}")
print(f"  Mean: {X_test.mean():.4f}")
print(f"  Std: {X_test.std():.4f}")

print("\n" + "="*80)
print("诊断4: 加载模型并预测")
print("="*80)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = create_model('lstm', n_features=len(feature_cols), window_size=50, dropout=0.3)
model.load_state_dict(torch.load(output_dir / 'best_model_lstm.pth', map_location=device))
model = model.to(device)
model.eval()

# 预测前100个窗口
with torch.no_grad():
    X_sample = torch.tensor(X_test[:100], dtype=torch.float32).to(device)
    y_sample = y_test[:100]
    
    logits = model(X_sample).cpu().numpy().flatten()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    
    print(f"\n预测统计 (前100个窗口):")
    print(f"  真实标签: 正常={( y_sample==0).sum()}, 攻击={(y_sample==1).sum()}")
    print(f"  预测标签: 正常={(preds==0).sum()}, 攻击={(preds==1).sum()}")
    print(f"  Logits: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}, std={logits.std():.4f}")
    print(f"  Probs:  min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}, std={probs.std():.4f}")
    
    print(f"\n前10个样本:")
    print(f"  真实: {y_sample[:10]}")
    print(f"  预测: {preds[:10]}")
    print(f"  概率: {probs[:10].round(4)}")

print("\n" + "="*80)
print("诊断5: 检查模型权重")
print("="*80)

print("\n模型层统计:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  {name:30s}  shape={str(list(param.shape)):20s}  mean={param.data.mean():.6f}  std={param.data.std():.6f}")

print("\n" + "="*80)
print("诊断完成")
print("="*80)
