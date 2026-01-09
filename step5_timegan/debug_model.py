"""
调试模型输出
"""
import sys
sys.path.append('../step3b_multiModelGeneral')

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json

from labeling import generate_point_labels
from feature_engineering import compute_all_features, get_feature_columns
from window_creation import create_windows_from_dataset
from model import create_model

output_dir = Path('output')

# 加载少量数据
test_df = pd.read_csv(output_dir / 'test_complete.csv', low_memory=False)
attack_info = pd.read_csv(output_dir / 'test_attack_info.csv')

# 只取前1000个点
test_df = test_df.head(1000)
test_df = generate_point_labels(test_df, attack_info)
test_df = compute_all_features(test_df)

# 归一化
with open(output_dir / 'normalization_stats.json', 'r') as f:
    norm_stats = json.load(f)

feature_cols = get_feature_columns()
for col in feature_cols:
    if col in norm_stats:
        mean = norm_stats[col]['mean']
        std = norm_stats[col]['std']
        test_df[col] = (test_df[col] - mean) / (std + 1e-8)

# 创建窗口
X_test, y_test, _ = create_windows_from_dataset(test_df, feature_cols)
print(f"窗口数: {len(X_test)}, 正样本: {y_test.sum()}/{len(y_test)}")

# 加载模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = create_model('lstm', n_features=len(feature_cols), window_size=50, dropout=0.3)
model.load_state_dict(torch.load(output_dir / 'best_model_lstm.pth', map_location=device))
model = model.to(device)
model.eval()

# 预测前10个窗口
with torch.no_grad():
    X_sample = torch.tensor(X_test[:10], dtype=torch.float32).to(device)
    logits = model(X_sample).cpu().numpy().flatten()
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    
    print("\n前10个窗口:")
    print("真实标签:", y_test[:10])
    print("原始logits:", logits)
    print("概率值:", probs)
    print("预测 (>=0.5):", (probs >= 0.5).astype(int))
    
    # 检查模型权重
    print("\n模型最后一层权重统计:")
    for name, param in model.named_parameters():
        if 'fc' in name:  # 最后全连接层
            print(f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}, min={param.data.min():.4f}, max={param.data.max():.4f}")
