"""
检查模型中间层激活
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

# 加载数据
test_df = pd.read_csv(output_dir / 'test_complete.csv', low_memory=False).head(5000)
attack_info = pd.read_csv(output_dir / 'test_attack_info.csv')
test_df = generate_point_labels(test_df, attack_info)
test_df = compute_all_features(test_df)

# 归一化
with open(output_dir / 'normalization_stats.json', 'r') as f:
    norm_stats = json.load(f)

feature_cols = get_feature_columns()
for col in feature_cols:
    mean = norm_stats['mean'][col]
    std = norm_stats['std'][col]
    test_df[col] = (test_df[col] - mean) / (std + 1e-8)

# 创建窗口
X_test, y_test, _ = create_windows_from_dataset(test_df, feature_cols)
print(f"窗口数: {len(X_test)}")

# 加载模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = create_model('lstm', n_features=len(feature_cols), window_size=50, dropout=0.3)
model.load_state_dict(torch.load(output_dir / 'best_model_lstm.pth', map_location=device))
model = model.to(device)
model.eval()

# Hook to capture intermediate outputs
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# 注册hooks
model.fc1.register_forward_hook(get_activation('fc1'))
model.fc2.register_forward_hook(get_activation('fc2'))
model.fc3.register_forward_hook(get_activation('fc3'))

# 前向传播
with torch.no_grad():
    X_sample = torch.tensor(X_test[:100], dtype=torch.float32).to(device)
    output = model(X_sample)
    
    print("\n中间层激活统计:")
    for name in ['fc1', 'fc2', 'fc3']:
        act = activations[name].cpu().numpy()
        print(f"\n{name}:")
        print(f"  Shape: {act.shape}")
        print(f"  Min: {act.min():.6f}")
        print(f"  Max: {act.max():.6f}")
        print(f"  Mean: {act.mean():.6f}")
        print(f"  Std: {act.std():.6f}")
        print(f"  前10个值: {act.flatten()[:10].round(6)}")
        
        # 检查是否所有值都相同
        if act.std() < 0.001:
            print(f"  ⚠️  WARNING: 激活值几乎不变！")
