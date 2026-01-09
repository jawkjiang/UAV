"""
快速验证修复后的模型
检查预测分布是否正常
"""
import sys
sys.path.append('../step5_timegan')
sys.path.append('../step3b_multiModelGeneral')

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import json

from labeling import generate_point_labels
from feature_engineering import compute_all_features, get_feature_columns
from window_creation import create_windows_from_dataset, WindowDataset
from model import create_model

def quick_test():
    """快速测试修复后的模型"""
    
    output_dir = Path('../step5_timegan/output')
    
    # 加载数据
    print("加载测试数据...")
    test_df = pd.read_csv(output_dir / 'test_complete.csv', low_memory=False)
    attack_info = pd.read_csv(output_dir / 'test_attack_info.csv')
    
    print(f"  测试集: {len(test_df):,} points, {test_df['flight'].nunique()} flights")
    print(f"  攻击信息: {len(attack_info)} attacks")
    
    # 生成标签
    test_df = generate_point_labels(test_df, attack_info)
    print(f"  标签比例: {test_df['label'].mean()*100:.2f}%")
    
    # 特征工程
    test_df = compute_all_features(test_df)
    
    # 加载归一化
    with open(output_dir / 'normalization_stats.json', 'r') as f:
        norm_stats = json.load(f)
    
    feature_cols = get_feature_columns()
    for col in feature_cols:
        if col in norm_stats['mean']:
            mean = norm_stats['mean'][col]
            std = norm_stats['std'][col]
            test_df[col] = (test_df[col] - mean) / (std + 1e-8)
    
    # 创建窗口
    X_test, y_test, _ = create_windows_from_dataset(test_df, feature_cols)
    print(f"  窗口数: {len(X_test)}, 正样本: {y_test.sum()} ({y_test.mean()*100:.1f}%)")
    
    # 加载模型
    model_path = output_dir / 'best_model_lstm.pth'
    if not model_path.exists():
        print(f"\n模型文件不存在: {model_path}")
        print("请先运行: python main_simple.py --model lstm")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model('lstm', n_features=len(feature_cols), window_size=50, dropout=0.3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 预测
    print("\n预测...")
    test_dataset = WindowDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
            # 模型已经返回sigmoid概率，直接使用
            probs = output.cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)
            
            all_probs.append(probs)
            all_preds.append(preds)
    
    y_prob = np.concatenate(all_probs)
    y_pred = np.concatenate(all_preds)
    
    # 分析结果
    print("\n" + "="*60)
    print("预测分析")
    print("="*60)
    
    print(f"\n预测类别分布:")
    unique, counts = np.unique(y_pred, return_counts=True)
    for val, count in zip(unique, counts):
        label = "攻击" if val == 1 else "正常"
        print(f"  {label}: {count:,} ({count/len(y_pred)*100:.1f}%)")
    
    print(f"\n概率统计:")
    print(f"  最小值: {y_prob.min():.4f}")
    print(f"  最大值: {y_prob.max():.4f}")
    print(f"  平均值: {y_prob.mean():.4f}")
    print(f"  中位数: {np.median(y_prob):.4f}")
    
    print(f"\n真实标签分布:")
    print(f"  正常: {(y_test==0).sum():,} ({(y_test==0).mean()*100:.1f}%)")
    print(f"  攻击: {(y_test==1).sum():,} ({(y_test==1).mean()*100:.1f}%)")
    
    print(f"\n性能:")
    accuracy = (y_pred == y_test).mean()
    if y_test.sum() > 0:
        recall = ((y_pred == 1) & (y_test == 1)).sum() / y_test.sum()
    else:
        recall = 0
    if (y_test == 0).sum() > 0:
        fpr = ((y_pred == 1) & (y_test == 0)).sum() / (y_test == 0).sum()
    else:
        fpr = 0
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Recall:   {recall:.4f}")
    print(f"  FPR:      {fpr:.4f}")
    
    if y_pred.max() == 0:
        print("\n⚠️  警告: 所有预测都是正常（没有攻击预测）")
    elif y_pred.min() == 1:
        print("\n❌ 错误: 所有预测都是攻击（问题未解决）")
    else:
        print("\n✅ 正常: 模型输出多样化")


if __name__ == "__main__":
    quick_test()
