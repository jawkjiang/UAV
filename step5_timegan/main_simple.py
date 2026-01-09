"""
Step5简化版主流程：TimeGAN数据增强 + 低比例攻击注入 + 模型训练
直接使用step3b的API
"""
import os
import sys
sys.path.append('../step3b_multiModelGeneral')

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
import config

# 从step3b导入
from labeling import generate_point_labels
from feature_engineering import compute_all_features, get_feature_columns, normalize_features
from window_creation import create_windows_from_dataset, WindowDataset
from model import create_model
from training import train_model
from evaluation import evaluate_model


def run_step5_training(model_type='lstm'):
    """
    Step5训练流程
    """
    print("=" * 80)
    print(" " * 20 + f"Step5: {model_type.upper()} 训练")
    print("=" * 80)
    
    output_dir = Path(config.OUTPUT_DIR)
    
    # 1. 加载数据
    print("\n[1/6] 加载带攻击的数据...")
    train_df = pd.read_csv(output_dir / 'train_complete.csv', low_memory=False)
    val_df = pd.read_csv(output_dir / 'val_complete.csv', low_memory=False)
    test_df = pd.read_csv(output_dir / 'test_complete.csv', low_memory=False)
    
    train_attack_info = pd.read_csv(output_dir / 'train_attack_info.csv')
    val_attack_info = pd.read_csv(output_dir / 'val_attack_info.csv')
    test_attack_info = pd.read_csv(output_dir / 'test_attack_info.csv')
    
    print(f"  训练集: {len(train_df):,} 记录, {train_df['flight'].nunique()} 飞行")
    print(f"  验证集: {len(val_df):,} 记录, {val_df['flight'].nunique()} 飞行")
    print(f"  测试集: {len(test_df):,} 记录, {test_df['flight'].nunique()} 飞行")
    
    # 2. 生成标签
    print("\n[2/6] 生成点级标签...")
    train_df = generate_point_labels(train_df, train_attack_info)
    val_df = generate_point_labels(val_df, val_attack_info)
    test_df = generate_point_labels(test_df, test_attack_info)
    
    print(f"  训练集: {train_df['label'].sum():,} / {len(train_df):,} ({train_df['label'].mean()*100:.2f}%)")
    print(f"  验证集: {val_df['label'].sum():,} / {len(val_df):,} ({val_df['label'].mean()*100:.2f}%)")
    print(f"  测试集: {test_df['label'].sum():,} / {len(test_df):,} ({test_df['label'].mean()*100:.2f}%)")
    
    # 3. 特征工程
    print("\n[3/6] 特征工程...")
    train_df = compute_all_features(train_df)
    val_df = compute_all_features(val_df)
    test_df = compute_all_features(test_df)
    
    feature_cols = get_feature_columns()
    train_df, val_df, test_df, norm_stats = normalize_features(
        train_df, val_df, test_df, feature_cols
    )
    
    with open(output_dir / 'normalization_stats.json', 'w') as f:
        json.dump(norm_stats, f, indent=2)
    
    print(f"  特征数: {len(feature_cols)}")
    
    # 4. 创建窗口
    print("\n[4/6] 创建滑动窗口...")
    print(f"  窗口大小: {config.WINDOW_SIZE}, 步长: {config.STEP_SIZE}")
    
    X_train, y_train, _ = create_windows_from_dataset(train_df, feature_cols)
    X_val, y_val, _ = create_windows_from_dataset(val_df, feature_cols)
    X_test, y_test, test_flight_ids = create_windows_from_dataset(test_df, feature_cols)
    
    print(f"  训练: {len(X_train):,} 窗口, 正样本 {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
    print(f"  验证: {len(X_val):,} 窗口, 正样本 {y_val.sum():,} ({y_val.mean()*100:.1f}%)")
    print(f"  测试: {len(X_test):,} 窗口, 正样本 {y_test.sum():,} ({y_test.mean()*100:.1f}%)")
    
    # 5. 创建DataLoader
    print("\n[5/6] 训练模型...")
    train_dataset = WindowDataset(X_train, y_train)
    val_dataset = WindowDataset(X_val, y_val)
    test_dataset = WindowDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  设备: {device}")
    
    model = create_model(
        model_type=model_type,
        n_features=len(feature_cols),
        window_size=config.WINDOW_SIZE,
        dropout=0.3
    )
    
    pos_weight = train_dataset.get_positive_weight()
    # 限制正样本权重，避免数值不稳定
    pos_weight = min(pos_weight, 5.0)
    print(f"  正样本权重: {pos_weight:.2f} (原始: {train_dataset.get_positive_weight():.2f})")
    print(f"  学习率: {config.LEARNING_RATE}")
    print(f"  开始训练...")
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=config.MAX_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        patience=config.EARLY_STOP_PATIENCE,
        pos_weight=pos_weight
    )
    
    # 保存模型
    model_path = output_dir / f'best_model_{model_type}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\n  ✓ 模型已保存: {model_path}")
    
    # 6. 评估
    print("\n[6/6] 评估模型...")
    metrics = evaluate_model(model, test_loader, test_flight_ids, device=device)
    
    # 保存结果
    with open(output_dir / f'{model_type}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "=" * 80)
    print("测试集性能")
    print("=" * 80)
    print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
    print(f"  Precision: {metrics.get('precision', 0):.4f}")
    print(f"  Recall:    {metrics.get('recall', 0):.4f}")
    print(f"  F1 Score:  {metrics.get('f1', 0):.4f}")
    if 'fpr' in metrics:
        print(f"  FPR:       {metrics['fpr']:.4f}")
    if 'auc' in metrics:
        print(f"  AUC:       {metrics['auc']:.4f}")
    
    return model, metrics, history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lstm',
                       choices=['lstm', 'gru', 'cnn_lstm', 'transformer'])
    args = parser.parse_args()
    
    run_step5_training(model_type=args.model)
