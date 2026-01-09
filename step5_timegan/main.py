"""
Step5主流程：TimeGAN数据增强 + 低比例攻击注入 + 多模型训练
支持7种模型：LSTM, GRU, BiLSTM, CNN (1D-CNN), CNN-LSTM, TCN, Transformer
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

# 直接从step3b导入，避免循环导入
from labeling import generate_point_labels
from feature_engineering import compute_all_features, get_feature_columns, normalize_features
from window_creation import create_windows_from_dataset, WindowDataset
from model import create_model
from training import train_model
from evaluation import evaluate_model

# 支持的所有模型类型
ALL_MODEL_TYPES = ['lstm', 'gru', 'bilstm', 'cnn', 'cnn_lstm', 'tcn', 'transformer']

def run_step5_pipeline(model_type='lstm'):
    """
    完整的step5流程
    
    Args:
        model_type: 模型类型 ('lstm', 'gru', 'cnn_lstm', 'transformer')
    """
    print("=" * 80)
    print(" " * 20 + "Step5: TimeGAN增强训练流程")
    print("=" * 80)
    print(f"模型类型: {model_type}")
    
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # 1. 加载带攻击的数据
    # ========================================================================
    print("\n" + "=" * 80)
    print("1. 加载数据")
    print("=" * 80)
    
    train_df = pd.read_csv(output_dir / 'train_with_attacks.csv')
    val_df = pd.read_csv(output_dir / 'val_with_attacks.csv')
    test_df = pd.read_csv(output_dir / 'test_with_attacks.csv')
    
    train_attack_info = pd.read_csv(output_dir / 'train_attack_info.csv')
    val_attack_info = pd.read_csv(output_dir / 'val_attack_info.csv')
    test_attack_info = pd.read_csv(output_dir / 'test_attack_info.csv')
    
    print(f"\n训练集: {len(train_df):,} 记录, {train_df['flight'].nunique()} 飞行")
    print(f"验证集: {len(val_df):,} 记录, {val_df['flight'].nunique()} 飞行")
    print(f"测试集: {len(test_df):,} 记录, {test_df['flight'].nunique()} 飞行")
    
    # ========================================================================
    # 2. 生成标签
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. 生成点级标签")
    print("=" * 80)
    
    train_df = generate_point_labels(train_df, train_attack_info)
    val_df = generate_point_labels(val_df, val_attack_info)
    test_df = generate_point_labels(test_df, test_attack_info)
    
    print(f"\n训练集攻击点: {train_df['label'].sum():,} / {len(train_df):,} ({train_df['label'].mean()*100:.2f}%)")
    print(f"验证集攻击点: {val_df['label'].sum():,} / {len(val_df):,} ({val_df['label'].mean()*100:.2f}%)")
    print(f"测试集攻击点: {test_df['label'].sum():,} / {len(test_df):,} ({test_df['label'].mean()*100:.2f}%)")
    
    # ========================================================================
    # 3. 特征工程
    # ========================================================================
    print("\n" + "=" * 80)
    print("3. 特征工程")
    print("=" * 80)
    
    print("\n处理训练集...")
    train_df = compute_all_features(train_df)
    print("处理验证集...")
    val_df = compute_all_features(val_df)
    print("处理测试集...")
    test_df = compute_all_features(test_df)
    
    # 获取特征列并归一化
    feature_cols = get_feature_columns()
    print(f"\n特征列数: {len(feature_cols)}")
    
    train_df, val_df, test_df, normalization_stats = normalize_features(
        train_df, val_df, test_df, feature_cols
    )
    
    # 保存归一化统计
    import json
    with open(output_dir / 'normalization_stats.json', 'w') as f:
        json.dump(normalization_stats, f, indent=2)
    
    # ========================================================================
    # 4. 创建滑动窗口
    # ========================================================================
    print("\n" + "=" * 80)
    print("4. 创建滑动窗口")
    print("=" * 80)
    
    print(f"\n窗口大小: {config.WINDOW_SIZE}")
    print(f"步长: {config.STEP_SIZE}")
    
    print("\n训练集...")
    X_train, y_train, train_flight_ids = create_windows_from_dataset(train_df, feature_cols)
    
    print("验证集...")
    X_val, y_val, val_flight_ids = create_windows_from_dataset(val_df, feature_cols)
    
    print("测试集...")
    X_test, y_test, test_flight_ids = create_windows_from_dataset(test_df, feature_cols)
    
    print(f"\n窗口创建完成:")
    print(f"  训练集: {len(X_train):,} 窗口, 正样本: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
    print(f"  验证集: {len(X_val):,} 窗口, 正样本: {y_val.sum():,} ({y_val.mean()*100:.1f}%)")
    print(f"  测试集: {len(X_test):,} 窗口, 正样本: {y_test.sum():,} ({y_test.mean()*100:.1f}%)")
    
    # ========================================================================
    # 5. 保存窗口元数据
    # ========================================================================
    window_metadata = []
    
    for flight_id in train_df['flight'].unique():
        flight_info = train_attack_info[train_attack_info['flight'] == flight_id]
        if len(flight_info) > 0:
            window_metadata.append({
                'flight_id': flight_id,
                'attack_type': flight_info.iloc[0]['attack_type'],
                'dataset': 'train'
            })
    
    for flight_id in val_df['flight'].unique():
        flight_info = val_attack_info[val_attack_info['flight'] == flight_id]
        if len(flight_info) > 0:
            window_metadata.append({
                'flight_id': flight_id,
                'attack_type': flight_info.iloc[0]['attack_type'],
                'dataset': 'val'
            })
    
    for flight_id in test_df['flight'].unique():
        flight_info = test_attack_info[test_attack_info['flight'] == flight_id]
        if len(flight_info) > 0:
            window_metadata.append({
                'flight_id': flight_id,
                'attack_type': flight_info.iloc[0]['attack_type'],
                'dataset': 'test'
            })
    
    metadata_df = pd.DataFrame(window_metadata)
    metadata_df.to_csv(output_dir / 'window_metadata.csv', index=False)
    
    print(f"\n窗口统计:")
    print(f"  训练: {len(X_train)} 窗口, 攻击率: {y_train.mean()*100:.2f}%")
    print(f"  验证: {len(X_val)} 窗口, 攻击率: {y_val.mean()*100:.2f}%")
    print(f"  测试: {len(X_test)} 窗口, 攻击率: {y_test.mean()*100:.2f}%")
    
    # ========================================================================
    # 6. 训练模型
    # ========================================================================
    print("\n" + "=" * 80)
    print("5. 训练模型")
    print("=" * 80)
    
    # 创建DataLoader
    train_dataset = WindowDataset(X_train, y_train)
    val_dataset = WindowDataset(X_val, y_val)
    test_dataset = WindowDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建模型
    model = create_model(
        model_type=model_type,
        n_features=len(feature_cols),
        window_size=config.WINDOW_SIZE,
        dropout=0.3
    )
    
    print(f"\n模型类型: {model_type.upper()}")
    print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 计算正样本权重
    pos_weight = train_dataset.get_positive_weight()
    pos_weight = min(pos_weight, 5.0)  # 限制最大值避免数值不稳定
    print(f"正样本权重: {pos_weight:.2f}")
    
    # 训练
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
    print(f"\n✓ 模型已保存: {model_path}")
    
    # ========================================================================
    # 7. 评估模型
    # ========================================================================
    print("\n" + "=" * 80)
    print("6. 评估模型")
    print("=" * 80)
    
    metrics, predictions, targets = evaluate_model(model, test_loader, test_flight_ids, device=device)
    
    print("\n测试集性能:")
    print(f"  PR-AUC:    {metrics.get('pr_auc', 0):.4f}")
    print(f"  ROC-AUC:   {metrics.get('roc_auc', 0):.4f}")
    print(f"  Precision: {metrics.get('precision', 0):.4f}")
    print(f"  Recall:    {metrics.get('recall', 0):.4f}")
    print(f"  F1 Score:  {metrics.get('f1', 0):.4f}")
    if 'avg_fp_per_flight' in metrics:
        print(f"  Avg FP/flight: {metrics['avg_fp_per_flight']:.2f}")
    
    # 保存指标
    metrics_path = output_dir / f'test_metrics_{model_type}.json'
    with open(metrics_path, 'w') as f:
        # 转换numpy类型为Python原生类型
        serializable_metrics = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                               for k, v in metrics.items()}
        json.dump(serializable_metrics, f, indent=2)
    print(f"\n✓ 指标已保存: {metrics_path}")
    
    # 保存训练历史
    history_path = output_dir / f'training_history_{model_type}.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Step5 流程完成!")
    print("=" * 80)
    
    return model, metrics, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Step5: TimeGAN增强多模型训练')
    parser.add_argument('--model', type=str, default='lstm',
                      choices=ALL_MODEL_TYPES,
                      help=f'模型类型: {", ".join(ALL_MODEL_TYPES)}')
    parser.add_argument('--all', action='store_true',
                      help='训练所有7种模型')
    
    args = parser.parse_args()
    
    if args.all:
        print("\n" + "=" * 80)
        print("训练所有模型")
        print("=" * 80)
        
        all_results = {}
        for model_type in ALL_MODEL_TYPES:
            print(f"\n\n{'='*80}")
            print(f"开始训练: {model_type.upper()}")
            print(f"{'='*80}\n")
            
            try:
                model, metrics, history = run_step5_pipeline(model_type=model_type)
                all_results[model_type] = {
                    'metrics': metrics,
                    'best_epoch': len(history['train_loss'])
                }
                print(f"\n✓ {model_type.upper()} 训练完成")
            except Exception as e:
                print(f"\n✗ {model_type.upper()} 训练失败: {e}")
                all_results[model_type] = {'error': str(e)}
        
        # 保存所有结果对比
        output_dir = Path(config.OUTPUT_DIR)
        
        # 转换所有numpy类型为Python原生类型以支持JSON序列化
        def convert_to_serializable(obj):
            """递归转换numpy类型为Python原生类型"""
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        serializable_results = convert_to_serializable(all_results)
        
        with open(output_dir / 'all_models_comparison.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print("\n" + "=" * 80)
        print("所有模型训练完成！")
        print("=" * 80)
        
        # 打印对比
        print("\n模型性能对比:")
        print("-" * 80)
        print(f"{'模型':<15} {'PR-AUC':<10} {'F1':<10} {'Precision':<10} {'Recall':<10}")
        print("-" * 80)
        for model_type, result in all_results.items():
            if 'error' not in result:
                m = result['metrics']
                print(f"{model_type:<15} {m.get('pr_auc', 0):<10.4f} {m.get('f1', 0):<10.4f} {m.get('precision', 0):<10.4f} {m.get('recall', 0):<10.4f}")
            else:
                print(f"{model_type:<15} ERROR: {result['error']}")
    else:
        model, metrics, history = run_step5_pipeline(model_type=args.model)
        print(f"\n✓ {args.model.upper()} 训练完成")
    
    print("\n提示: 使用 --all 参数可以训练所有7种模型")
