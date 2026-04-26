"""
Data Loader for Step6
加载Step5的测试数据和模型，生成窗口级预测
"""
import sys
sys.path.append('../step5_timegan')
sys.path.append('../step3b_multiModelGeneral')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import json
from typing import Dict, List, Tuple
import os

import config

# Import from step3b
from labeling import generate_point_labels
from feature_engineering import compute_all_features, get_feature_columns
from window_creation import create_windows_from_dataset, WindowDataset
from model import create_model


def load_test_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载Step5的测试数据
    
    Returns:
        test_df: 测试数据DataFrame
        attack_info: 攻击信息DataFrame
    """
    test_path = config.get_step5_test_data_path()
    attack_info_path = config.get_step5_test_attack_info_path()
    
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found: {test_path}")
    if not os.path.exists(attack_info_path):
        raise FileNotFoundError(f"Attack info not found: {attack_info_path}")
    
    test_df = pd.read_csv(test_path, low_memory=False)
    attack_info = pd.read_csv(attack_info_path)
    
    print(f"  Loaded test data: {len(test_df):,} points, {test_df['flight'].nunique()} flights")
    print(f"  Attack info: {len(attack_info)} attacks")
    
    return test_df, attack_info


def prepare_test_windows(test_df: pd.DataFrame, attack_info: pd.DataFrame) -> Tuple:
    """
    准备测试窗口数据
    
    Returns:
        X_test: 特征窗口 (N, window_size, n_features)
        y_test: 标签 (N,)
        timestamps: 时间戳 (N,)
        flight_ids: 飞行ID (N,)
        attack_types: 攻击类型 (N,)
        attack_segments_info: 攻击段信息列表
    """
    print("\n[1/3] 生成点级标签...")
    test_df = generate_point_labels(test_df, attack_info)
    
    attack_ratio = test_df['label'].mean()
    print(f"  Attack ratio: {attack_ratio*100:.2f}%")
    
    print("\n[2/3] 特征工程...")
    test_df = compute_all_features(test_df)
    
    # 加载归一化参数
    norm_path = config.get_step5_normalization_path()
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Normalization stats not found: {norm_path}")
    
    with open(norm_path, 'r') as f:
        norm_stats = json.load(f)
    
    feature_cols = get_feature_columns()
    
    # 应用归一化
    for col in feature_cols:
        if col in norm_stats['mean']:
            mean = norm_stats['mean'][col]
            std = norm_stats['std'][col]
            test_df[col] = (test_df[col] - mean) / (std + 1e-8)
    
    print(f"  Features: {len(feature_cols)}")
    
    print("\n[3/3] 创建滑动窗口...")
    X_test, y_test, flight_ids = create_windows_from_dataset(
        test_df, feature_cols,
        window_size=config.WINDOW_SIZE,
        step_size=config.STEP_SIZE
    )
    
    # 重建时间戳和攻击类型
    timestamps, attack_types, attack_segments_info = reconstruct_metadata(
        test_df, flight_ids, attack_info
    )
    
    print(f"  Windows: {len(X_test):,}")
    print(f"  Positive ratio: {y_test.mean()*100:.1f}%")
    
    return X_test, y_test, timestamps, flight_ids, attack_types, attack_segments_info


def reconstruct_metadata(test_df: pd.DataFrame, 
                         flight_ids: np.ndarray,
                         attack_info: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    重建窗口级元数据
    
    Args:
        test_df: 测试数据DataFrame
        flight_ids: 窗口对应的飞行ID
        attack_info: 攻击信息
    
    Returns:
        timestamps: 窗口中心时间戳 (秒)
        attack_types: 窗口对应的攻击类型
        attack_segments_info: 攻击段信息列表
    """
    n_windows = len(flight_ids)
    timestamps = np.zeros(n_windows)
    attack_types = np.array(['normal'] * n_windows, dtype=object)
    
    # 为每个飞行构建索引映射
    flight_to_windows = {}
    for idx, fid in enumerate(flight_ids):
        if fid not in flight_to_windows:
            flight_to_windows[fid] = []
        flight_to_windows[fid].append(idx)
    
    # 构建攻击信息字典
    attack_dict = {}
    for _, row in attack_info.iterrows():
        fid = int(row['flight'])
        attack_dict[fid] = {
            'attack_type': row['attack_type'],
            'attack_start_time': float(row['attack_start_time'])
        }
    
    # 为每个飞行处理窗口
    for fid in np.unique(flight_ids):
        # 获取该飞行的数据
        flight_mask = test_df['flight'] == fid
        flight_data = test_df[flight_mask].reset_index(drop=True)
        
        if len(flight_data) == 0:
            continue
        
        # 计算该飞行每个采样点的时间
        flight_timestamps = np.arange(len(flight_data)) * config.TIME_PER_POINT
        
        # 获取窗口索引
        window_indices = flight_to_windows.get(fid, [])
        
        # 计算每个窗口在飞行数据中的起始位置
        for i, win_idx in enumerate(window_indices):
            # 窗口起始位置 = i * step_size
            start_pos = i * config.STEP_SIZE
            # 使用窗口最后一个点的时间戳（修复时间不一致性问题）
            # 原因：窗口标签使用最后点，时间戳也应使用最后点以保持一致性
            last_pos = start_pos + config.WINDOW_SIZE - 1
            
            # 窗口最后点时间戳
            if last_pos < len(flight_timestamps):
                timestamps[win_idx] = flight_timestamps[last_pos]
            
            # 攻击类型
            if fid in attack_dict:
                attack_types[win_idx] = attack_dict[fid]['attack_type']
    
    # 构建attack_segments_info
    attack_segments_info = []
    for fid, info in attack_dict.items():
        attack_segments_info.append({
            'flight_id': int(fid),
            'attack_type': info['attack_type'],
            'attack_start_time': info['attack_start_time']
        })
    
    return timestamps, attack_types, attack_segments_info


def load_model_and_predict(model_name: str, 
                           X_test: np.ndarray,
                           y_test: np.ndarray) -> Dict[str, np.ndarray]:
    """
    加载模型并进行预测
    
    Args:
        model_name: 模型名称
        X_test: 测试特征
        y_test: 测试标签
    
    Returns:
        Dictionary with y_true, y_pred, y_prob
    """
    print(f"\n{'='*60}")
    print(f"  Loading and testing: {model_name.upper()}")
    print(f"{'='*60}")
    
    # 检查模型文件
    model_path = config.get_step5_model_path(model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"  Model path: {model_path}")
    
    # 创建模型
    n_features = X_test.shape[2]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = create_model(
        model_type=model_name,
        n_features=n_features,
        window_size=config.WINDOW_SIZE,
        dropout=0.3
    )
    
    # 加载权重
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"  Device: {device}")
    print(f"  Testing on {len(X_test):,} windows...")
    
    # 创建DataLoader
    test_dataset = WindowDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # 预测
    all_probs = []
    all_preds = []
    
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
    
    # 计算简单指标
    accuracy = (y_pred == y_test).mean()
    if y_test.sum() > 0:
        recall = ((y_pred == 1) & (y_test == 1)).sum() / y_test.sum()
    else:
        recall = 0
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Recall: {recall:.4f}")
    
    return {
        'y_true': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob
    }


def load_all_models_predictions(X_test: np.ndarray,
                               y_test: np.ndarray,
                               timestamps: np.ndarray,
                               flight_ids: np.ndarray,
                               attack_types: np.ndarray,
                               attack_segments_info: List,
                               use_cache: bool = True) -> Dict[str, Dict]:
    """
    加载所有模型并生成预测
    
    Args:
        X_test, y_test: 测试数据
        timestamps, flight_ids, attack_types: 元数据
        attack_segments_info: 攻击段信息
        use_cache: 是否使用缓存的预测结果
    
    Returns:
        Dictionary mapping model_name -> evaluation data
    """
    all_data = {}
    
    for model_name in config.MODEL_NAMES:
        pred_path = config.get_predictions_path(model_name)
        
        # 检查缓存
        if use_cache and os.path.exists(pred_path):
            print(f"\n[CACHE] Loading predictions for {model_name}...")
            data = np.load(pred_path)
            predictions = {
                'y_true': data['y_true'],
                'y_pred': data['y_pred'],
                'y_prob': data['y_prob']
            }
        else:
            # 生成新预测
            try:
                predictions = load_model_and_predict(model_name, X_test, y_test)
                
                # 保存缓存
                np.savez(
                    pred_path,
                    y_true=predictions['y_true'],
                    y_pred=predictions['y_pred'],
                    y_prob=predictions['y_prob']
                )
                print(f"  Saved predictions to: {pred_path}")
                
            except Exception as e:
                print(f"  ERROR: Failed to process {model_name}: {e}")
                continue
        
        # 组合所有数据
        all_data[model_name] = {
            'y_true': predictions['y_true'],
            'y_pred': predictions['y_pred'],
            'y_prob': predictions['y_prob'],
            'timestamps': timestamps,
            'flight_ids': flight_ids,
            'attack_types': attack_types,
            'attack_segments_info': attack_segments_info
        }
    
    return all_data


if __name__ == '__main__':
    # 测试数据加载
    print("Testing data loader...")
    
    test_df, attack_info = load_test_data()
    X_test, y_test, timestamps, flight_ids, attack_types, attack_segments_info = \
        prepare_test_windows(test_df, attack_info)
    
    print("\nData preparation successful!")
    print(f"  Windows: {len(X_test)}")
    print(f"  Features: {X_test.shape[2]}")
    print(f"  Attack segments: {len(attack_segments_info)}")
