"""
Data Loader with Complete Metadata
使用step3b生成的完整元数据进行时间感知评估
"""
import numpy as np
import pandas as pd
from typing import Dict
import os
import config


def load_model_data_with_metadata(model_name: str) -> Dict[str, np.ndarray]:
    """
    加载模型测试数据及完整的窗口元数据
    
    Args:
        model_name: 模型名称
    
    Returns:
        包含所有必要数据的字典：
        - y_true: 真实标签 [n_windows]
        - y_pred: 二值化预测 (probabilities > 0.5) [n_windows]
        - y_prob: 预测概率 [n_windows]
        - timestamps: 每个窗口的时间戳（秒）[n_windows]
        - flight_ids: 每个窗口的飞行ID [n_windows]
        - attack_types: 每个窗口的攻击类型 [n_windows]
        - window_indices: 窗口索引 [n_windows]
        - attack_segments_info: 攻击片段信息列表
    """
    print(f"Loading data with metadata for {model_name}...")
    
    model_dir = config.get_step3b_model_dir(model_name)
    
    # 1. 加载预测文件
    pred_file = os.path.join(model_dir, 'test_predictions.npz')
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Predictions file not found: {pred_file}")
    
    data = np.load(pred_file)
    
    # 使用probabilities并二值化（关键修复！）
    y_prob = data['probabilities']
    y_pred = (y_prob > 0.5).astype(int)
    y_true = data['labels'].astype(int)
    
    # 从npz获取元数据（如果存在）
    if 'flight_ids' in data:
        flight_ids = data['flight_ids'].astype(int)
        window_indices = data.get('window_indices', np.arange(len(y_true)))
        
        # 获取配置参数
        window_size = int(data.get('window_size', config.WINDOW_SIZE))
        step_size = int(data.get('step_size', config.STEP_SIZE))
        sampling_rate = float(data.get('sampling_rate', config.SAMPLING_RATE))
    else:
        # 回退方案
        flight_ids = None
        window_indices = np.arange(len(y_true))
        window_size = config.WINDOW_SIZE
        step_size = config.STEP_SIZE
        sampling_rate = config.SAMPLING_RATE
    
    # 2. 加载window_metadata.csv
    metadata_file = os.path.join(model_dir, 'window_metadata.csv')
    attack_info_file = os.path.join(model_dir, 'test_attack_info.csv')
    
    if os.path.exists(metadata_file):
        print(f"  ✓ Loading window_metadata.csv")
        metadata = pd.read_csv(metadata_file)
        
        # 确保索引对齐
        metadata = metadata.sort_values('window_index').reset_index(drop=True)
        
        if flight_ids is None:
            flight_ids = metadata['flight_id'].values.astype(int)
        
        attack_types = metadata['attack_type'].fillna('normal').values
        
        # 获取攻击片段信息
        attacked_flights = metadata[metadata['attacked'] == True]
        attack_segments_info = []
        
        for flight_id in attacked_flights['flight_id'].unique():
            flight_attack = attacked_flights[attacked_flights['flight_id'] == flight_id].iloc[0]
            attack_segments_info.append({
                'flight_id': int(flight_id),
                'attack_type': flight_attack['attack_type'],
                'attack_start_time': float(flight_attack['attack_start_time'])
            })
        
    else:
        print(f"  ⚠ window_metadata.csv not found, using fallback method")
        attack_types = np.array(['normal'] * len(y_true))
        attack_segments_info = []
        
        if os.path.exists(attack_info_file):
            attack_info = pd.read_csv(attack_info_file)
            
            if flight_ids is None:
                # 均匀分配窗口到飞行
                flight_ids = np.zeros(len(y_true), dtype=int)
                n_flights = len(attack_info)
                windows_per_flight = len(y_true) // n_flights
                
                for i, row in attack_info.iterrows():
                    start_idx = i * windows_per_flight
                    end_idx = min(start_idx + windows_per_flight, len(y_true))
                    if end_idx <= len(flight_ids):
                        flight_ids[start_idx:end_idx] = int(row['flight'])
                        if row['attacked']:
                            attack_types[start_idx:end_idx] = row['attack_type']
            
            # 构建attack_segments_info
            for _, row in attack_info.iterrows():
                if row['attacked']:
                    attack_segments_info.append({
                        'flight_id': int(row['flight']),
                        'attack_type': row['attack_type'],
                        'attack_start_time': float(row['attack_start_time'])
                    })
    
    # 3. 生成时间戳
    # 使用窗口索引（在原始数据中的位置）计算时间戳
    # 每个窗口的中心时间 = 窗口起始索引 * 采样间隔 + 窗口持续时间/2
    sampling_interval = 1.0 / sampling_rate  # 秒
    window_duration = window_size * sampling_interval  # 秒
    
    # window_indices是窗口在原始数据中的起始索引
    # 窗口中心时间 = 起始采样点时间 + 窗口持续时间/2
    timestamps = window_indices * sampling_interval + window_duration / 2
    
    print(f"  Loaded {len(y_true)} windows")
    print(f"  Flights: {len(np.unique(flight_ids))}")
    print(f"  Attack windows (label=1): {(y_true == 1).sum()}")
    print(f"  Predicted attack windows (pred=1): {(y_pred == 1).sum()}")
    print(f"  Detection rate: {(y_pred == 1).sum() / max(1, (y_true == 1).sum()):.2%}")
    print(f"  Timestamp range: {timestamps.min():.2f}s - {timestamps.max():.2f}s")
    print(f"  Attack types: {np.unique(attack_types[attack_types != 'normal'])}")
    print(f"  Attack segments with timing: {len(attack_segments_info)}")
    
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'timestamps': timestamps,
        'flight_ids': flight_ids,
        'attack_types': attack_types,
        'window_indices': window_indices,
        'attack_segments_info': attack_segments_info
    }


if __name__ == '__main__':
    # 测试数据加载
    print("Testing data loader with metadata...")
    print("="*70)
    
    for model_name in ['tcn', 'cnn']:
        try:
            print(f"\nTesting {model_name}:")
            data = load_model_data_with_metadata(model_name)
            print(f"✓ Success!")
            print(f"  Keys: {list(data.keys())}")
            print()
        except Exception as e:
            print(f"✗ Error: {e}")
            print()
