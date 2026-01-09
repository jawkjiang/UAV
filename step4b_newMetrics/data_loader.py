"""data_loader.py

Load test predictions and reconstruct full test data from step3b outputs.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import os
import config


def load_model_predictions(model_name: str) -> Dict[str, np.ndarray]:
    """
    Load test predictions for a specific model.
    
    Args:
        model_name: Name of the model (e.g., 'cnn', 'lstm')
    
    Returns:
        Dictionary with arrays: y_true, y_pred, y_prob
    """
    pred_file = config.get_step3b_predictions_file(model_name)
    
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Predictions file not found: {pred_file}")
    
    # Load predictions
    data = np.load(pred_file)
    
    # Handle different possible key names
    if 'y_true' in data:
        y_true = data['y_true']
        y_pred = data['y_pred']
        y_prob = data.get('y_prob', data.get('y_probs', None))
    elif 'labels' in data:
        y_true = data['labels']
        y_pred = data['predictions']
        y_prob = data.get('probabilities', None)
    else:
        raise ValueError(f"Unexpected keys in predictions file: {list(data.keys())}")
    
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob
    }


def load_attack_info(model_name: str) -> pd.DataFrame:
    """
    Load test attack info for a specific model.
    
    Args:
        model_name: Name of the model
    
    Returns:
        DataFrame with attack information
    """
    info_file = config.get_step3b_attack_info_file(model_name)
    
    if not os.path.exists(info_file):
        raise FileNotFoundError(f"Attack info file not found: {info_file}")
    
    return pd.read_csv(info_file)


def reconstruct_test_metadata(model_name: str, 
                              window_size: int = config.WINDOW_SIZE,
                              step_size: int = config.STEP_SIZE) -> Dict[str, np.ndarray]:
    """
    Reconstruct timestamps, flight IDs, and attack types for test windows.
    
    This is a key function that maps window-level predictions back to 
    point-level metadata required for time-aware metrics.
    
    Args:
        model_name: Name of the model
        window_size: Window size used during training
        step_size: Step size used for sliding windows
    
    Returns:
        Dictionary with arrays: timestamps, flight_ids, attack_types
    """
    # Load attack info
    attack_info = load_attack_info(model_name)
    
    # Load predictions to get number of windows
    preds = load_model_predictions(model_name)
    n_windows = len(preds['y_true'])
    
    # Initialize arrays
    timestamps = np.zeros(n_windows)
    flight_ids = np.zeros(n_windows, dtype=int)
    attack_types = np.array(['normal'] * n_windows, dtype=object)
    
    # Build lookup dictionary from attack info
    flight_info = {}
    for _, row in attack_info.iterrows():
        flight_id = int(row['flight'])
        flight_info[flight_id] = {
            'attacked': bool(row['attacked']),
            'attack_type': str(row.get('attack_type', 'none')),
            'attack_start_time': float(row.get('attack_start_time', 0)) if row['attacked'] else None,
            'start_time': float(row.get('start_time', 0)),
            'end_time': float(row.get('end_time', 0)),
            'n_samples': int(row.get('n_samples', 0))
        }
    
    # Process windows sequentially
    # Assumption: windows are ordered by flight and time
    window_idx = 0
    
    for flight_id in sorted(flight_info.keys()):
        info = flight_info[flight_id]
        n_samples = info['n_samples']
        
        # Calculate number of windows for this flight
        n_windows_flight = (n_samples - window_size) // step_size + 1
        
        if n_windows_flight <= 0:
            continue
        
        # Generate window center times
        for i in range(n_windows_flight):
            if window_idx >= n_windows:
                break
            
            # Center of window in terms of sample index
            window_center_sample = i * step_size + window_size // 2
            
            # Convert to time (assuming 100 Hz sampling)
            window_center_time = info['start_time'] + window_center_sample * config.TIME_PER_SAMPLE
            
            timestamps[window_idx] = window_center_time
            flight_ids[window_idx] = flight_id
            
            # Determine attack type for this window
            if info['attacked'] and info['attack_start_time'] is not None:
                if window_center_time >= info['attack_start_time']:
                    attack_types[window_idx] = info['attack_type']
                else:
                    attack_types[window_idx] = 'normal'
            else:
                attack_types[window_idx] = 'normal'
            
            window_idx += 1
    
    # Verify we processed all windows
    if window_idx != n_windows:
        print(f"Warning: Processed {window_idx} windows but expected {n_windows}")
        # Truncate arrays
        timestamps = timestamps[:window_idx]
        flight_ids = flight_ids[:window_idx]
        attack_types = attack_types[:window_idx]
    
    return {
        'timestamps': timestamps,
        'flight_ids': flight_ids,
        'attack_types': attack_types
    }


def load_model_data(model_name: str) -> Dict[str, np.ndarray]:
    """
    Load all data needed for time-aware evaluation of a model.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Dictionary with all arrays needed for evaluation
    """
    print(f"Loading data for {model_name}...")
    
    # Load predictions
    preds = load_model_predictions(model_name)
    
    # Reconstruct metadata
    metadata = reconstruct_test_metadata(model_name)
    
    # Combine everything
    data = {
        'y_true': preds['y_true'],
        'y_pred': preds['y_pred'],
        'y_prob': preds['y_prob'],
        'timestamps': metadata['timestamps'],
        'flight_ids': metadata['flight_ids'],
        'attack_types': metadata['attack_types']
    }
    
    print(f"  Loaded {len(data['y_true'])} windows")
    print(f"  Flights: {len(np.unique(data['flight_ids']))}")
    print(f"  Attack instances: {(data['y_true'] == 1).sum()}")
    
    return data


def load_all_models() -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load data for all models.
    
    Returns:
        Dictionary mapping model_name -> model_data
    """
    all_data = {}
    
    for model_name in config.MODELS:
        try:
            all_data[model_name] = load_model_data(model_name)
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            continue
    
    return all_data


if __name__ == "__main__":
    # Test data loading
    print("Testing data loader...\n")
    
    # Test single model
    model = 'cnn'
    print(f"=== Testing {model} ===")
    data = load_model_data(model)
    
    print(f"\nData shapes:")
    for key, val in data.items():
        if isinstance(val, np.ndarray):
            print(f"  {key}: {val.shape}, dtype={val.dtype}")
    
    print(f"\nSample data:")
    print(f"  First 5 timestamps: {data['timestamps'][:5]}")
    print(f"  First 5 flight IDs: {data['flight_ids'][:5]}")
    print(f"  First 5 attack types: {data['attack_types'][:5]}")
    print(f"  First 5 labels: {data['y_true'][:5]}")
    
    # Test all models
    print(f"\n=== Testing all models ===")
    all_data = load_all_models()
    print(f"\nSuccessfully loaded {len(all_data)} models: {list(all_data.keys())}")
