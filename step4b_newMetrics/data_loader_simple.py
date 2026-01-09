"""data_loader_simple.py

Simplified data loader that doesn't require precise flight timing reconstruction.
"""

import numpy as np
import pandas as pd
from typing import Dict
import os
import config


def load_model_data_simple(model_name: str) -> Dict[str, np.ndarray]:
    """
    Load model test data with simplified metadata reconstruction.
    
    Args:
        model_name: Name of the model (e.g., 'cnn', 'lstm')
    
    Returns:
        Dictionary with all arrays needed for time-aware evaluation
    """
    print(f"Loading data for {model_name}...")
    
    # Load predictions file
    pred_file = config.get_step3b_predictions_file(model_name)
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Predictions file not found: {pred_file}")
    
    data = np.load(pred_file)
    
    # Handle different possible key names
    if 'labels' in data:
        y_true = data['labels']
        y_pred = data['predictions']
        y_prob = data.get('probabilities', None)
    elif 'y_true' in data:
        y_true = data['y_true']
        y_pred = data['y_pred']
        y_prob = data.get('y_prob', data.get('y_probs', None))
    else:
        raise ValueError(f"Unexpected keys in predictions file: {list(data.keys())}")
    
    n_windows = len(y_true)
    
    # Load attack info
    info_file = config.get_step3b_attack_info_file(model_name)
    if not os.path.exists(info_file):
        raise FileNotFoundError(f"Attack info file not found: {info_file}")
    
    attack_info = pd.read_csv(info_file)
    
    # Create simple sequential timestamps
    # Each window is separated by STEP_SIZE * TIME_PER_SAMPLE seconds
    time_increment = config.STEP_SIZE * config.TIME_PER_SAMPLE
    timestamps = np.arange(n_windows, dtype=float) * time_increment
    
    # Assign flight IDs based on attack info
    # Distribute windows across flights proportionally
    flight_ids = np.zeros(n_windows, dtype=int)
    attack_types = np.array(['normal'] * n_windows, dtype=object)
    
    # Calculate windows per flight (rough estimate)
    n_flights = len(attack_info)
    windows_per_flight = n_windows // n_flights
    
    window_idx = 0
    for idx, row in attack_info.iterrows():
        flight_id = int(row['flight'])
        
        # Determine how many windows belong to this flight
        if idx < n_flights - 1:
            n_win_flight = windows_per_flight
        else:
            # Last flight gets remainder
            n_win_flight = n_windows - window_idx
        
        end_idx = min(window_idx + n_win_flight, n_windows)
        
        # Assign flight ID
        flight_ids[window_idx:end_idx] = flight_id
        
        # Assign attack types based on ground truth labels
        if bool(row['attacked']):
            attack_type = str(row['attack_type'])
            # Mark windows as attacked where y_true == 1
            for i in range(window_idx, end_idx):
                if y_true[i] == 1:
                    attack_types[i] = attack_type
        
        window_idx = end_idx
        
        if window_idx >= n_windows:
            break
    
    print(f"  Loaded {n_windows} windows")
    print(f"  Flights: {len(np.unique(flight_ids))}")
    print(f"  Attack windows: {(y_true == 1).sum()}")
    print(f"  Attack types: {np.unique(attack_types[attack_types != 'normal'])}")
    
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'timestamps': timestamps,
        'flight_ids': flight_ids,
        'attack_types': attack_types
    }


def load_all_models_simple() -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load data for all models using simplified approach.
    
    Returns:
        Dictionary mapping model_name -> model_data
    """
    all_data = {}
    
    for model_name in config.MODELS:
        try:
            all_data[model_name] = load_model_data_simple(model_name)
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return all_data


if __name__ == "__main__":
    # Test data loading
    print("Testing simplified data loader...\n")
    
    # Test single model
    model = 'cnn'
    print(f"=== Testing {model} ===")
    data = load_model_data_simple(model)
    
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
    all_data = load_all_models_simple()
    print(f"\nSuccessfully loaded {len(all_data)} models: {list(all_data.keys())}")
