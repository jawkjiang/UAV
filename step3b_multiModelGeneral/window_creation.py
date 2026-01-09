"""
Window Creation and Dataset Module

Create sliding windows for training, validation, and test sets.
Control positive sample ratios through window-level sampling.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple
import torch
from torch.utils.data import Dataset
import config
from labeling import get_window_label


def create_windows_from_flight(flight_df: pd.DataFrame,
                               feature_columns: List[str],
                               window_size: int = config.WINDOW_SIZE,
                               step_size: int = config.STEP_SIZE) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Create sliding windows from a single flight.
    
    Windows CANNOT cross flight boundaries.
    
    Args:
        flight_df: DataFrame for single flight (sorted by time)
        feature_columns: List of feature column names
        window_size: Number of samples per window
        step_size: Step size for sliding window
    
    Returns:
        Tuple of (windows_array, labels_array, flight_id)
        - windows_array: shape [n_windows, window_size, n_features]
        - labels_array: shape [n_windows], window-level labels
    """
    n_samples = len(flight_df)
    
    if n_samples < window_size:
        # Flight too short, no windows can be created
        return np.array([]), np.array([]), flight_df['flight'].iloc[0]
    
    # Extract features and labels
    features = flight_df[feature_columns].values  # [n_samples, n_features]
    labels = flight_df['label'].values  # [n_samples]
    
    windows = []
    window_labels = []
    
    # Create sliding windows
    for start_idx in range(0, n_samples - window_size + 1, step_size):
        end_idx = start_idx + window_size
        
        # Extract window
        window = features[start_idx:end_idx]  # [window_size, n_features]
        window_point_labels = labels[start_idx:end_idx]
        
        # Compute window-level label (label of last point)
        window_label = get_window_label(window_point_labels)
        
        windows.append(window)
        window_labels.append(window_label)
    
    windows_array = np.array(windows)  # [n_windows, window_size, n_features]
    labels_array = np.array(window_labels)  # [n_windows]
    flight_id = flight_df['flight'].iloc[0]
    
    return windows_array, labels_array, flight_id


def create_windows_from_dataset(df: pd.DataFrame,
                                feature_columns: List[str],
                                window_size: int = config.WINDOW_SIZE,
                                step_size: int = config.STEP_SIZE) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create windows from all flights in dataset.
    
    Args:
        df: DataFrame with multiple flights
        feature_columns: List of feature column names
        window_size: Window size
        step_size: Step size
    
    Returns:
        Tuple of (all_windows, all_labels, flight_ids)
    """
    all_windows = []
    all_labels = []
    all_flight_ids = []
    
    flights = df['flight'].unique()
    
    for flight_id in flights:
        flight_df = df[df['flight'] == flight_id].copy()
        
        windows, labels, fid = create_windows_from_flight(
            flight_df, feature_columns, window_size, step_size
        )
        
        if len(windows) > 0:
            all_windows.append(windows)
            all_labels.append(labels)
            all_flight_ids.extend([fid] * len(labels))
    
    # Concatenate all windows
    if len(all_windows) > 0:
        all_windows = np.concatenate(all_windows, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_flight_ids = np.array(all_flight_ids)
    else:
        all_windows = np.array([])
        all_labels = np.array([])
        all_flight_ids = np.array([])
    
    return all_windows, all_labels, all_flight_ids


def balance_windows(windows: np.ndarray,
                   labels: np.ndarray,
                   flight_ids: np.ndarray,
                   target_pos_ratio: float,
                   random_seed: int = config.RANDOM_SEED) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Balance window samples to achieve target positive ratio.
    
    For training: target ~30-40% positive
    For val/test: target ~1-5% positive
    
    Args:
        windows: Window features
        labels: Window labels
        flight_ids: Flight IDs for each window
        target_pos_ratio: Target ratio of positive samples
        random_seed: Random seed
    
    Returns:
        Tuple of (balanced_windows, balanced_labels, balanced_flight_ids)
    """
    np.random.seed(random_seed)
    
    # Get positive and negative indices
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    
    current_pos_ratio = n_pos / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0
    
    print(f"  Current positive ratio: {current_pos_ratio:.3f} ({n_pos}/{n_pos+n_neg})")
    
    # Special case: if no positive samples, keep all negatives
    if n_pos == 0:
        print(f"  Warning: No positive samples found. Keeping all {n_neg} negative samples.")
        return windows, labels, flight_ids
    
    if current_pos_ratio < target_pos_ratio:
        # Need more positive samples (or fewer negative)
        # Calculate required number of negatives
        n_neg_target = int(n_pos / target_pos_ratio - n_pos)
        n_neg_target = max(1, min(n_neg_target, n_neg))
        
        # Sample negative windows
        neg_idx_sampled = np.random.choice(neg_idx, size=n_neg_target, replace=False)
        selected_idx = np.concatenate([pos_idx, neg_idx_sampled])
    else:
        # Need fewer positive samples
        n_pos_target = int(n_neg * target_pos_ratio / (1 - target_pos_ratio))
        n_pos_target = max(1, min(n_pos_target, n_pos))
        
        # Sample positive windows
        pos_idx_sampled = np.random.choice(pos_idx, size=n_pos_target, replace=False)
        selected_idx = np.concatenate([pos_idx_sampled, neg_idx])
    
    # Shuffle selected indices
    np.random.shuffle(selected_idx)
    
    balanced_windows = windows[selected_idx]
    balanced_labels = labels[selected_idx]
    balanced_flight_ids = flight_ids[selected_idx]
    
    final_pos_ratio = balanced_labels.sum() / len(balanced_labels)
    print(f"  Balanced positive ratio: {final_pos_ratio:.3f} "
          f"({balanced_labels.sum()}/{len(balanced_labels)})")
    
    return balanced_windows, balanced_labels, balanced_flight_ids


class WindowDataset(Dataset):
    """
    PyTorch Dataset for window-based GPS spoofing detection.
    """
    
    def __init__(self, windows: np.ndarray, labels: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            windows: [n_windows, window_size, n_features]
            labels: [n_windows]
        """
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]
    
    def get_positive_weight(self):
        """
        Compute weight for positive class in loss function.
        
        Returns:
            Weight for positive samples
        """
        n_pos = self.labels.sum().item()
        n_neg = len(self.labels) - n_pos
        
        if n_pos == 0:
            return 1.0
        
        # Weight = n_neg / n_pos
        return n_neg / n_pos


if __name__ == '__main__':
    import os
    import pickle
    from feature_engineering import create_consistency_features
    
    print("="*80)
    print("Window Creation and Feature Engineering")
    print("="*80)
    
    # Load processed data
    print("\n[1] Loading processed data...")
    train_df = pd.read_pickle(os.path.join(config.OUTPUT_DIR, 'train_data.pkl'))
    val_df = pd.read_pickle(os.path.join(config.OUTPUT_DIR, 'val_data.pkl'))
    test_df = pd.read_pickle(os.path.join(config.OUTPUT_DIR, 'test_data.pkl'))
    
    print(f"    Train: {len(train_df)} samples")
    print(f"    Val:   {len(val_df)} samples")
    print(f"    Test:  {len(test_df)} samples")
    
    # Create consistency features
    print("\n[2] Creating consistency features...")
    train_df = create_consistency_features(train_df)
    val_df = create_consistency_features(val_df)
    test_df = create_consistency_features(test_df)
    
    feature_columns = [
        'position_x', 'position_y',
        'velocity_x', 'velocity_y',
        'linear_acceleration_x', 'linear_acceleration_y',
        'delta_t',
        'residual_pos_vel',
        'residual_vel_acc'
    ]
    
    print(f"    Using {len(feature_columns)} features: {feature_columns}")
    
    # Create windows for each split
    print("\n[3] Creating windows...")
    
    # Training set
    print("    Processing training set...")
    train_windows, train_labels, train_flight_ids = create_windows_from_dataset(
        train_df, feature_columns
    )
    
    # Balance training set to target positive ratio
    print("    Balancing training set...")
    train_windows, train_labels, train_flight_ids = balance_windows(
        train_windows, train_labels, train_flight_ids,
        target_pos_ratio=0.35,  # Target 30-40%
        random_seed=config.RANDOM_SEED
    )
    
    # Validation set
    print("    Processing validation set...")
    val_windows, val_labels, val_flight_ids = create_windows_from_dataset(
        val_df, feature_columns
    )
    
    # Balance validation set to target positive ratio
    print("    Balancing validation set...")
    val_windows, val_labels, val_flight_ids = balance_windows(
        val_windows, val_labels, val_flight_ids,
        target_pos_ratio=0.03,  # Target 1-5%
        random_seed=config.RANDOM_SEED + 1
    )
    
    # Test set
    print("    Processing test set...")
    test_windows, test_labels, test_flight_ids = create_windows_from_dataset(
        test_df, feature_columns
    )
    
    # Balance test set to target positive ratio
    print("    Balancing test set...")
    test_windows, test_labels, test_flight_ids = balance_windows(
        test_windows, test_labels, test_flight_ids,
        target_pos_ratio=0.03,  # Target 1-5%
        random_seed=config.RANDOM_SEED + 2
    )
    
    # Summary
    print("\n[4] Window creation summary:")
    print(f"    Train: {len(train_labels)} windows, "
          f"positive ratio: {train_labels.mean():.2%}")
    print(f"    Val:   {len(val_labels)} windows, "
          f"positive ratio: {val_labels.mean():.2%}")
    print(f"    Test:  {len(test_labels)} windows, "
          f"positive ratio: {test_labels.mean():.2%}")
    
    # Save windows
    print("\n[5] Saving windows...")
    windows_data = {
        'train': {
            'windows': train_windows,
            'labels': train_labels,
            'flight_ids': train_flight_ids
        },
        'val': {
            'windows': val_windows,
            'labels': val_labels,
            'flight_ids': val_flight_ids
        },
        'test': {
            'windows': test_windows,
            'labels': test_labels,
            'flight_ids': test_flight_ids
        },
        'feature_columns': feature_columns
    }
    
    with open(os.path.join(config.OUTPUT_DIR, 'windows_data.pkl'), 'wb') as f:
        pickle.dump(windows_data, f)
    
    print(f"    ✓ Saved to {config.OUTPUT_DIR}/windows_data.pkl")
    
    print("\n" + "="*80)
    print("Window creation complete!")
    print("="*80)
