"""
Data loading and preprocessing module
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import config


def load_flights_data(data_path: str = config.DATA_PATH) -> pd.DataFrame:
    """
    Load flight data from CSV file.
    
    Returns:
        DataFrame with all flights
    """
    df = pd.read_csv(data_path)
    
    # Sort by flight and time to ensure temporal ordering
    df = df.sort_values(['flight', 'time']).reset_index(drop=True)
    
    print(f"Loaded {len(df)} rows from {len(df['flight'].unique())} flights")
    return df


def split_flights(df: pd.DataFrame, 
                  train_ratio: float = config.TRAIN_RATIO,
                  val_ratio: float = config.VAL_RATIO,
                  test_ratio: float = config.TEST_RATIO,
                  random_seed: int = config.RANDOM_SEED) -> Tuple[List[int], List[int], List[int]]:
    """
    Split flights into train, validation, and test sets.
    
    Args:
        df: DataFrame with flight data
        train_ratio: Ratio of flights for training
        val_ratio: Ratio of flights for validation
        test_ratio: Ratio of flights for testing
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_flights, val_flights, test_flights)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    # Get unique flight IDs
    flights = sorted(df['flight'].unique())
    n_flights = len(flights)
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Shuffle flights
    shuffled_flights = np.random.permutation(flights)
    
    # Calculate split indices
    n_train = int(n_flights * train_ratio)
    n_val = int(n_flights * val_ratio)
    
    # Split flights
    train_flights = sorted(shuffled_flights[:n_train].tolist())
    val_flights = sorted(shuffled_flights[n_train:n_train + n_val].tolist())
    test_flights = sorted(shuffled_flights[n_train + n_val:].tolist())
    
    print(f"Split flights: Train={len(train_flights)}, Val={len(val_flights)}, Test={len(test_flights)}")
    
    return train_flights, val_flights, test_flights


def get_flight_subset(df: pd.DataFrame, flight_ids: List[int]) -> pd.DataFrame:
    """
    Extract subset of DataFrame for specific flights.
    
    Args:
        df: Full DataFrame
        flight_ids: List of flight IDs to extract
    
    Returns:
        DataFrame subset for specified flights
    """
    return df[df['flight'].isin(flight_ids)].copy()


def compute_delta_t(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute real time deltas (Δt) for each sample.
    Must be computed per flight, cannot cross flight boundaries.
    
    Args:
        df: DataFrame with 'flight' and 'time' columns
    
    Returns:
        DataFrame with added 'delta_t' column
    """
    df = df.copy()
    df['delta_t'] = 0.0
    
    for flight_id in df['flight'].unique():
        flight_mask = df['flight'] == flight_id
        flight_indices = df[flight_mask].index
        
        # Compute delta_t for this flight
        time_values = df.loc[flight_indices, 'time'].values
        delta_t = np.diff(time_values, prepend=time_values[0])
        delta_t[0] = 0.0  # First sample has no previous sample
        
        df.loc[flight_indices, 'delta_t'] = delta_t
    
    return df


def convert_to_local_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert position from lat/lon (if needed) to local planar coordinates.
    Currently assumes position_x, position_y are already in meters.
    
    This function is a placeholder for potential coordinate conversion.
    
    Args:
        df: DataFrame with position_x, position_y
    
    Returns:
        DataFrame with local coordinates
    """
    # Currently a no-op since data appears to be in local coordinates already
    # If lat/lon conversion is needed, implement here
    return df


if __name__ == '__main__':
    import os
    from attack_injector import GPSSpoofingInjector
    from labeling import create_point_labels
    
    print("="*80)
    print("Data Loading and Attack Injection")
    print("="*80)
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # 1. Load data
    print("\n[1] Loading flight data...")
    df = load_flights_data()
    print(f"    Loaded {len(df)} samples from {len(df['flight'].unique())} flights")
    
    # 2. Split flights
    print("\n[2] Splitting flights...")
    train_flights, val_flights, test_flights = split_flights(df)
    
    # 3. Compute delta_t
    print("\n[3] Computing time deltas...")
    df = compute_delta_t(df)
    
    # 4. Convert coordinates (if needed)
    df = convert_to_local_coordinates(df)
    
    # 5. Split data
    train_df = get_flight_subset(df, train_flights)
    val_df = get_flight_subset(df, val_flights)
    test_df = get_flight_subset(df, test_flights)
    
    print(f"    Train: {len(train_df)} samples from {len(train_flights)} flights")
    print(f"    Val:   {len(val_df)} samples from {len(val_flights)} flights")
    print(f"    Test:  {len(test_df)} samples from {len(test_flights)} flights")
    
    # 6. Inject attacks
    print("\n[4] Injecting GPS spoofing attacks...")
    injector = GPSSpoofingInjector(random_seed=config.RANDOM_SEED)
    
    # Train set: inject attacks
    print("    Injecting attacks to training set...")
    train_attacked = []
    train_attack_stats = []
    for flight_id in train_flights:
        flight_data = train_df[train_df['flight'] == flight_id].copy()
        attacked_flight, attack_info = injector.inject_attack_to_flight(
            flight_data, attack_prob=1.0
        )
        train_attacked.append(attacked_flight)
        if attack_info['attacked']:
            train_attack_stats.append(attack_info)
    
    train_df = pd.concat(train_attacked, ignore_index=True)
    print(f"    ✓ Attacked {len(train_attack_stats)}/{len(train_flights)} training flights")
    
    # Val set: inject attacks with lower probability
    print("    Injecting attacks to validation set...")
    val_attacked = []
    val_attack_stats = []
    for flight_id in val_flights:
        flight_data = val_df[val_df['flight'] == flight_id].copy()
        attacked_flight, attack_info = injector.inject_attack_to_flight(
            flight_data, attack_prob=0.3  # Lower attack rate for val
        )
        val_attacked.append(attacked_flight)
        if attack_info['attacked']:
            val_attack_stats.append(attack_info)
    
    val_df = pd.concat(val_attacked, ignore_index=True)
    print(f"    ✓ Attacked {len(val_attack_stats)}/{len(val_flights)} validation flights")
    
    # Test set: inject attacks with lower probability
    print("    Injecting attacks to test set...")
    test_attacked = []
    test_attack_stats = []
    for flight_id in test_flights:
        flight_data = test_df[test_df['flight'] == flight_id].copy()
        attacked_flight, attack_info = injector.inject_attack_to_flight(
            flight_data, attack_prob=0.3  # Lower attack rate for test
        )
        test_attacked.append(attacked_flight)
        if attack_info['attacked']:
            test_attack_stats.append(attack_info)
    
    test_df = pd.concat(test_attacked, ignore_index=True)
    print(f"    ✓ Attacked {len(test_attack_stats)}/{len(test_flights)} test flights")
    
    # 7. Create labels
    print("\n[5] Creating point-level labels...")
    train_df = create_point_labels(train_df)
    val_df = create_point_labels(val_df)
    test_df = create_point_labels(test_df)
    
    train_pos_ratio = train_df['label'].mean()
    val_pos_ratio = val_df['label'].mean()
    test_pos_ratio = test_df['label'].mean()
    
    print(f"    Train positive ratio: {train_pos_ratio:.2%}")
    print(f"    Val positive ratio:   {val_pos_ratio:.2%}")
    print(f"    Test positive ratio:  {test_pos_ratio:.2%}")
    
    # 8. Save processed data
    print("\n[6] Saving processed data...")
    train_df.to_pickle(os.path.join(config.OUTPUT_DIR, 'train_data.pkl'))
    val_df.to_pickle(os.path.join(config.OUTPUT_DIR, 'val_data.pkl'))
    test_df.to_pickle(os.path.join(config.OUTPUT_DIR, 'test_data.pkl'))
    
    print(f"    ✓ Saved to {config.OUTPUT_DIR}/")
    
    # 9. Save attack statistics
    import json
    attack_stats = {
        'train': {
            'total_flights': len(train_flights),
            'attacked_flights': len(train_attack_stats),
            'attack_magnitudes': [s['attack_magnitude'] for s in train_attack_stats]
        },
        'val': {
            'total_flights': len(val_flights),
            'attacked_flights': len(val_attack_stats),
            'attack_magnitudes': [s['attack_magnitude'] for s in val_attack_stats]
        },
        'test': {
            'total_flights': len(test_flights),
            'attacked_flights': len(test_attack_stats),
            'attack_magnitudes': [s['attack_magnitude'] for s in test_attack_stats]
        }
    }
    
    with open(os.path.join(config.OUTPUT_DIR, 'attack_stats.json'), 'w') as f:
        json.dump(attack_stats, f, indent=2)
    
    print("\n" + "="*80)
    print("Data loading and attack injection complete!")
    print("="*80)
