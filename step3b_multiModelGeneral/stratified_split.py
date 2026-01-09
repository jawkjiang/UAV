"""
Stratified data splitting with flight reuse support (Strategy C)

This module implements Conservative Hybrid strategy:
- Train/Val sets: Use flight reuse (each flight generates 6 attack versions + 1 normal)
- Test set: No reuse, stratified sampling ensures all attack types represented
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import config


def stratified_split_with_reuse(
    df: pd.DataFrame,
    attack_types: List[str],
    train_ratio: float = 0.50,
    val_ratio: float = 0.25,
    test_ratio: float = 0.25,
    min_test_per_attack: int = 8,
    random_seed: int = 42
) -> Tuple[List[int], List[int], List[int], Dict[str, List[int]]]:
    """
    Split flights with stratified sampling for test set to ensure attack coverage.
    
    Strategy:
    1. Identify available flights for each attack type
    2. Reserve min_test_per_attack flights per attack for test set (stratified)
    3. Reserve additional normal flights for test set
    4. Split remaining flights into train/val
    5. Train/Val will later be expanded via flight reuse
    6. Test remains original (no reuse)
    
    Args:
        df: DataFrame with flight data
        attack_types: List of attack types to ensure coverage
        train_ratio: Ratio for training (applied to remaining flights after test)
        val_ratio: Ratio for validation
        test_ratio: Ratio for test
        min_test_per_attack: Minimum flights per attack type in test set
        random_seed: Random seed
    
    Returns:
        train_flights: List of flight IDs for training (will be expanded via reuse)
        val_flights: List of flight IDs for validation (will be expanded via reuse)
        test_flights: List of flight IDs for testing (NO reuse)
        test_attack_allocation: Dict mapping attack_type to list of flight IDs
    """
    rng = np.random.RandomState(random_seed)
    all_flights = df['flight'].unique()
    
    print(f"\n{'='*70}")
    print(f"STRATIFIED SPLIT WITH FLIGHT REUSE (Strategy C)")
    print(f"{'='*70}")
    print(f"Total flights: {len(all_flights)}")
    print(f"Target split: Train={train_ratio:.0%}, Val={val_ratio:.0%}, Test={test_ratio:.0%}")
    print(f"Min test flights per attack: {min_test_per_attack}")
    
    # Step 1: Reserve flights for test set (stratified by attack type)
    test_flights = []
    test_attack_allocation = {}
    
    print(f"\nStep 1: Allocating test set with stratified sampling...")
    
    # For each attack type, select flights that will be used for that attack
    for attack_type in attack_types:
        # Get all flights (we'll assign attacks to them later)
        available_flights = list(all_flights)
        
        # Remove already allocated test flights
        available_flights = [f for f in available_flights if f not in test_flights]
        
        # Randomly select flights for this attack type in test set
        if len(available_flights) >= min_test_per_attack:
            selected = rng.choice(
                available_flights, 
                size=min_test_per_attack, 
                replace=False
            )
            test_attack_allocation[attack_type] = list(selected)
            test_flights.extend(selected)
            print(f"  {attack_type:20s}: {len(selected)} flights")
        else:
            print(f"  {attack_type:20s}: WARNING - Not enough flights! "
                  f"(need {min_test_per_attack}, have {len(available_flights)})")
            test_attack_allocation[attack_type] = available_flights
            test_flights.extend(available_flights)
    
    # Add normal flights to test set
    available_flights = [f for f in all_flights if f not in test_flights]
    normal_test_count = min(min_test_per_attack, len(available_flights))
    if normal_test_count > 0:
        normal_selected = rng.choice(available_flights, size=normal_test_count, replace=False)
        test_attack_allocation['none'] = list(normal_selected)
        test_flights.extend(normal_selected)
        print(f"  {'none (normal)':20s}: {normal_test_count} flights")
    
    test_flights = list(set(test_flights))  # Remove duplicates
    print(f"\nTotal test flights: {len(test_flights)}")
    
    # Step 2: Split remaining flights into train and val
    remaining_flights = [f for f in all_flights if f not in test_flights]
    rng.shuffle(remaining_flights)
    
    # Calculate train/val split from remaining
    n_remaining = len(remaining_flights)
    # Adjust ratios: train_ratio + val_ratio should sum to (1 - test_ratio)
    adjusted_train_ratio = train_ratio / (train_ratio + val_ratio)
    n_train = int(n_remaining * adjusted_train_ratio)
    
    train_flights = remaining_flights[:n_train]
    val_flights = remaining_flights[n_train:]
    
    print(f"\nStep 2: Splitting remaining flights...")
    print(f"Remaining flights: {n_remaining}")
    print(f"  Train flights: {len(train_flights)} (will be expanded via reuse)")
    print(f"  Val flights: {len(val_flights)} (will be expanded via reuse)")
    
    # Calculate expected samples after reuse
    n_attacks = len(attack_types)
    train_samples = len(train_flights) * (n_attacks + 1)  # 6 attacks + 1 normal per flight
    val_samples = len(val_flights) * (n_attacks + 1)
    
    print(f"\nExpected samples after reuse:")
    print(f"  Train: {len(train_flights)} flights × {n_attacks + 1} versions = {train_samples} samples")
    print(f"  Val: {len(val_flights)} flights × {n_attacks + 1} versions = {val_samples} samples")
    print(f"  Test: {len(test_flights)} flights (NO reuse)")
    
    return train_flights, val_flights, test_flights, test_attack_allocation


def expand_flights_with_all_attacks(
    flight_ids: List[int],
    attack_types: List[str]
) -> List[Tuple[int, str]]:
    """
    Expand flight list by creating versions for each attack type + normal.
    
    Args:
        flight_ids: List of flight IDs
        attack_types: List of attack types
    
    Returns:
        List of (flight_id, attack_type) tuples
        Each flight appears (len(attack_types) + 1) times
    """
    expanded = []
    
    for flight_id in flight_ids:
        # Add normal version
        expanded.append((flight_id, 'none'))
        
        # Add attacked versions
        for attack_type in attack_types:
            expanded.append((flight_id, attack_type))
    
    return expanded


def assign_attacks_to_test_flights(
    test_flights: List[int],
    test_attack_allocation: Dict[str, List[int]],
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Assign specific attack types to test flights based on allocation.
    
    Args:
        test_flights: List of all test flight IDs
        test_attack_allocation: Dict mapping attack_type to flight IDs
        random_seed: Random seed
    
    Returns:
        DataFrame with columns: flight, attack_type
    """
    assignments = []
    
    # Create reverse mapping: flight_id -> attack_type
    flight_to_attack = {}
    for attack_type, flights in test_attack_allocation.items():
        for flight_id in flights:
            flight_to_attack[flight_id] = attack_type
    
    # Create assignment list
    for flight_id in test_flights:
        attack_type = flight_to_attack.get(flight_id, 'none')
        assignments.append({
            'flight': flight_id,
            'attack_type': attack_type
        })
    
    return pd.DataFrame(assignments)


if __name__ == '__main__':
    # Test the stratified split
    print("Testing stratified split with flight reuse...")
    
    # Create dummy flight data
    n_flights = 209
    dummy_df = pd.DataFrame({'flight': range(n_flights)})
    
    attack_types = ['step', 'drift_ramp', 'drift_sigmoid', 'delay', 'takeover_step', 'takeover_ramp']
    
    train_flights, val_flights, test_flights, test_allocation = stratified_split_with_reuse(
        df=dummy_df,
        attack_types=attack_types,
        train_ratio=0.50,
        val_ratio=0.25,
        test_ratio=0.25,
        min_test_per_attack=8,
        random_seed=42
    )
    
    print(f"\n{'='*70}")
    print("EXPANSION SIMULATION")
    print(f"{'='*70}")
    
    # Simulate expansion
    train_expanded = expand_flights_with_all_attacks(train_flights, attack_types)
    val_expanded = expand_flights_with_all_attacks(val_flights, attack_types)
    
    print(f"\nTrain expanded: {len(train_flights)} flights → {len(train_expanded)} samples")
    print(f"Val expanded: {len(val_flights)} flights → {len(val_expanded)} samples")
    print(f"Test (no expansion): {len(test_flights)} flights")
    
    # Show test set allocation
    print(f"\n{'='*70}")
    print("TEST SET ALLOCATION")
    print(f"{'='*70}")
    for attack_type, flights in test_allocation.items():
        print(f"{attack_type:20s}: {len(flights)} flights")
