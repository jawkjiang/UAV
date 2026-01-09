"""
Labeling Module

Generate point-level and window-level labels for GPS spoofing detection.
"""
import numpy as np
import pandas as pd
from typing import Dict


def create_point_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate point-level labels for each sample based on attack_start_time column.
    
    Label rule:
        y_t = 0 if t < t_s (normal)
        y_t = 1 if t >= t_s (under attack)
    
    This function assumes that attack information is already embedded in the DataFrame
    through an 'attack_start_time' column created during attack injection.
    
    Args:
        df: DataFrame with flight data and 'attack_start_time' column
    
    Returns:
        DataFrame with added 'label' column
    """
    df = df.copy()
    df['label'] = 0  # Default: normal
    
    # For flights with attacks, label points after attack start
    for flight_id in df['flight'].unique():
        flight_mask = df['flight'] == flight_id
        flight_data = df[flight_mask]
        
        # Check if this flight has an attack (attack_start_time exists and is not None)
        if 'attack_start_time' in df.columns:
            attack_times = flight_data['attack_start_time'].dropna().unique()
            
            if len(attack_times) > 0 and not pd.isna(attack_times[0]):
                t_s = attack_times[0]
                # Label samples at or after attack start time as positive
                attack_sample_mask = flight_mask & (df['time'] >= t_s)
                df.loc[attack_sample_mask, 'label'] = 1
    
    n_positive = (df['label'] == 1).sum()
    n_total = len(df)
    if n_total > 0:
        print(f"    Point-level labels: {n_positive}/{n_total} positive "
              f"({n_positive/n_total*100:.2f}%)")
    
    return df


def generate_point_labels(df: pd.DataFrame, 
                         attack_info_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate point-level labels for each sample.
    
    Label rule:
        y_t = 0 if t < t_s (normal)
        y_t = 1 if t >= t_s (under attack)
    
    Args:
        df: DataFrame with flight data
        attack_info_df: DataFrame with attack information per flight
    
    Returns:
        DataFrame with added 'label' column
    """
    df = df.copy()
    df['label'] = 0  # Default: normal
    
    # Create attack info lookup
    attack_dict = {}
    for _, row in attack_info_df.iterrows():
        attack_dict[row['flight']] = {
            'attacked': row['attacked'],
            'attack_start_time': row.get('attack_start_time', None)
        }
    
    # Assign labels per flight
    for flight_id in df['flight'].unique():
        flight_mask = df['flight'] == flight_id
        
        if flight_id in attack_dict and attack_dict[flight_id]['attacked']:
            t_s = attack_dict[flight_id]['attack_start_time']
            
            # Label samples at or after attack start time as positive
            attack_sample_mask = flight_mask & (df['time'] >= t_s)
            df.loc[attack_sample_mask, 'label'] = 1
    
    n_positive = (df['label'] == 1).sum()
    n_total = len(df)
    print(f"Point-level labels: {n_positive}/{n_total} positive "
          f"({n_positive/n_total*100:.2f}%)")
    
    return df


def get_window_label(window_labels: np.ndarray) -> int:
    """
    Compute window-level label from point-level labels.
    
    Rule: Window label = label of the last (end) point in window
    
    Args:
        window_labels: Array of point-level labels in window
    
    Returns:
        Window-level label (0 or 1)
    """
    return int(window_labels[-1])
