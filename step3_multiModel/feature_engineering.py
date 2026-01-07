"""
Feature Engineering Module

Compute consistency residual features and prepare feature matrix.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple
import config


def compute_position_velocity_residual(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute position-velocity consistency residual.
    
    r_t = |Δp_t - v_t * Δt_t|
    
    where Δp_t = p_t - p_{t-1}
    
    Args:
        df: DataFrame with position, velocity, and delta_t
    
    Returns:
        DataFrame with added residual columns
    """
    df = df.copy()
    
    # Initialize residuals
    df['residual_pv_x'] = 0.0
    df['residual_pv_y'] = 0.0
    df['residual_pv_norm'] = 0.0
    
    # Compute per flight to avoid cross-flight calculations
    for flight_id in df['flight'].unique():
        flight_mask = df['flight'] == flight_id
        flight_indices = df[flight_mask].index
        
        if len(flight_indices) < 2:
            continue
        
        # Get position, velocity, delta_t arrays
        pos_x = df.loc[flight_indices, 'position_x'].values
        pos_y = df.loc[flight_indices, 'position_y'].values
        vel_x = df.loc[flight_indices, 'velocity_x'].values
        vel_y = df.loc[flight_indices, 'velocity_y'].values
        delta_t = df.loc[flight_indices, 'delta_t'].values
        
        # Compute position delta
        delta_p_x = np.diff(pos_x, prepend=pos_x[0])
        delta_p_y = np.diff(pos_y, prepend=pos_y[0])
        delta_p_x[0] = 0.0
        delta_p_y[0] = 0.0
        
        # Compute residual: |Δp - v*Δt|
        residual_x = delta_p_x - vel_x * delta_t
        residual_y = delta_p_y - vel_y * delta_t
        residual_norm = np.sqrt(residual_x**2 + residual_y**2)
        
        df.loc[flight_indices, 'residual_pv_x'] = residual_x
        df.loc[flight_indices, 'residual_pv_y'] = residual_y
        df.loc[flight_indices, 'residual_pv_norm'] = residual_norm
    
    return df


def compute_velocity_acceleration_residual(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute velocity-acceleration consistency residual.
    
    r^a_t = |(v_t - v_{t-1})/Δt_t - a_t|
    
    Args:
        df: DataFrame with velocity, acceleration, and delta_t
    
    Returns:
        DataFrame with added residual columns
    """
    df = df.copy()
    
    # Initialize residuals
    df['residual_va_x'] = 0.0
    df['residual_va_y'] = 0.0
    df['residual_va_norm'] = 0.0
    
    # Compute per flight
    for flight_id in df['flight'].unique():
        flight_mask = df['flight'] == flight_id
        flight_indices = df[flight_mask].index
        
        if len(flight_indices) < 2:
            continue
        
        vel_x = df.loc[flight_indices, 'velocity_x'].values
        vel_y = df.loc[flight_indices, 'velocity_y'].values
        acc_x = df.loc[flight_indices, 'linear_acceleration_x'].values
        acc_y = df.loc[flight_indices, 'linear_acceleration_y'].values
        delta_t = df.loc[flight_indices, 'delta_t'].values
        
        # Compute velocity delta
        delta_v_x = np.diff(vel_x, prepend=vel_x[0])
        delta_v_y = np.diff(vel_y, prepend=vel_y[0])
        delta_v_x[0] = 0.0
        delta_v_y[0] = 0.0
        
        # Compute residual: |(Δv/Δt) - a|
        residual_x = np.zeros_like(vel_x)
        residual_y = np.zeros_like(vel_y)
        
        for i in range(1, len(vel_x)):
            if delta_t[i] > 0:
                residual_x[i] = (delta_v_x[i] / delta_t[i]) - acc_x[i]
                residual_y[i] = (delta_v_y[i] / delta_t[i]) - acc_y[i]
        
        residual_norm = np.sqrt(residual_x**2 + residual_y**2)
        
        df.loc[flight_indices, 'residual_va_x'] = residual_x
        df.loc[flight_indices, 'residual_va_y'] = residual_y
        df.loc[flight_indices, 'residual_va_norm'] = residual_norm
    
    return df


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all engineered features.
    
    Args:
        df: DataFrame with basic sensor data
    
    Returns:
        DataFrame with all features
    """
    df = compute_position_velocity_residual(df)
    df = compute_velocity_acceleration_residual(df)
    return df


def create_consistency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create consistency residual features.
    Alias for compute_all_features for backward compatibility.
    
    Args:
        df: DataFrame with basic sensor data
    
    Returns:
        DataFrame with consistency features added
    """
    # Compute position-velocity residual
    df = compute_position_velocity_residual(df)
    
    # Compute velocity-acceleration residual
    df = compute_velocity_acceleration_residual(df)
    
    # Simplify: create aggregated residual features
    df['residual_pos_vel'] = df['residual_pv_norm']
    df['residual_vel_acc'] = df['residual_va_norm']
    
    return df


def get_feature_columns() -> List[str]:
    """
    Get list of all feature column names for model input.
    
    Returns:
        List of feature column names
    """
    features = []
    
    # Basic features
    features.extend(config.POSITION_FEATURES)
    features.extend(config.VELOCITY_FEATURES)
    features.extend(config.ACCELERATION_FEATURES)
    features.append('delta_t')
    
    # Consistency residual features
    features.extend([
        'residual_pv_x', 'residual_pv_y', 'residual_pv_norm',
        'residual_va_x', 'residual_va_y', 'residual_va_norm'
    ])
    
    return features


def normalize_features(train_df: pd.DataFrame, 
                      val_df: pd.DataFrame, 
                      test_df: pd.DataFrame,
                      feature_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Normalize features using train set statistics.
    
    Args:
        train_df: Training data
        val_df: Validation data
        test_df: Test data
        feature_columns: List of features to normalize
    
    Returns:
        Tuple of (normalized_train, normalized_val, normalized_test, normalization_stats)
    """
    # Compute mean and std from training set
    train_mean = train_df[feature_columns].mean()
    train_std = train_df[feature_columns].std()
    
    # Avoid division by zero
    train_std = train_std.replace(0, 1)
    
    # Normalize all sets using train statistics
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    train_df[feature_columns] = (train_df[feature_columns] - train_mean) / train_std
    val_df[feature_columns] = (val_df[feature_columns] - train_mean) / train_std
    test_df[feature_columns] = (test_df[feature_columns] - train_mean) / train_std
    
    normalization_stats = {
        'mean': train_mean.to_dict(),
        'std': train_std.to_dict()
    }
    
    return train_df, val_df, test_df, normalization_stats
