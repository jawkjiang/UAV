"""
GPS Spoofing Attack Injection Module

Implements position-level step-type GPS spoofing with:
- Step position offset
- Exponentially decaying velocity transient
- Derived acceleration transient
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict
import config


class GPSSpoofingInjector:
    """
    Inject position-level GPS spoofing attacks into flight data.
    
    Attack characteristics:
    - Step offset in position (horizontal plane)
    - Exponentially decaying velocity transient
    - Derived acceleration transient
    - NO modification to orientation
    """
    
    def __init__(self, random_seed: int = config.RANDOM_SEED):
        """
        Initialize injector with random seed.
        
        Args:
            random_seed: Seed for reproducible random attack parameters
        """
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
    
    def inject_attack_to_flight(self, 
                                flight_df: pd.DataFrame,
                                attack_prob: float = 1.0) -> Tuple[pd.DataFrame, Dict]:
        """
        Inject GPS spoofing attack into a single flight.
        
        Args:
            flight_df: DataFrame for single flight (sorted by time)
            attack_prob: Probability of injecting attack (0.0 to 1.0)
        
        Returns:
            Tuple of (modified_df, attack_info_dict)
        """
        df = flight_df.copy()
        
        # Initialize attack_start_time column
        df['attack_start_time'] = np.nan
        
        # Decide whether to attack this flight
        if self.rng.random() >= attack_prob:
            attack_info = {
                'attacked': False,
                'attack_start_time': None,
                'attack_magnitude': None,
                'attack_direction': None
            }
            return df, attack_info
        
        # Get flight time bounds
        times = df['time'].values
        min_time = times.min()
        max_time = times.max()
        
        # Check if flight is long enough for attack
        flight_duration = max_time - min_time
        if flight_duration < (config.ATTACK_START_BUFFER + config.ATTACK_END_BUFFER):
            # Flight too short, skip attack
            attack_info = {
                'attacked': False,
                'attack_start_time': None,
                'attack_magnitude': None,
                'attack_direction': None,
                'reason': 'flight_too_short'
            }
            return df, attack_info
        
        # Sample attack start time
        t_s_min = min_time + config.ATTACK_START_BUFFER
        t_s_max = max_time - config.ATTACK_END_BUFFER
        t_s = self.rng.uniform(t_s_min, t_s_max)
        
        # Mark attack start time in DataFrame
        df['attack_start_time'] = t_s
        
        # Sample attack magnitude (horizontal offset norm)
        magnitude = self.rng.choice(config.ATTACK_MAGNITUDES)
        
        # Sample random horizontal direction (2D unit vector)
        angle = self.rng.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])
        
        # Sample velocity transient parameters
        v_max = self.rng.uniform(config.VELOCITY_TRANSIENT_MIN, 
                                 config.VELOCITY_TRANSIENT_MAX)
        
        # Apply position offset
        attack_mask = times >= t_s
        position_offset = magnitude * direction
        
        df.loc[attack_mask, 'position_x'] += position_offset[0]
        df.loc[attack_mask, 'position_y'] += position_offset[1]
        
        # Apply velocity and acceleration transients
        df = self._apply_velocity_acceleration_transient(
            df, t_s, v_max, direction, times
        )
        
        attack_info = {
            'attacked': True,
            'attack_start_time': t_s,
            'attack_magnitude': magnitude,
            'attack_direction': direction.tolist(),
            'velocity_transient_max': v_max
        }
        
        return df, attack_info
    
    def _apply_velocity_acceleration_transient(self,
                                               df: pd.DataFrame,
                                               t_s: float,
                                               v_max: float,
                                               direction: np.ndarray,
                                               times: np.ndarray) -> pd.DataFrame:
        """
        Apply exponentially decaying velocity transient and derived acceleration.
        
        Velocity: Δv(t) = v_max * d̂ * exp(-(t-t_s)/τ) for t in [t_s, t_s + 3.0s]
        Acceleration: Δa(t) = (Δv(t) - Δv(t-Δt)) / Δt
        
        Args:
            df: Flight DataFrame
            t_s: Attack start time
            v_max: Maximum velocity transient magnitude
            direction: 2D unit vector for transient direction
            times: Array of time values
        
        Returns:
            Modified DataFrame
        """
        # Define transient interval
        t_end = t_s + config.ATTACK_DURATION
        transient_mask = (times >= t_s) & (times <= t_end)
        
        # Compute velocity transient
        delta_v_x = np.zeros_like(times)
        delta_v_y = np.zeros_like(times)
        
        t_relative = times - t_s
        decay = np.exp(-t_relative / config.ATTACK_TIME_CONSTANT)
        
        delta_v_x[transient_mask] = v_max * direction[0] * decay[transient_mask]
        delta_v_y[transient_mask] = v_max * direction[1] * decay[transient_mask]
        
        # Apply velocity transient
        df['velocity_x'] += delta_v_x
        df['velocity_y'] += delta_v_y
        
        # Compute acceleration transient using real Δt
        delta_a_x = np.zeros_like(times)
        delta_a_y = np.zeros_like(times)
        
        delta_t = df['delta_t'].values
        
        for i in range(1, len(times)):
            if transient_mask[i] and delta_t[i] > 0:
                delta_a_x[i] = (delta_v_x[i] - delta_v_x[i-1]) / delta_t[i]
                delta_a_y[i] = (delta_v_y[i] - delta_v_y[i-1]) / delta_t[i]
        
        # Apply acceleration transient
        df['linear_acceleration_x'] += delta_a_x
        df['linear_acceleration_y'] += delta_a_y
        
        return df
    
    def inject_attacks_to_dataset(self,
                                  df: pd.DataFrame,
                                  attack_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Inject attacks to multiple flights in dataset.
        
        Args:
            df: DataFrame with multiple flights
            attack_ratio: Ratio of flights to attack
        
        Returns:
            Tuple of (attacked_df, attack_info_df)
        """
        attacked_dfs = []
        attack_records = []
        
        flights = df['flight'].unique()
        
        for flight_id in flights:
            flight_df = df[df['flight'] == flight_id].copy()
            
            attacked_flight, attack_info = self.inject_attack_to_flight(
                flight_df, attack_prob=attack_ratio
            )
            
            attacked_dfs.append(attacked_flight)
            
            # Record attack info
            attack_record = {'flight': flight_id}
            attack_record.update(attack_info)
            attack_records.append(attack_record)
        
        # Combine all flights
        result_df = pd.concat(attacked_dfs, ignore_index=True)
        attack_info_df = pd.DataFrame(attack_records)
        
        n_attacked = attack_info_df['attacked'].sum()
        print(f"Injected attacks to {n_attacked}/{len(flights)} flights "
              f"({n_attacked/len(flights)*100:.1f}%)")
        
        return result_df, attack_info_df
