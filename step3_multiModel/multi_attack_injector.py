"""
Multi-Attack GPS Spoofing Injection Module

Implements four types of GPS spoofing attacks:
A. Drift Spoofing (gradual offset accumulation)
B1. Delay (fixed time delay)
B2. Replay (segment replay with stitching)
C. Consistent Takeover (first-order tracking with kinematics)
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from scipy import interpolate
import config_step2 as config


class MultiAttackInjector:
    """
    Unified GPS spoofing attack injector supporting multiple attack types.
    
    Attack types:
    - drift: Gradual position drift (ramp or sigmoid profile)
    - delay: Fixed time delay (meaconing-like)
    - replay: Segment replay with hard/soft stitching
    - takeover: Consistent takeover with first-order dynamics
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
                                attack_type: str,
                                attack_prob: float = 1.0,
                                **attack_params) -> Tuple[pd.DataFrame, Dict]:
        """
        Inject GPS spoofing attack into a single flight.
        
        Args:
            flight_df: DataFrame for single flight (sorted by time)
            attack_type: One of ['drift', 'delay', 'replay', 'takeover']
            attack_prob: Probability of injecting attack (0.0 to 1.0)
            **attack_params: Attack-specific parameters
        
        Returns:
            Tuple of (modified_df, attack_info_dict)
        """
        df = flight_df.copy()
        
        # Ensure delta_t exists
        df = self._ensure_delta_t(df)
        
        # Initialize attack metadata columns
        df['attack_start_time'] = np.nan
        df['attack_type'] = 'none'
        
        # Decide whether to attack this flight
        if self.rng.random() >= attack_prob:
            attack_info = {
                'attacked': False,
                'attack_type': 'none',
                'attack_start_time': None,
                'attack_params': {},
                'consistency_mode': None,
                'reason': 'not_selected'
            }
            return df, attack_info
        
        # Get flight time bounds
        times = df['time'].values
        min_time = times.min()
        max_time = times.max()
        
        # Check if flight is long enough
        flight_duration = max_time - min_time
        min_duration = config.ATTACK_START_BUFFER + config.ATTACK_END_BUFFER
        
        if flight_duration < min_duration:
            attack_info = {
                'attacked': False,
                'attack_type': attack_type,
                'attack_start_time': None,
                'attack_params': {},
                'consistency_mode': None,
                'reason': 'flight_too_short'
            }
            return df, attack_info
        
        # Sample attack start time
        t_s_min = min_time + config.ATTACK_START_BUFFER
        t_s_max = max_time - config.ATTACK_END_BUFFER
        t_s = self.rng.uniform(t_s_min, t_s_max)
        
        # Route to appropriate attack method
        if attack_type == 'step':
            df, attack_info = self._inject_step_attack(df, t_s, **attack_params)
        elif attack_type == 'drift':
            df, attack_info = self._inject_drift_attack(df, t_s, **attack_params)
        elif attack_type == 'delay':
            df, attack_info = self._inject_delay_attack(df, t_s, **attack_params)
        elif attack_type == 'replay':
            df, attack_info = self._inject_replay_attack(df, t_s, **attack_params)
        elif attack_type == 'takeover':
            df, attack_info = self._inject_takeover_attack(df, t_s, **attack_params)
        else:
            raise ValueError(f"Unknown attack_type: {attack_type}")
        
        # Mark attack metadata in DataFrame
        if attack_info['attacked']:
            df['attack_start_time'] = attack_info['attack_start_time']
            df['attack_type'] = attack_info['attack_type']
        
        return df, attack_info
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _ensure_delta_t(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure delta_t column exists and is properly computed."""
        if 'delta_t' not in df.columns:
            times = df['time'].values
            delta_t = np.zeros_like(times)
            delta_t[1:] = times[1:] - times[:-1]
            
            # Set first delta_t to median of rest
            valid_deltas = delta_t[1:]
            if len(valid_deltas) > 0:
                delta_t[0] = np.median(valid_deltas)
            else:
                delta_t[0] = 0.0
            
            df['delta_t'] = delta_t
        
        return df
    
    def _sample_direction(self, 
                         df: pd.DataFrame,
                         t_s: float,
                         direction_mode: str) -> np.ndarray:
        """
        Sample 2D horizontal direction vector.
        
        Args:
            df: Flight DataFrame
            t_s: Attack start time
            direction_mode: One of DIRECTION_MODES
        
        Returns:
            Normalized 2D direction vector [dx, dy]
        """
        if direction_mode == 'random_xy':
            angle = self.rng.uniform(0, 2 * np.pi)
            return np.array([np.cos(angle), np.sin(angle)])
        
        elif direction_mode == 'fixed_east':
            return np.array([1.0, 0.0])
        
        elif direction_mode == 'fixed_north':
            return np.array([0.0, 1.0])
        
        elif direction_mode == 'along_track_xy':
            return self._compute_along_track_direction(df, t_s)
        
        elif direction_mode == 'cross_track_xy':
            along_track = self._compute_along_track_direction(df, t_s)
            # Rotate 90 degrees: (x, y) -> (-y, x)
            return np.array([-along_track[1], along_track[0]])
        
        else:
            raise ValueError(f"Unknown direction_mode: {direction_mode}")
    
    def _compute_along_track_direction(self,
                                       df: pd.DataFrame,
                                       t_s: float) -> np.ndarray:
        """Compute along-track direction from velocity at t_s."""
        times = df['time'].values
        mask_after_ts = times >= t_s
        
        if not mask_after_ts.any():
            # Fallback to random
            angle = self.rng.uniform(0, 2 * np.pi)
            return np.array([np.cos(angle), np.sin(angle)])
        
        # Get first sample at or after t_s
        idx = np.where(mask_after_ts)[0][0]
        
        vx = df['velocity_x'].iloc[idx]
        vy = df['velocity_y'].iloc[idx]
        v_norm = np.sqrt(vx**2 + vy**2)
        
        # Try velocity first
        if v_norm >= config.VELOCITY_EPSILON:
            return np.array([vx / v_norm, vy / v_norm])
        
        # Try position difference
        if idx > 0:
            px_curr = df['position_x'].iloc[idx]
            py_curr = df['position_y'].iloc[idx]
            px_prev = df['position_x'].iloc[idx - 1]
            py_prev = df['position_y'].iloc[idx - 1]
            
            dx = px_curr - px_prev
            dy = py_curr - py_prev
            d_norm = np.sqrt(dx**2 + dy**2)
            
            if d_norm >= config.VELOCITY_EPSILON:
                return np.array([dx / d_norm, dy / d_norm])
        
        # Fallback to random
        angle = self.rng.uniform(0, 2 * np.pi)
        return np.array([np.cos(angle), np.sin(angle)])
    
    def _apply_consistency(self,
                          df: pd.DataFrame,
                          consistency_mode: str,
                          attack_mask: np.ndarray = None) -> pd.DataFrame:
        """
        Apply consistency recomputation for velocity and/or acceleration.
        
        Args:
            df: DataFrame
            consistency_mode: One of ['pos_only', 'pos_vel', 'pos_vel_acc']
            attack_mask: Optional mask for which samples to recompute
        
        Returns:
            Modified DataFrame
        """
        if consistency_mode == 'pos_only':
            return df
        
        delta_t = df['delta_t'].values
        
        # Recompute velocity
        if consistency_mode in ['pos_vel', 'pos_vel_acc']:
            for coord in ['x', 'y', 'z']:
                pos_col = f'position_{coord}'
                vel_col = f'velocity_{coord}'
                
                if pos_col in df.columns and vel_col in df.columns:
                    positions = df[pos_col].values
                    velocities = df[vel_col].values.copy()
                    
                    for i in range(1, len(positions)):
                        if delta_t[i] > 0:
                            if attack_mask is None or attack_mask[i]:
                                velocities[i] = (positions[i] - positions[i-1]) / delta_t[i]
                    
                    df[vel_col] = velocities
        
        # Recompute acceleration
        if consistency_mode == 'pos_vel_acc':
            for coord in ['x', 'y', 'z']:
                vel_col = f'velocity_{coord}'
                acc_col = f'linear_acceleration_{coord}'
                
                if vel_col in df.columns and acc_col in df.columns:
                    velocities = df[vel_col].values
                    accelerations = df[acc_col].values.copy()
                    
                    for i in range(1, len(velocities)):
                        if delta_t[i] > 0:
                            if attack_mask is None or attack_mask[i]:
                                accelerations[i] = (velocities[i] - velocities[i-1]) / delta_t[i]
                    
                    df[acc_col] = accelerations
        
        return df
    
    # ========================================================================
    # ATTACK STEP: BASELINE STEP ATTACK (from step1)
    # ========================================================================
    
    def _inject_step_attack(self,
                           df: pd.DataFrame,
                           t_s: float,
                           magnitude: float = None,
                           direction_mode: str = None,
                           v_max: float = None,
                           consistency_mode: str = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Inject baseline step attack (from step1_benchmark).
        
        Attack characteristics:
        - Step offset in position (horizontal plane)
        - Optionally apply velocity/acceleration transients (when pos_only)
        - With pos_vel_acc: recompute velocity and acceleration for consistency
        """
        # Sample parameters if not provided
        if magnitude is None:
            magnitude = self.rng.choice(config.ATTACK_MAGNITUDES)
        if direction_mode is None:
            direction_mode = 'random_xy'  # step1 always uses random direction
        if consistency_mode is None:
            consistency_mode = self.rng.choice(config.CONSISTENCY_MODES)
        
        # Sample direction
        direction = self._sample_direction(df, t_s, direction_mode)
        
        times = df['time'].values
        
        # Apply position offset
        attack_mask = times >= t_s
        position_offset = magnitude * direction
        
        df.loc[attack_mask, 'position_x'] += position_offset[0]
        df.loc[attack_mask, 'position_y'] += position_offset[1]
        
        # For pos_only mode, apply step1's velocity/acceleration transients
        # For pos_vel/pos_vel_acc, recompute from positions
        if consistency_mode == 'pos_only':
            if v_max is None:
                v_max = self.rng.uniform(config.STEP1_VELOCITY_TRANSIENT_MIN,
                                        config.STEP1_VELOCITY_TRANSIENT_MAX)
            df = self._apply_step_velocity_acceleration_transient(
                df, t_s, v_max, direction, times
            )
            v_max_value = v_max
        else:
            # With consistency recomputation, no manual transients
            df = self._apply_consistency(df, consistency_mode, attack_mask)
            v_max_value = None
        
        attack_params = {
            'magnitude': magnitude,
            'direction_mode': direction_mode,
            'direction_vector': direction.tolist(),
            'attack_duration': config.STEP1_ATTACK_DURATION,
            'time_constant': config.STEP1_ATTACK_TIME_CONSTANT
        }
        if v_max_value is not None:
            attack_params['velocity_transient_max'] = v_max_value
        
        attack_info = {
            'attacked': True,
            'attack_type': 'step',
            'attack_start_time': t_s,
            'attack_params': attack_params,
            'consistency_mode': consistency_mode
        }
        
        return df, attack_info
    
    def _apply_step_velocity_acceleration_transient(self,
                                                   df: pd.DataFrame,
                                                   t_s: float,
                                                   v_max: float,
                                                   direction: np.ndarray,
                                                   times: np.ndarray) -> pd.DataFrame:
        """
        Apply exponentially decaying velocity transient and derived acceleration.
        
        This matches the step1_benchmark implementation exactly.
        
        Velocity: Δv(t) = v_max * d̂ * exp(-(t-t_s)/τ) for t in [t_s, t_s + duration]
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
        # Define transient interval (from step1 config)
        t_end = t_s + config.STEP1_ATTACK_DURATION
        transient_mask = (times >= t_s) & (times <= t_end)
        
        # Compute velocity transient
        delta_v_x = np.zeros_like(times)
        delta_v_y = np.zeros_like(times)
        
        t_relative = times - t_s
        decay = np.exp(-t_relative / config.STEP1_ATTACK_TIME_CONSTANT)
        
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
    
    # ========================================================================
    # ATTACK A: DRIFT SPOOFING
    # ========================================================================
    
    def _inject_drift_attack(self,
                            df: pd.DataFrame,
                            t_s: float,
                            profile: str = None,
                            T_drift: float = None,
                            M: float = None,
                            direction_mode: str = None,
                            consistency_mode: str = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Inject drift spoofing attack.
        
        Position offset grows from 0 to M over T_drift, then stays at M.
        """
        # Sample parameters if not provided
        if profile is None:
            profile = self.rng.choice(config.DRIFT_PROFILES)
        if T_drift is None:
            T_drift = self.rng.choice(config.DRIFT_DURATIONS)
        if M is None:
            M = self.rng.choice(config.ATTACK_MAGNITUDES)
        if direction_mode is None:
            direction_mode = self.rng.choice(config.DIRECTION_MODES)
        if consistency_mode is None:
            consistency_mode = self.rng.choice(config.CONSISTENCY_MODES)
        
        times = df['time'].values
        t_max = times.max()
        
        # Clip T_drift if needed
        T_drift_clipped = False
        if T_drift > (t_max - t_s):
            T_drift = t_max - t_s
            T_drift_clipped = True
        
        # Sample direction
        direction = self._sample_direction(df, t_s, direction_mode)
        
        # Compute growth function g(t)
        g = np.zeros_like(times)
        
        for i, t in enumerate(times):
            if t < t_s:
                g[i] = 0.0
            elif t <= t_s + T_drift:
                if profile == 'ramp':
                    g[i] = (t - t_s) / T_drift
                elif profile == 'sigmoid':
                    x = (t - t_s) / T_drift
                    # Standard logistic
                    sigmoid_raw = 1.0 / (1.0 + np.exp(-config.SIGMOID_K * (x - 0.5)))
                    # Normalize to [0, 1]
                    s_0 = 1.0 / (1.0 + np.exp(-config.SIGMOID_K * (-0.5)))
                    s_1 = 1.0 / (1.0 + np.exp(-config.SIGMOID_K * 0.5))
                    g[i] = (sigmoid_raw - s_0) / (s_1 - s_0)
            else:
                g[i] = 1.0
        
        # Apply position offset
        offset_x = M * g * direction[0]
        offset_y = M * g * direction[1]
        
        df['position_x'] += offset_x
        df['position_y'] += offset_y
        
        # Apply consistency
        attack_mask = times >= t_s
        df = self._apply_consistency(df, consistency_mode, attack_mask)
        
        attack_info = {
            'attacked': True,
            'attack_type': 'drift',
            'attack_start_time': t_s,
            'attack_params': {
                'profile': profile,
                'T_drift': T_drift,
                'T_drift_clipped': T_drift_clipped,
                'M': M,
                'direction_mode': direction_mode,
                'direction_vector': direction.tolist()
            },
            'consistency_mode': consistency_mode
        }
        
        return df, attack_info
    
    # ========================================================================
    # ATTACK B1: DELAY
    # ========================================================================
    
    def _inject_delay_attack(self,
                            df: pd.DataFrame,
                            t_s: float,
                            delay_seconds: float = None,
                            consistency_mode: str = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Inject fixed delay attack.
        
        For t >= t_s, position is taken from t - delay_seconds.
        """
        # Sample parameters if not provided
        if delay_seconds is None:
            delay_seconds = self.rng.choice(config.DELAY_SECONDS)
        if consistency_mode is None:
            consistency_mode = self.rng.choice(config.CONSISTENCY_MODES)
        
        times = df['time'].values
        t_min = times.min()
        
        # Build interpolators for original positions
        interp_x = interpolate.interp1d(times, df['position_x'].values,
                                       kind='linear', bounds_error=False,
                                       fill_value=(df['position_x'].iloc[0], df['position_x'].iloc[-1]))
        interp_y = interpolate.interp1d(times, df['position_y'].values,
                                       kind='linear', bounds_error=False,
                                       fill_value=(df['position_y'].iloc[0], df['position_y'].iloc[-1]))
        interp_z = interpolate.interp1d(times, df['position_z'].values,
                                       kind='linear', bounds_error=False,
                                       fill_value=(df['position_z'].iloc[0], df['position_z'].iloc[-1]))
        
        # Apply delay
        attack_mask = times >= t_s
        source_times = times[attack_mask] - delay_seconds
        
        # Clamp source times
        source_time_clamped_count = np.sum(source_times < t_min)
        source_times = np.maximum(source_times, t_min)
        
        # Replace positions
        df.loc[attack_mask, 'position_x'] = interp_x(source_times)
        df.loc[attack_mask, 'position_y'] = interp_y(source_times)
        df.loc[attack_mask, 'position_z'] = interp_z(source_times)
        
        # Apply consistency
        df = self._apply_consistency(df, consistency_mode, attack_mask)
        
        attack_info = {
            'attacked': True,
            'attack_type': 'delay',
            'attack_start_time': t_s,
            'attack_params': {
                'delay_seconds': delay_seconds,
                'source_time_clamped_count': int(source_time_clamped_count)
            },
            'consistency_mode': consistency_mode
        }
        
        return df, attack_info
    
    # ========================================================================
    # ATTACK B2: REPLAY
    # ========================================================================
    
    def _inject_replay_attack(self,
                             df: pd.DataFrame,
                             t_s: float,
                             segment_duration: float = None,
                             donor_source: str = None,
                             stitching: str = None,
                             transition_seconds: float = None,
                             consistency_mode: str = None,
                             all_flights_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Inject replay attack with segment stitching.
        
        Note: For 'same_route_other_flight', requires all_flights_df.
        """
        # Sample parameters if not provided
        if segment_duration is None:
            segment_duration = self.rng.choice(config.REPLAY_SEGMENT_DURATIONS)
        if donor_source is None:
            donor_source = self.rng.choice(config.REPLAY_DONOR_SOURCES)
        if stitching is None:
            stitching = self.rng.choice(config.REPLAY_STITCHING_MODES)
        if transition_seconds is None:
            transition_seconds = config.REPLAY_TRANSITION_SECONDS
        if consistency_mode is None:
            consistency_mode = self.rng.choice(config.CONSISTENCY_MODES)
        
        times = df['time'].values
        t_min = times.min()
        t_max = times.max()
        
        # Compute attack segment end
        t_e = t_s + segment_duration
        if t_e > t_max - config.ATTACK_END_BUFFER:
            t_e = t_max - config.ATTACK_END_BUFFER
            segment_duration = t_e - t_s
        
        # Find donor segment
        if donor_source == 'same_flight_earlier':
            result = self._find_donor_segment_same_flight(
                df, t_s, segment_duration, config.REPLAY_GAP_SECONDS
            )
        elif donor_source == 'same_route_other_flight':
            if all_flights_df is None:
                attack_info = {
                    'attacked': False,
                    'attack_type': 'replay',
                    'attack_start_time': None,
                    'attack_params': {},
                    'consistency_mode': None,
                    'reason': 'no_all_flights_df_provided'
                }
                return df, attack_info
            
            result = self._find_donor_segment_other_flight(
                df, all_flights_df, segment_duration
            )
        else:
            raise ValueError(f"Unknown donor_source: {donor_source}")
        
        if not result['found']:
            attack_info = {
                'attacked': False,
                'attack_type': 'replay',
                'attack_start_time': None,
                'attack_params': {},
                'consistency_mode': None,
                'reason': result.get('reason', 'no_valid_donor_segment')
            }
            return df, attack_info
        
        donor_df = result['donor_df']
        donor_t_a = result['t_a']
        donor_t_b = result['t_b']
        donor_flight_id = result.get('donor_flight_id', None)
        
        # Apply replay with stitching
        df = self._apply_replay_segment(
            df, t_s, t_e, donor_df, donor_t_a, donor_t_b,
            stitching, transition_seconds
        )
        
        # Apply consistency
        attack_mask = (times >= t_s) & (times <= t_e)
        df = self._apply_consistency(df, consistency_mode, attack_mask)
        
        attack_info = {
            'attacked': True,
            'attack_type': 'replay',
            'attack_start_time': t_s,
            'attack_params': {
                'segment_duration': segment_duration,
                'donor_source': donor_source,
                'donor_flight_id': donor_flight_id,
                'donor_t_a': donor_t_a,
                'donor_t_b': donor_t_b,
                'stitching': stitching,
                'transition_seconds': transition_seconds
            },
            'consistency_mode': consistency_mode
        }
        
        return df, attack_info
    
    def _find_donor_segment_same_flight(self,
                                       df: pd.DataFrame,
                                       t_s: float,
                                       duration: float,
                                       gap: float) -> Dict:
        """Find donor segment from earlier in the same flight."""
        times = df['time'].values
        t_min = times.min()
        
        # Donor must end before t_s - gap
        t_b_max = t_s - gap
        t_a_min = t_min + config.ATTACK_START_BUFFER
        
        # Donor segment duration
        t_a_max = t_b_max - duration
        
        if t_a_max < t_a_min:
            return {'found': False, 'reason': 'no_valid_donor_segment_same_flight'}
        
        # Sample donor start
        t_a = self.rng.uniform(t_a_min, t_a_max)
        t_b = t_a + duration
        
        return {
            'found': True,
            'donor_df': df,
            't_a': t_a,
            't_b': t_b
        }
    
    def _find_donor_segment_other_flight(self,
                                        df: pd.DataFrame,
                                        all_flights_df: pd.DataFrame,
                                        duration: float) -> Dict:
        """Find donor segment from another flight on the same route."""
        current_flight_id = df['flight'].iloc[0]
        
        if 'route' not in df.columns:
            return {'found': False, 'reason': 'no_route_column'}
        
        current_route = df['route'].iloc[0]
        
        # Find other flights on same route
        same_route = all_flights_df[all_flights_df['route'] == current_route]
        other_flights = same_route[same_route['flight'] != current_flight_id]['flight'].unique()
        
        if len(other_flights) == 0:
            return {'found': False, 'reason': 'no_other_flights_same_route'}
        
        # Shuffle and try flights
        self.rng.shuffle(other_flights)
        
        for donor_flight_id in other_flights:
            donor_df = all_flights_df[all_flights_df['flight'] == donor_flight_id].copy()
            donor_times = donor_df['time'].values
            donor_duration = donor_times.max() - donor_times.min()
            
            if donor_duration < duration + config.ATTACK_START_BUFFER + config.ATTACK_END_BUFFER:
                continue
            
            # Sample donor segment
            t_a_min = donor_times.min() + config.ATTACK_START_BUFFER
            t_a_max = donor_times.max() - config.ATTACK_END_BUFFER - duration
            
            if t_a_max < t_a_min:
                continue
            
            t_a = self.rng.uniform(t_a_min, t_a_max)
            t_b = t_a + duration
            
            return {
                'found': True,
                'donor_df': donor_df,
                'donor_flight_id': donor_flight_id,
                't_a': t_a,
                't_b': t_b
            }
        
        return {'found': False, 'reason': 'no_suitable_donor_flight'}
    
    def _apply_replay_segment(self,
                             df: pd.DataFrame,
                             t_s: float,
                             t_e: float,
                             donor_df: pd.DataFrame,
                             donor_t_a: float,
                             donor_t_b: float,
                             stitching: str,
                             transition_seconds: float) -> pd.DataFrame:
        """Apply replay segment with stitching."""
        times = df['time'].values
        
        # Build donor interpolators
        donor_times = donor_df['time'].values
        interp_donor_x = interpolate.interp1d(donor_times, donor_df['position_x'].values,
                                             kind='linear', bounds_error=False,
                                             fill_value='extrapolate')
        interp_donor_y = interpolate.interp1d(donor_times, donor_df['position_y'].values,
                                             kind='linear', bounds_error=False,
                                             fill_value='extrapolate')
        interp_donor_z = interpolate.interp1d(donor_times, donor_df['position_z'].values,
                                             kind='linear', bounds_error=False,
                                             fill_value='extrapolate')
        
        # Original positions (for soft stitching)
        orig_x = df['position_x'].values.copy()
        orig_y = df['position_y'].values.copy()
        orig_z = df['position_z'].values.copy()
        
        # Attack segment mask
        attack_mask = (times >= t_s) & (times <= t_e)
        attack_times = times[attack_mask]
        
        # Map to donor times
        donor_mapped_times = donor_t_a + (attack_times - t_s)
        
        # Get donor positions
        donor_x = interp_donor_x(donor_mapped_times)
        donor_y = interp_donor_y(donor_mapped_times)
        donor_z = interp_donor_z(donor_mapped_times)
        
        if stitching == 'hard':
            # Direct replacement
            df.loc[attack_mask, 'position_x'] = donor_x
            df.loc[attack_mask, 'position_y'] = donor_y
            df.loc[attack_mask, 'position_z'] = donor_z
        
        elif stitching == 'soft':
            # Soft blending at boundaries
            new_x = orig_x.copy()
            new_y = orig_y.copy()
            new_z = orig_z.copy()
            
            for i, t in enumerate(attack_times):
                if t <= t_s + transition_seconds:
                    # Start transition
                    w = (t - t_s) / transition_seconds
                elif t >= t_e - transition_seconds:
                    # End transition
                    w = (t_e - t) / transition_seconds
                    w = 1.0 - w
                else:
                    # Core segment
                    w = 1.0
                
                idx = np.where(times == t)[0][0]
                new_x[idx] = (1 - w) * orig_x[idx] + w * donor_x[i]
                new_y[idx] = (1 - w) * orig_y[idx] + w * donor_y[i]
                new_z[idx] = (1 - w) * orig_z[idx] + w * donor_z[i]
            
            df.loc[attack_mask, 'position_x'] = new_x[attack_mask]
            df.loc[attack_mask, 'position_y'] = new_y[attack_mask]
            df.loc[attack_mask, 'position_z'] = new_z[attack_mask]
        
        return df
    
    # ========================================================================
    # ATTACK C: CONSISTENT TAKEOVER
    # ========================================================================
    
    def _inject_takeover_attack(self,
                               df: pd.DataFrame,
                               t_s: float,
                               offset_profile: str = None,
                               M: float = None,
                               direction_mode: str = None,
                               takeover_alpha: float = None,
                               takeover_tau: float = None,
                               T_takeover: float = None,
                               consistency_mode: str = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Inject consistent takeover attack with first-order dynamics.
        
        Position follows: p'[i] = p'[i-1] + alpha * (p_ref[i] - p'[i-1])
        """
        # Sample parameters if not provided
        if offset_profile is None:
            offset_profile = self.rng.choice(config.TAKEOVER_OFFSET_PROFILES)
        if M is None:
            M = self.rng.choice(config.ATTACK_MAGNITUDES)
        if direction_mode is None:
            direction_mode = self.rng.choice(config.DIRECTION_MODES)
        
        # Use either alpha or tau (prefer tau if both provided)
        use_tau = takeover_tau is not None
        if not use_tau and takeover_alpha is None:
            # Sample one
            if self.rng.random() < 0.5:
                takeover_alpha = self.rng.choice(config.TAKEOVER_ALPHAS)
            else:
                takeover_tau = self.rng.choice(config.TAKEOVER_TAUS)
                use_tau = True
        
        # For ramp/sigmoid, need T_takeover
        if offset_profile in ['ramp', 'sigmoid'] and T_takeover is None:
            T_takeover = self.rng.choice(config.TAKEOVER_DURATIONS)
        
        # Consistency mode must be pos_vel or pos_vel_acc
        if consistency_mode is None:
            consistency_mode = self.rng.choice(['pos_vel', 'pos_vel_acc'])
        elif consistency_mode == 'pos_only':
            consistency_mode = 'pos_vel'  # Force upgrade
        
        times = df['time'].values
        delta_t = df['delta_t'].values
        t_max = times.max()
        
        # Clip T_takeover if needed
        T_takeover_clipped = False
        if T_takeover is not None and T_takeover > (t_max - t_s):
            T_takeover = t_max - t_s
            T_takeover_clipped = True
        
        # Validate alpha/tau
        alpha_clipped = False
        if use_tau:
            if takeover_tau <= 0:
                attack_info = {
                    'attacked': False,
                    'attack_type': 'takeover',
                    'attack_start_time': None,
                    'attack_params': {},
                    'consistency_mode': None,
                    'reason': 'invalid_tau_takeover'
                }
                return df, attack_info
        else:
            if takeover_alpha <= 0 or takeover_alpha > 1:
                takeover_alpha = np.clip(takeover_alpha, 0.01, 1.0)
                alpha_clipped = True
        
        # Sample direction
        direction = self._sample_direction(df, t_s, direction_mode)
        
        # Build reference trajectory p_ref
        p_ref_x, p_ref_y = self._build_takeover_reference(
            df, t_s, offset_profile, M, direction, T_takeover
        )
        
        # Apply first-order tracking
        p_new_x = df['position_x'].values.copy()
        p_new_y = df['position_y'].values.copy()
        
        invalid_delta_t_count = 0
        
        for i, t in enumerate(times):
            if t < t_s:
                continue
            
            # Compute alpha for this step
            if use_tau:
                if delta_t[i] > 0:
                    alpha_i = 1.0 - np.exp(-delta_t[i] / takeover_tau)
                else:
                    alpha_i = 0.0
                    invalid_delta_t_count += 1
            else:
                alpha_i = takeover_alpha
            
            # First-order update
            p_new_x[i] = p_new_x[i-1] + alpha_i * (p_ref_x[i] - p_new_x[i-1])
            p_new_y[i] = p_new_y[i-1] + alpha_i * (p_ref_y[i] - p_new_y[i-1])
        
        # Replace positions
        df['position_x'] = p_new_x
        df['position_y'] = p_new_y
        
        # Apply consistency (full recompute)
        df = self._apply_consistency(df, consistency_mode)
        
        attack_params = {
            'offset_profile': offset_profile,
            'M': M,
            'direction_mode': direction_mode,
            'direction_vector': direction.tolist()
        }
        
        if use_tau:
            attack_params['tau_takeover'] = takeover_tau
        else:
            attack_params['alpha'] = takeover_alpha
            attack_params['alpha_clipped'] = alpha_clipped
        
        if T_takeover is not None:
            attack_params['T_takeover'] = T_takeover
            attack_params['T_takeover_clipped'] = T_takeover_clipped
        
        if invalid_delta_t_count > 0:
            attack_params['invalid_delta_t_count'] = invalid_delta_t_count
        
        attack_info = {
            'attacked': True,
            'attack_type': 'takeover',
            'attack_start_time': t_s,
            'attack_params': attack_params,
            'consistency_mode': consistency_mode
        }
        
        return df, attack_info
    
    def _build_takeover_reference(self,
                                 df: pd.DataFrame,
                                 t_s: float,
                                 offset_profile: str,
                                 M: float,
                                 direction: np.ndarray,
                                 T_takeover: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Build reference trajectory for takeover attack."""
        times = df['time'].values
        orig_x = df['position_x'].values
        orig_y = df['position_y'].values
        
        g = np.zeros_like(times)
        
        for i, t in enumerate(times):
            if t < t_s:
                g[i] = 0.0
            elif offset_profile == 'step':
                g[i] = 1.0
            elif offset_profile == 'ramp':
                if t <= t_s + T_takeover:
                    g[i] = (t - t_s) / T_takeover
                else:
                    g[i] = 1.0
            elif offset_profile == 'sigmoid':
                if t <= t_s + T_takeover:
                    x = (t - t_s) / T_takeover
                    sigmoid_raw = 1.0 / (1.0 + np.exp(-config.SIGMOID_K * (x - 0.5)))
                    s_0 = 1.0 / (1.0 + np.exp(-config.SIGMOID_K * (-0.5)))
                    s_1 = 1.0 / (1.0 + np.exp(-config.SIGMOID_K * 0.5))
                    g[i] = (sigmoid_raw - s_0) / (s_1 - s_0)
                else:
                    g[i] = 1.0
        
        p_ref_x = orig_x + M * g * direction[0]
        p_ref_y = orig_y + M * g * direction[1]
        
        return p_ref_x, p_ref_y
    
    # ========================================================================
    # DATASET-LEVEL INJECTION
    # ========================================================================
    
    def inject_attacks_to_dataset(self,
                                  df: pd.DataFrame,
                                  attack_type: str,
                                  attack_ratio: float,
                                  **attack_params) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Inject attacks to multiple flights in dataset.
        
        Args:
            df: DataFrame with multiple flights
            attack_type: One of ['drift', 'delay', 'replay', 'takeover']
            attack_ratio: Ratio of flights to attack
            **attack_params: Attack-specific parameters
        
        Returns:
            Tuple of (attacked_df, attack_info_df)
        """
        attacked_dfs = []
        attack_records = []
        
        flights = df['flight'].unique()
        
        for flight_id in flights:
            flight_df = df[df['flight'] == flight_id].copy()
            
            # For replay with other_flight donor, pass all_flights_df
            if attack_type == 'replay':
                attack_params['all_flights_df'] = df
            
            attacked_flight, attack_info = self.inject_attack_to_flight(
                flight_df, attack_type, attack_prob=attack_ratio, **attack_params
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
        print(f"[{attack_type.upper()}] Injected attacks to {n_attacked}/{len(flights)} flights "
              f"({n_attacked/len(flights)*100:.1f}%)")
        
        return result_df, attack_info_df
