"""
Flight Reuse Injector (Strategy C)

Supports creating multiple attack versions of the same flight.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from multi_attack_injector import MultiAttackInjector
import config


class FlightReuseInjector(MultiAttackInjector):
    """
    Injector that creates multiple attack versions of each flight.
    
    Used for Strategy C: Train/Val sets use flight reuse for data expansion.
    """
    
    def inject_with_flight_reuse(self,
                                 df: pd.DataFrame,
                                 flight_attack_pairs: List[Tuple[int, str]],
                                 verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Inject attacks according to pre-defined flight-attack pairs.
        
        This method is used when we have explicit assignments like:
        - (flight_123, 'step')
        - (flight_123, 'drift_ramp')
        - (flight_123, 'none')
        etc.
        
        Args:
            df: DataFrame with all flights
            flight_attack_pairs: List of (flight_id, attack_type) tuples
            verbose: Print progress
        
        Returns:
            Tuple of (attacked_df, attack_info_df)
        """
        if verbose:
            print(f"\n[FLIGHT REUSE INJECTION]")
            print(f"Total flight-attack pairs: {len(flight_attack_pairs)}")
            
            # Count attack distribution
            from collections import Counter
            attack_counts = Counter([pair[1] for pair in flight_attack_pairs])
            print(f"\nAttack distribution:")
            for attack_type, count in sorted(attack_counts.items()):
                print(f"  {attack_type:20s}: {count:4d} samples")
        
        attacked_dfs = []
        attack_records = []
        
        # Process each flight-attack pair
        for flight_id, attack_type in flight_attack_pairs:
            flight_df = df[df['flight'] == flight_id].copy()
            
            if len(flight_df) == 0:
                if verbose:
                    print(f"Warning: Flight {flight_id} not found in data")
                continue
            
            if attack_type == 'none':
                # No attack - just copy the flight as-is
                attacked_dfs.append(flight_df)
                attack_records.append({
                    'flight': flight_id,
                    'attacked': False,
                    'attack_type': 'none',
                    'attack_start_time': np.nan,
                    'attack_params': {},
                    'consistency_mode': '',
                    'reason': 'normal_flight'
                })
            else:
                # Inject specified attack
                attack_params = config.ATTACK_PARAMS.get(attack_type, {})
                injection_type = config.ATTACK_TYPE_MAP.get(attack_type, attack_type)
                
                # For replay attacks, pass all_flights_df
                if injection_type == 'replay':
                    attack_params['all_flights_df'] = df
                
                # Inject with probability=1.0 (deterministic)
                attacked_flight, attack_info = self.inject_attack_to_flight(
                    flight_df,
                    attack_type=injection_type,
                    attack_prob=1.0,
                    **attack_params
                )
                
                # Add flight ID to attack_info (CRITICAL for labeling)
                attack_info['flight'] = flight_id
                
                # Override attack_type in info to match the requested type
                # (e.g., 'drift_ramp' instead of just 'drift')
                if attack_info['attacked']:
                    attack_info['attack_type'] = attack_type
                
                attacked_dfs.append(attacked_flight)
                attack_records.append(attack_info)
        
        # Combine all flights
        result_df = pd.concat(attacked_dfs, ignore_index=True)
        attack_info_df = pd.DataFrame(attack_records)
        
        if verbose:
            print(f"\nInjection complete:")
            print(f"  Total samples: {len(result_df)}")
            print(f"  Flights processed: {len(attack_records)}")
            actual_attacked = attack_info_df['attacked'].sum()
            print(f"  Actually attacked: {actual_attacked}/{len(attack_records)} "
                  f"({actual_attacked/len(attack_records)*100:.1f}%)")
        
        return result_df, attack_info_df
    
    
    def inject_stratified_attacks_to_test(self,
                                         df: pd.DataFrame,
                                         test_attack_allocation: dict,
                                         verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Inject attacks to test set according to stratified allocation.
        
        Args:
            df: DataFrame with all test flights
            test_attack_allocation: Dict mapping attack_type -> list of flight_ids
            verbose: Print progress
        
        Returns:
            Tuple of (attacked_df, attack_info_df)
        """
        if verbose:
            print(f"\n[STRATIFIED TEST INJECTION]")
            print(f"Test set flights: {len(df['flight'].unique())}")
            print(f"\nAttack allocation:")
            for attack_type, flights in test_attack_allocation.items():
                print(f"  {attack_type:20s}: {len(flights):3d} flights")
        
        attacked_dfs = []
        attack_records = []
        
        # Create flight -> attack mapping
        flight_to_attack = {}
        for attack_type, flight_ids in test_attack_allocation.items():
            for flight_id in flight_ids:
                flight_to_attack[flight_id] = attack_type
        
        # Process each flight
        all_flights = df['flight'].unique()
        for flight_id in all_flights:
            flight_df = df[df['flight'] == flight_id].copy()
            attack_type = flight_to_attack.get(flight_id, 'none')
            
            if attack_type == 'none':
                # No attack
                attacked_dfs.append(flight_df)
                attack_records.append({
                    'flight': flight_id,
                    'attacked': False,
                    'attack_type': 'none',
                    'attack_start_time': np.nan,
                    'attack_params': {},
                    'consistency_mode': '',
                    'reason': 'not_selected'
                })
            else:
                # Inject specified attack
                attack_params = config.ATTACK_PARAMS.get(attack_type, {})
                injection_type = config.ATTACK_TYPE_MAP.get(attack_type, attack_type)
                
                if injection_type == 'replay':
                    attack_params['all_flights_df'] = df
                
                attacked_flight, attack_info = self.inject_attack_to_flight(
                    flight_df,
                    attack_type=injection_type,
                    attack_prob=1.0,
                    **attack_params
                )
                
                # Add flight ID to attack_info (CRITICAL for labeling)
                attack_info['flight'] = flight_id
                
                # Override attack_type
                if attack_info['attacked']:
                    attack_info['attack_type'] = attack_type
                
                attacked_dfs.append(attacked_flight)
                attack_records.append(attack_info)
        
        result_df = pd.concat(attacked_dfs, ignore_index=True)
        attack_info_df = pd.DataFrame(attack_records)
        
        if verbose:
            print(f"\nTest injection complete:")
            print(f"  Total samples: {len(result_df)}")
            print(f"  Flights: {len(attack_records)}")
            print(f"  Attacked: {attack_info_df['attacked'].sum()}")
            print(f"\nActual attack distribution:")
            print(attack_info_df['attack_type'].value_counts())
        
        return result_df, attack_info_df
