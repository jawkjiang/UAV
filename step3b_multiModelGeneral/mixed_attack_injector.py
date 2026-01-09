"""
Mixed Attack Injector for Multi-Model Training

Extends the base MultiAttackInjector to support injecting mixed attack types
across different flights in a single dataset.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from multi_attack_injector import MultiAttackInjector
import config


class MixedAttackInjector(MultiAttackInjector):
    """
    Injector that supports mixed attack types across flights.
    
    Strategy: Each flight gets at most one attack type, but different flights
    can have different attack types in the same dataset.
    """
    
    def inject_mixed_attacks_with_guarantee(self,
                                            df: pd.DataFrame,
                                            attack_ratio: float,
                                            attack_types: List[str],
                                            min_per_attack: Optional[int] = None,
                                            **common_params) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Inject mixed attack types with GUARANTEED coverage of all attack types.
        
        This method ensures that ALL attack types are represented in the dataset,
        preventing the issue where some attacks are missing in the test set.
        
        Strategy:
        1. Reserve min_per_attack flights for EACH attack type (round-robin)
        2. Distribute remaining attacked flights evenly across attack types
        3. Remaining flights stay clean (no attack)
        
        Args:
            df: DataFrame with multiple flights
            attack_ratio: Ratio of flights to attack (0.0 to 1.0)
            attack_types: List of attack types to use
            min_per_attack: Minimum flights per attack type (default: 1)
            **common_params: Common parameters for all attacks
        
        Returns:
            Tuple of (attacked_df, attack_info_df)
        
        Example:
            attack_types = ['step', 'drift_ramp', 'drift_sigmoid', 'delay', 'takeover_step', 'takeover_ramp']
            attack_ratio = 0.5, total flights = 20
            min_per_attack = 2
            
            Result:
            - 6 types × 2 flights = 12 flights (guaranteed coverage)
            - Remaining 10 × 0.5 - 12 = -2, so 0 additional
            - Total attacked: 12 flights (60%)
            - Each attack type: 2 flights minimum
        """
        flights = df['flight'].unique()
        n_flights = len(flights)
        n_to_attack = int(n_flights * attack_ratio)
        
        if min_per_attack is None:
            min_per_attack = max(1, n_to_attack // len(attack_types))
        
        print(f"\n[GUARANTEED MIXED ATTACK INJECTION]")
        print(f"Total flights: {n_flights}")
        print(f"Target attack ratio: {attack_ratio*100:.1f}%")
        print(f"Flights to attack: {n_to_attack}")
        print(f"Attack types: {len(attack_types)}")
        print(f"Min per attack: {min_per_attack}")
        
        # Check if we have enough flights
        min_required = len(attack_types) * min_per_attack
        if n_to_attack < min_required:
            print(f"\nWARNING: Not enough flights to attack!")
            print(f"  Required: {min_required} (for {min_per_attack} per attack)")
            print(f"  Available: {n_to_attack}")
            print(f"  Adjusting min_per_attack to {n_to_attack // len(attack_types)}")
            min_per_attack = max(1, n_to_attack // len(attack_types))
        
        # Shuffle flights
        shuffled_flights = self.rng.permutation(flights)
        
        # Build flight-to-attack mapping
        flight_attack_map = {}
        attack_type_counts = {at: 0 for at in attack_types}
        
        # Phase 1: Guarantee minimum coverage for each attack type
        flight_idx = 0
        for attack_type in attack_types:
            for _ in range(min_per_attack):
                if flight_idx >= len(shuffled_flights):
                    break
                flight_id = shuffled_flights[flight_idx]
                flight_attack_map[flight_id] = attack_type
                attack_type_counts[attack_type] += 1
                flight_idx += 1
        
        # Phase 2: Distribute remaining attacks evenly
        remaining_attacks = n_to_attack - flight_idx
        if remaining_attacks > 0:
            for i in range(remaining_attacks):
                if flight_idx >= len(shuffled_flights):
                    break
                attack_type = attack_types[i % len(attack_types)]
                flight_id = shuffled_flights[flight_idx]
                flight_attack_map[flight_id] = attack_type
                attack_type_counts[attack_type] += 1
                flight_idx += 1
        
        # Print planned distribution
        print(f"\nPlanned attack distribution (GUARANTEED):")
        for attack_type, count in attack_type_counts.items():
            print(f"  {attack_type:20s}: {count:3d} flights ({count/n_flights*100:5.2f}%)")
        
        # Inject attacks
        attacked_dfs = []
        attack_records = []
        
        for flight_id in flights:
            flight_df = df[df['flight'] == flight_id].copy()
            
            if flight_id in flight_attack_map:
                attack_type = flight_attack_map[flight_id]
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
                
                if attack_info['attacked']:
                    attack_info['attack_type'] = attack_type
            else:
                attacked_flight = flight_df
                attacked_flight['attack_start_time'] = np.nan
                attacked_flight['attack_type'] = 'none'
                
                attack_info = {
                    'attacked': False,
                    'attack_type': 'none',
                    'attack_start_time': None,
                    'attack_params': {},
                    'consistency_mode': None,
                    'reason': 'not_selected'
                }
            
            attacked_dfs.append(attacked_flight)
            attack_record = {'flight': flight_id}
            attack_record.update(attack_info)
            attack_records.append(attack_record)
        
        result_df = pd.concat(attacked_dfs, ignore_index=True)
        attack_info_df = pd.DataFrame(attack_records)
        
        # Verify coverage
        print(f"\nActual injection results:")
        total_attacked = attack_info_df['attacked'].sum()
        print(f"Total attacked: {total_attacked}/{n_flights} ({total_attacked/n_flights*100:.1f}%)")
        
        missing_attacks = []
        for attack_type in attack_types:
            count = (attack_info_df['attack_type'] == attack_type).sum()
            print(f"  {attack_type:20s}: {count:3d} flights ({count/n_flights*100:5.2f}%)")
            if count == 0:
                missing_attacks.append(attack_type)
        
        if missing_attacks:
            print(f"\n⚠️  WARNING: Missing attack types: {missing_attacks}")
        else:
            print(f"\n✓ All {len(attack_types)} attack types covered!")
        
        return result_df, attack_info_df
    
    def inject_mixed_attacks_to_dataset(self,
                                       df: pd.DataFrame,
                                       attack_ratio: float,
                                       attack_types: List[str],
                                       **common_params) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Inject mixed attack types to multiple flights in dataset.
        
        Each flight is randomly assigned one attack type from the list.
        Attack types are distributed evenly across attacked flights.
        
        Args:
            df: DataFrame with multiple flights
            attack_ratio: Ratio of flights to attack (0.0 to 1.0)
            attack_types: List of attack types to use, e.g., ['step', 'drift_ramp', ...]
            **common_params: Common parameters for all attacks (usually empty)
        
        Returns:
            Tuple of (attacked_df, attack_info_df)
        
        Example:
            If attack_ratio=0.5 and 100 flights total:
            - 50 flights are attacked
            - Each of 6 attack types gets ~8-9 flights
            - 50 flights remain clean
        """
        flights = df['flight'].unique()
        n_flights = len(flights)
        n_to_attack = int(n_flights * attack_ratio)
        
        print(f"\n[MIXED ATTACK INJECTION]")
        print(f"Total flights: {n_flights}")
        print(f"Target attack ratio: {attack_ratio*100:.1f}%")
        print(f"Flights to attack: {n_to_attack}")
        
        # 1. Randomly select flights to attack
        attacked_flights = self.rng.choice(flights, n_to_attack, replace=False)
        
        # 2. Assign attack types evenly to attacked flights
        # Use modulo to cycle through attack types for even distribution
        flight_attack_map = {}
        attack_type_counts = {at: 0 for at in attack_types}
        
        for i, flight_id in enumerate(attacked_flights):
            attack_type = attack_types[i % len(attack_types)]
            flight_attack_map[flight_id] = attack_type
            attack_type_counts[attack_type] += 1
        
        # Print planned distribution
        print(f"\nPlanned attack distribution:")
        for attack_type, count in attack_type_counts.items():
            print(f"  {attack_type:20s}: {count:3d} flights ({count/n_flights*100:5.2f}%)")
        
        # 3. Inject attacks flight by flight
        attacked_dfs = []
        attack_records = []
        
        for flight_id in flights:
            flight_df = df[df['flight'] == flight_id].copy()
            
            if flight_id in flight_attack_map:
                # This flight should be attacked
                attack_type = flight_attack_map[flight_id]
                
                # Get attack-specific parameters
                attack_params = config.ATTACK_PARAMS.get(attack_type, {})
                injection_type = config.ATTACK_TYPE_MAP.get(attack_type, attack_type)
                
                # For replay attacks, pass all_flights_df
                if injection_type == 'replay':
                    attack_params['all_flights_df'] = df
                
                # Inject with probability=1.0 (already selected)
                attacked_flight, attack_info = self.inject_attack_to_flight(
                    flight_df,
                    attack_type=injection_type,
                    attack_prob=1.0,
                    **attack_params
                )
                
                # Override attack_type with the original name (not the mapped one)
                if attack_info['attacked']:
                    attack_info['attack_type'] = attack_type  # Use original name like 'drift_ramp' instead of 'drift'
            else:
                # This flight is clean
                attacked_flight = flight_df
                attacked_flight['attack_start_time'] = np.nan
                attacked_flight['attack_type'] = 'none'
                
                attack_info = {
                    'attacked': False,
                    'attack_type': 'none',
                    'attack_start_time': None,
                    'attack_params': {},
                    'consistency_mode': None,
                    'reason': 'not_selected'
                }
            
            attacked_dfs.append(attacked_flight)
            
            # Record attack info
            attack_record = {'flight': flight_id}
            attack_record.update(attack_info)
            attack_records.append(attack_record)
        
        # 4. Combine all flights
        result_df = pd.concat(attacked_dfs, ignore_index=True)
        attack_info_df = pd.DataFrame(attack_records)
        
        # 5. Print actual results
        print(f"\nActual injection results:")
        total_attacked = attack_info_df['attacked'].sum()
        print(f"Total attacked: {total_attacked}/{n_flights} ({total_attacked/n_flights*100:.1f}%)")
        
        for attack_type in attack_types:
            count = (attack_info_df['attack_type'] == attack_type).sum()
            print(f"  {attack_type:20s}: {count:3d} flights ({count/n_flights*100:5.2f}%)")
        
        # Count failures
        n_failed = n_to_attack - total_attacked
        if n_failed > 0:
            print(f"\nWarning: {n_failed} flights failed to inject attacks")
            failure_reasons = attack_info_df[~attack_info_df['attacked']]['reason'].value_counts()
            print(f"\nFailure breakdown:")
            for reason, count in failure_reasons.items():
                if reason != 'not_selected':
                    print(f"  {reason}: {count} flights")
        
        return result_df, attack_info_df
