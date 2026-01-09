"""
Debug labeling issue - check if attack_info contains attack_start_time
"""
import pandas as pd
import config
from data_loader import load_flights_data, get_flight_subset
from stratified_split import stratified_split_with_reuse, expand_flights_with_all_attacks
from flight_reuse_injector import FlightReuseInjector

print("="*80)
print("DEBUGGING LABEL GENERATION")
print("="*80)

# Load data
df = pd.read_csv(config.DATA_PATH)

# Split data
train_flights, val_flights, test_flights, test_allocation = stratified_split_with_reuse(
    df=df,
    attack_types=config.ATTACK_TYPES,
    train_ratio=config.TRAIN_RATIO,
    val_ratio=config.VAL_RATIO,
    test_ratio=config.TEST_RATIO,
    min_test_per_attack=config.MIN_TEST_FLIGHTS_PER_ATTACK,
    random_seed=config.RANDOM_SEED
)

# Take just 2 flights for quick test
test_flights_sample = train_flights[:2]
test_pairs = expand_flights_with_all_attacks(test_flights_sample, config.ATTACK_TYPES[:1])  # Just 'step'

print(f"\nTest pairs: {test_pairs}")

# Inject attacks
injector = FlightReuseInjector(random_seed=config.RANDOM_SEED)
base_df = load_flights_data(config.DATA_PATH)

print("\n" + "="*80)
print("INJECTING ATTACKS")
print("="*80)
test_df, attack_info = injector.inject_with_flight_reuse(
    base_df,
    test_pairs,
    verbose=True
)

print("\n" + "="*80)
print("ATTACK INFO DATAFRAME")
print("="*80)
print(attack_info)
print(f"\nColumns: {attack_info.columns.tolist()}")

print("\n" + "="*80)
print("CHECKING ATTACK_START_TIME")
print("="*80)
for idx, row in attack_info.iterrows():
    print(f"Flight {row['flight']}: attacked={row['attacked']}, " + 
          f"attack_start_time={row.get('attack_start_time', 'MISSING')}")

# Now test labeling
from data_loader import compute_delta_t, convert_to_local_coordinates
from feature_engineering import compute_all_features
from labeling import generate_point_labels

print("\n" + "="*80)
print("PREPROCESSING AND LABELING")
print("="*80)

test_df = compute_delta_t(test_df)
test_df = convert_to_local_coordinates(test_df)

print(f"\nBefore labeling: label column exists? {'label' in test_df.columns}")

test_df = generate_point_labels(test_df, attack_info)

print(f"\nAfter labeling:")
print(f"  Total samples: {len(test_df)}")
print(f"  Positive samples: {(test_df['label'] == 1).sum()}")
print(f"  Negative samples: {(test_df['label'] == 0).sum()}")

# Check a specific attacked flight
if attack_info['attacked'].any():
    attacked_flight_id = attack_info[attack_info['attacked']]['flight'].iloc[0]
    flight_data = test_df[test_df['flight'] == attacked_flight_id]
    attack_start = attack_info[attack_info['flight'] == attacked_flight_id]['attack_start_time'].iloc[0]
    
    print(f"\n" + "="*80)
    print(f"EXAMINING FLIGHT {attacked_flight_id}")
    print("="*80)
    print(f"Attack start time: {attack_start}")
    print(f"Flight time range: {flight_data['time'].min()} to {flight_data['time'].max()}")
    print(f"Samples before attack: {(flight_data['time'] < attack_start).sum()}")
    print(f"Samples at/after attack: {(flight_data['time'] >= attack_start).sum()}")
    print(f"Positive labels in flight: {(flight_data['label'] == 1).sum()}")
    print(f"Negative labels in flight: {(flight_data['label'] == 0).sum()}")
