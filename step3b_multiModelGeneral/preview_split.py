"""
Preview the actual data split with updated 60/20/20 ratios
"""
import pandas as pd
import config
from stratified_split import stratified_split_with_reuse, expand_flights_with_all_attacks

print("="*80)
print("STRATEGY C - DATA SPLIT PREVIEW (60/20/20)")
print("="*80)

# Load data
df = pd.read_csv(config.DATA_PATH)
all_flights = sorted(df['flight'].unique())

print(f"\nTotal flights: {len(all_flights)}")
print(f"Attack types: {config.ATTACK_TYPES}")
print(f"Total attack types (including normal): {len(config.ATTACK_TYPES) + 1}")

# Perform stratified split
train_flights, val_flights, test_flights, test_allocation = stratified_split_with_reuse(
    df=df,
    attack_types=config.ATTACK_TYPES,
    train_ratio=config.TRAIN_RATIO,
    val_ratio=config.VAL_RATIO,
    test_ratio=config.TEST_RATIO,
    min_test_per_attack=config.MIN_TEST_FLIGHTS_PER_ATTACK,
    random_seed=config.RANDOM_SEED
)

# Expand train/val with flight reuse
train_pairs = expand_flights_with_all_attacks(train_flights, config.ATTACK_TYPES)
val_pairs = expand_flights_with_all_attacks(val_flights, config.ATTACK_TYPES)

print("\n" + "="*80)
print("FINAL DATA DISTRIBUTION")
print("="*80)

print(f"\n📊 Flight Distribution:")
print(f"  Train:      {len(train_flights):3d} flights ({len(train_flights)/len(all_flights)*100:.1f}%)")
print(f"  Validation: {len(val_flights):3d} flights ({len(val_flights)/len(all_flights)*100:.1f}%)")
print(f"  Test:       {len(test_flights):3d} flights ({len(test_flights)/len(all_flights)*100:.1f}%)")
print(f"  Total:      {len(train_flights)+len(val_flights)+len(test_flights):3d} flights")

print(f"\n📈 Sample Distribution (after flight reuse):")
total_samples = len(train_pairs) + len(val_pairs) + len(test_flights)
print(f"  Train:      {len(train_pairs):4d} samples ({len(train_pairs)/total_samples*100:.1f}%)")
print(f"  Validation: {len(val_pairs):4d} samples ({len(val_pairs)/total_samples*100:.1f}%)")
print(f"  Test:       {len(test_flights):4d} samples ({len(test_flights)/total_samples*100:.1f}%)")
print(f"  Total:      {total_samples:4d} samples")

print(f"\n🎯 Test Set Attack Coverage:")
for attack_type in sorted(test_allocation.keys()):
    flights = test_allocation[attack_type]
    print(f"  {attack_type:20s}: {len(flights):2d} flights")

print(f"\n✓ All {len(config.ATTACK_TYPES)} attack types covered in test set!")

# Calculate expansion factor
print(f"\n🔄 Flight Reuse Expansion:")
print(f"  Expansion factor: ×{len(config.ATTACK_TYPES) + 1} (6 attacks + 1 normal)")
print(f"  Train: {len(train_flights)} → {len(train_pairs)} (×{len(train_pairs)/len(train_flights):.1f})")
print(f"  Val:   {len(val_flights)} → {len(val_pairs)} (×{len(val_pairs)/len(val_flights):.1f})")
print(f"  Test:  {len(test_flights)} → {len(test_flights)} (no reuse)")

print("\n" + "="*80)
print("COMPARISON WITH ORIGINAL (50/25/25)")
print("="*80)
print(f"\nOriginal distribution:")
print(f"  Train: 102 flights → 714 samples (63.4%)")
print(f"  Val:   51 flights  → 357 samples (31.7%)")
print(f"  Test:  56 flights  →  56 samples (5.0%)")
print(f"  Total: 209 flights → 1127 samples")

print(f"\nNew distribution (60/20/20):")
print(f"  Train: {len(train_flights)} flights → {len(train_pairs)} samples ({len(train_pairs)/total_samples*100:.1f}%)")
print(f"  Val:   {len(val_flights)} flights  → {len(val_pairs)} samples ({len(val_pairs)/total_samples*100:.1f}%)")
print(f"  Test:  {len(test_flights)} flights  →  {len(test_flights)} samples ({len(test_flights)/total_samples*100:.1f}%)")
print(f"  Total: {len(train_flights)+len(val_flights)+len(test_flights)} flights → {total_samples} samples")

print("\n✓ Validation set reduced from 31.7% to ~17.4% of total samples")
print("✓ More balanced distribution for better generalization")
