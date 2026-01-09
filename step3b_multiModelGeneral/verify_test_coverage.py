"""
Verify test set coverage for attack types

This script checks whether all attack types are properly represented in the test set,
helping to ensure comprehensive evaluation and avoid inflated metrics.
"""
import os
import json
import pandas as pd
import numpy as np
from collections import Counter
import config


def analyze_attack_coverage(attack_info_csv):
    """Analyze attack type coverage in a dataset."""
    if not os.path.exists(attack_info_csv):
        print(f"File not found: {attack_info_csv}")
        return None
    
    df = pd.read_csv(attack_info_csv)
    
    print(f"\n{'='*70}")
    print(f"Attack Coverage Analysis: {os.path.basename(attack_info_csv)}")
    print(f"{'='*70}")
    
    total_flights = len(df)
    attacked_flights = df['attacked'].sum()
    normal_flights = total_flights - attacked_flights
    
    print(f"Total flights: {total_flights}")
    print(f"Attacked flights: {attacked_flights} ({attacked_flights/total_flights*100:.1f}%)")
    print(f"Normal flights: {normal_flights} ({normal_flights/total_flights*100:.1f}%)")
    
    # Count each attack type
    attack_counts = df['attack_type'].value_counts()
    
    print(f"\nAttack type distribution:")
    print(f"{'Attack Type':<25} {'Count':>6} {'Percentage':>12}")
    print(f"{'-'*45}")
    
    for attack_type in config.ATTACK_TYPES + ['none']:
        count = attack_counts.get(attack_type, 0)
        percentage = count / total_flights * 100
        status = "✓" if count > 0 else "✗ MISSING"
        print(f"{attack_type:<25} {count:>6} {percentage:>10.1f}%  {status}")
    
    # Check for missing attack types
    missing_attacks = [at for at in config.ATTACK_TYPES if attack_counts.get(at, 0) == 0]
    
    if missing_attacks:
        print(f"\n⚠️  WARNING: Missing attack types: {missing_attacks}")
        print(f"   This will lead to incomplete testing and inflated metrics!")
        return False
    else:
        print(f"\n✓ All {len(config.ATTACK_TYPES)} attack types are covered!")
        return True


def compare_main_vs_strategyC():
    """Compare attack coverage between main.py and main_strategyC.py outputs."""
    print("\n" + "="*70)
    print("COMPARING MAIN.PY vs MAIN_STRATEGYC.PY")
    print("="*70)
    
    # Check main.py output
    main_test_info = os.path.join(config.OUTPUT_DIR, 'cnn', 'test_attack_info.csv')
    
    # Check if we're using the old output structure
    if not os.path.exists(main_test_info):
        print(f"\nSearching for test_attack_info.csv in output directory...")
        # Try to find any test_attack_info.csv
        for root, dirs, files in os.walk(config.OUTPUT_DIR):
            if 'test_attack_info.csv' in files:
                main_test_info = os.path.join(root, 'test_attack_info.csv')
                print(f"Found: {main_test_info}")
                break
    
    print("\n" + "-"*70)
    print("MAIN.PY OUTPUT (current implementation)")
    print("-"*70)
    main_ok = analyze_attack_coverage(main_test_info)
    
    # Check strategyC output (if exists)
    strategyC_dir = config.OUTPUT_DIR.replace('output', 'output_strategyC')
    strategyC_test_info = os.path.join(strategyC_dir, 'cnn', 'test_attack_info.csv')
    
    if os.path.exists(strategyC_test_info):
        print("\n" + "-"*70)
        print("MAIN_STRATEGYC.PY OUTPUT (flight reuse)")
        print("-"*70)
        strategyC_ok = analyze_attack_coverage(strategyC_test_info)
    else:
        print(f"\nStrategyC output not found at: {strategyC_test_info}")
        strategyC_ok = None
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if main_ok:
        print("✓ main.py: Test set has complete attack coverage")
    else:
        print("✗ main.py: Test set is MISSING some attack types")
        print("  → This leads to inflated metrics (models look better than they are)")
        print("  → Solution: Use inject_mixed_attacks_with_guarantee() for test set")
    
    if strategyC_ok is not None:
        if strategyC_ok:
            print("✓ main_strategyC.py: Test set has complete attack coverage")
        else:
            print("✗ main_strategyC.py: Test set is MISSING some attack types")


def verify_flight_independence():
    """Verify that train/val/test sets use different flights (no leakage)."""
    print("\n" + "="*70)
    print("FLIGHT INDEPENDENCE CHECK")
    print("="*70)
    
    # Load flight splits
    splits_file = os.path.join(config.OUTPUT_DIR, 'flight_splits.json')
    if not os.path.exists(splits_file):
        print(f"Flight splits file not found: {splits_file}")
        return
    
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    train_flights = set(splits['train'])
    val_flights = set(splits['val'])
    test_flights = set(splits['test'])
    
    print(f"Train flights: {len(train_flights)}")
    print(f"Val flights: {len(val_flights)}")
    print(f"Test flights: {len(test_flights)}")
    
    # Check overlaps
    train_val_overlap = train_flights & val_flights
    train_test_overlap = train_flights & test_flights
    val_test_overlap = val_flights & test_flights
    
    print(f"\nOverlap analysis:")
    print(f"  Train ∩ Val: {len(train_val_overlap)} flights")
    print(f"  Train ∩ Test: {len(train_test_overlap)} flights")
    print(f"  Val ∩ Test: {len(val_test_overlap)} flights")
    
    if train_test_overlap or val_test_overlap:
        print(f"\n✗ DATA LEAKAGE DETECTED!")
        print(f"  Test set shares flights with train/val!")
    else:
        print(f"\n✓ No data leakage - all sets use different flights")


if __name__ == '__main__':
    print("="*70)
    print("TEST SET VERIFICATION TOOL")
    print("="*70)
    print("\nThis tool helps diagnose why metrics might be inflated.")
    print("It checks:")
    print("  1. Whether ALL attack types are represented in test set")
    print("  2. Whether train/val/test sets use different flights (no leakage)")
    print("  3. Comparison between main.py and main_strategyC.py approaches")
    
    verify_flight_independence()
    compare_main_vs_strategyC()
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("\n1. For MAIN.PY (recommended approach):")
    print("   - Use inject_mixed_attacks_with_guarantee() for test set")
    print("   - This ensures all attack types are tested")
    print("   - No flight reuse = better generalization test")
    print("\n2. For comprehensive testing:")
    print("   - Increase TEST_ATTACK_RATIO if needed (e.g., 0.7)")
    print("   - Set MIN_TEST_FLIGHTS_PER_ATTACK ≥ 3")
    print("   - Verify coverage with this script after training")
    print("\n3. AVOID:")
    print("   - Flight reuse in test set (leads to data leakage)")
    print("   - Too low TEST_ATTACK_RATIO (causes missing attack types)")
