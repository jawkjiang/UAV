"""
Quick Test Script for Step3b

Validates that all components are properly configured before running
the full training pipeline.
"""

import sys
import os

print("="*80)
print("Step3b Quick Validation Test")
print("="*80)

# Test 1: Import config
print("\n[1/7] Testing config import...")
try:
    import config
    print(f"✓ Config loaded")
    print(f"  - Attack types: {len(config.ATTACK_TYPES)}")
    print(f"  - Model types: {len(config.MODEL_TYPES)}")
    print(f"  - Train attack ratio: {config.TRAIN_ATTACK_RATIO}")
except Exception as e:
    print(f"✗ Failed to import config: {e}")
    sys.exit(1)

# Test 2: Import base injector
print("\n[2/7] Testing multi_attack_injector import...")
try:
    from multi_attack_injector import MultiAttackInjector
    print(f"✓ MultiAttackInjector loaded")
except Exception as e:
    print(f"✗ Failed to import MultiAttackInjector: {e}")
    sys.exit(1)

# Test 3: Import mixed injector
print("\n[3/7] Testing mixed_attack_injector import...")
try:
    from mixed_attack_injector import MixedAttackInjector
    injector = MixedAttackInjector(random_seed=42)
    print(f"✓ MixedAttackInjector loaded and instantiated")
except Exception as e:
    print(f"✗ Failed to import MixedAttackInjector: {e}")
    sys.exit(1)

# Test 4: Import model module
print("\n[4/7] Testing model import...")
try:
    from model import create_model
    # Test creating a simple model
    test_model = create_model('cnn', n_features=10, window_size=50)
    print(f"✓ Model module loaded")
    print(f"  - Test CNN model created successfully")
except Exception as e:
    print(f"✗ Failed to import model: {e}")
    sys.exit(1)

# Test 5: Import data processing modules
print("\n[5/7] Testing data processing modules...")
try:
    from data_loader import load_flights_data, split_flights
    from feature_engineering import get_feature_columns
    from window_creation import WindowDataset
    from labeling import generate_point_labels
    print(f"✓ Data processing modules loaded")
except Exception as e:
    print(f"✗ Failed to import data processing modules: {e}")
    sys.exit(1)

# Test 6: Import training modules
print("\n[6/7] Testing training modules...")
try:
    from training import train_model
    from evaluation import evaluate_model
    print(f"✓ Training modules loaded")
except Exception as e:
    print(f"✗ Failed to import training modules: {e}")
    sys.exit(1)

# Test 7: Check data file
print("\n[7/7] Checking data file...")
if os.path.exists(config.DATA_PATH):
    print(f"✓ Data file found: {config.DATA_PATH}")
else:
    print(f"✗ Data file not found: {config.DATA_PATH}")
    print(f"  Note: Update DATA_PATH in config.py if needed")

# Summary
print("\n" + "="*80)
print("VALIDATION COMPLETE")
print("="*80)
print("\nAll core components validated successfully!")
print("\nYou can now run:")
print("  python main.py          # Train all models")
print("  python compare_models.py # Analyze results")
print("\n" + "="*80)
