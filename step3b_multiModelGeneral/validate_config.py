"""
Comprehensive validation script to check for all potential runtime errors
before running the main training script.
"""
import os
import sys
import importlib
import inspect

def validate_config():
    """Validate config has all required attributes"""
    import config
    
    # List all config attributes used in the code
    required_attrs = [
        # Basic paths
        'DATA_PATH', 'OUTPUT_DIR',
        # Data split
        'TRAIN_RATIO', 'VAL_RATIO', 'TEST_RATIO',
        # Window params
        'WINDOW_SIZE', 'STEP_SIZE',
        # Features
        'POSITION_FEATURES', 'VELOCITY_FEATURES', 'ACCELERATION_FEATURES',
        # Training
        'BATCH_SIZE', 'MAX_EPOCHS', 'NUM_EPOCHS', 'LEARNING_RATE', 'EARLY_STOPPING_PATIENCE',
        # Attack config
        'ATTACK_TYPES', 'TRAIN_ATTACK_RATIO', 'VAL_ATTACK_RATIO', 'TEST_ATTACK_RATIO',
        # Models
        'MODEL_TYPES',
        # Window balancing
        'TRAIN_POS_RATIO', 'VAL_POS_RATIO', 'TEST_POS_RATIO',
        # Random seed
        'RANDOM_SEED',
        # Skip existing
        'SKIP_EXISTING',
        # Attack parameters
        'ATTACK_START_BUFFER', 'ATTACK_END_BUFFER',
        'ATTACK_MAGNITUDES', 'DIRECTION_MODES', 'CONSISTENCY_MODES',
        'VELOCITY_EPSILON',
        'STEP1_ATTACK_DURATION', 'STEP1_ATTACK_TIME_CONSTANT',
        'STEP1_VELOCITY_TRANSIENT_MIN', 'STEP1_VELOCITY_TRANSIENT_MAX',
        'DRIFT_PROFILES', 'DRIFT_DURATIONS', 'SIGMOID_K',
        'DELAY_SECONDS',
        'REPLAY_SEGMENT_DURATIONS', 'REPLAY_DONOR_SOURCES', 'REPLAY_STITCHING_MODES',
        'REPLAY_TRANSITION_SECONDS', 'REPLAY_GAP_SECONDS',
        'TAKEOVER_OFFSET_PROFILES', 'TAKEOVER_DURATIONS', 'TAKEOVER_ALPHAS', 'TAKEOVER_TAUS',
        # Evaluation
        'FPR_THRESHOLDS',
        # Attack mapping
        'ATTACK_TYPE_MAP', 'ATTACK_PARAMS'
    ]
    
    missing = []
    for attr in required_attrs:
        if not hasattr(config, attr):
            missing.append(attr)
    
    if missing:
        print("✗ Missing config attributes:")
        for attr in missing:
            print(f"  - {attr}")
        return False
    else:
        print("✓ All config attributes present")
        return True


def validate_function_signatures():
    """Validate function signatures match between modules"""
    issues = []
    
    # Check evaluate_model signature
    from evaluation import evaluate_model
    sig = inspect.signature(evaluate_model)
    params = list(sig.parameters.keys())
    if 'flight_ids' not in params:
        issues.append("evaluate_model missing 'flight_ids' parameter")
    
    # Check train_model signature
    from training import train_model
    sig = inspect.signature(train_model)
    params = list(sig.parameters.keys())
    if 'num_epochs' not in params:
        issues.append("train_model missing 'num_epochs' parameter")
    if 'max_epochs' in params and 'num_epochs' not in params:
        issues.append("train_model uses 'max_epochs' instead of 'num_epochs'")
    
    if issues:
        print("✗ Function signature issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✓ Function signatures OK")
        return True


def validate_return_values():
    """Check that evaluate_model returns 3 values"""
    from evaluation import evaluate_model
    sig = inspect.signature(evaluate_model)
    
    # Read the source to check return statement
    source = inspect.getsource(evaluate_model)
    if 'return metrics, predictions, targets' in source:
        print("✓ evaluate_model returns 3 values (metrics, predictions, targets)")
        return True
    else:
        print("✗ evaluate_model return signature unclear")
        return False


def validate_imports():
    """Validate all imports work"""
    try:
        import config
        from data_loader import load_flights_data, split_flights, get_flight_subset, compute_delta_t, convert_to_local_coordinates
        from labeling import generate_point_labels
        from feature_engineering import compute_all_features, get_feature_columns, normalize_features
        from window_creation import create_windows_from_dataset, balance_windows, WindowDataset
        from model import create_model
        from training import train_model
        from evaluation import evaluate_model
        from mixed_attack_injector import MixedAttackInjector
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_paths():
    """Validate required paths exist"""
    import config
    
    # Check data path
    if not os.path.exists(config.DATA_PATH):
        print(f"✗ Data file not found: {config.DATA_PATH}")
        return False
    else:
        print(f"✓ Data file found: {config.DATA_PATH}")
    
    # Output dir will be created
    print(f"✓ Output dir will be created: {config.OUTPUT_DIR}")
    return True


def main():
    """Run all validations"""
    print("="*60)
    print("COMPREHENSIVE VALIDATION")
    print("="*60)
    
    results = []
    
    print("\n1. Validating configuration...")
    results.append(validate_config())
    
    print("\n2. Validating imports...")
    results.append(validate_imports())
    
    print("\n3. Validating function signatures...")
    results.append(validate_function_signatures())
    
    print("\n4. Validating return values...")
    results.append(validate_return_values())
    
    print("\n5. Validating paths...")
    results.append(validate_paths())
    
    print("\n" + "="*60)
    if all(results):
        print("✓ ALL VALIDATIONS PASSED")
        print("="*60)
        return 0
    else:
        print("✗ SOME VALIDATIONS FAILED")
        print("="*60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
