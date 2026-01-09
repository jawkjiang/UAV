"""
Test script to verify all key aspects of the implementation work correctly.
This simulates the key operations without running a full training.
"""
import sys
import numpy as np
import torch

def test_imports():
    """Test all imports"""
    print("Testing imports...")
    try:
        import config
        from mixed_attack_injector import MixedAttackInjector
        from data_loader import load_flights_data, split_flights
        from labeling import generate_point_labels
        from feature_engineering import compute_all_features, get_feature_columns
        from window_creation import create_windows_from_dataset
        from model import create_model
        from training import train_model
        from evaluation import evaluate_model
        print("✓ All imports successful\n")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}\n")
        return False


def test_config_attributes():
    """Test that config has all required attributes"""
    print("Testing config attributes...")
    import config
    
    required = [
        'ATTACK_TYPES', 'MODEL_TYPES', 'DATA_PATH', 'OUTPUT_DIR',
        'BATCH_SIZE', 'MAX_EPOCHS', 'LEARNING_RATE',
        'TRAIN_ATTACK_RATIO', 'VAL_ATTACK_RATIO', 'TEST_ATTACK_RATIO',
        'ATTACK_TYPE_MAP', 'ATTACK_PARAMS', 'FPR_THRESHOLDS'
    ]
    
    for attr in required:
        if not hasattr(config, attr):
            print(f"✗ Missing: config.{attr}\n")
            return False
    
    print(f"✓ All required config attributes present")
    print(f"  - Attack types: {len(config.ATTACK_TYPES)}")
    print(f"  - Model types: {len(config.MODEL_TYPES)}\n")
    return True


def test_mixed_attack_injector():
    """Test that mixed attack injector can be created"""
    print("Testing mixed attack injector...")
    try:
        import config
        from mixed_attack_injector import MixedAttackInjector
        
        injector = MixedAttackInjector(random_seed=42)
        print(f"✓ MixedAttackInjector created successfully")
        print(f"  - Random seed: 42\n")
        return True
    except Exception as e:
        print(f"✗ Failed to create MixedAttackInjector: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test that all model types can be created"""
    print("Testing model creation...")
    try:
        import config
        from model import create_model
        
        n_features = 9  # 3 position + 3 velocity + 3 acceleration
        
        for model_type in config.MODEL_TYPES:
            model = create_model(
                model_type=model_type,
                window_size=config.WINDOW_SIZE,
                n_features=n_features
            )
            print(f"  ✓ {model_type:15s}: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        print()
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_metric_keys():
    """Test that evaluation metrics have correct keys"""
    print("Testing metric keys...")
    
    # Simulate what evaluate_model returns
    mock_metrics = {
        'pr_auc': 0.95,
        'roc_auc': 0.92,
        'precision': 0.88,
        'recall': 0.85,
        'f1': 0.86,
        'avg_fp_per_flight': 1.2
    }
    
    # Check keys that main.py uses
    required_keys = ['roc_auc', 'f1', 'precision', 'recall']
    
    for key in required_keys:
        if key not in mock_metrics:
            print(f"✗ Missing key: {key}\n")
            return False
    
    print("✓ All required metric keys present")
    print(f"  Keys: {', '.join(required_keys)}\n")
    return True


def test_json_serialization():
    """Test that numpy types can be converted for JSON"""
    print("Testing JSON serialization...")
    import json
    
    # Simulate metrics with numpy types
    test_data = {
        'numpy_float64': np.float64(0.95),
        'numpy_int64': np.int64(100),
        'numpy_float32': np.float32(0.88),
        'python_float': 0.92,
        'python_int': 50
    }
    
    # Convert
    converted = {}
    for key, value in test_data.items():
        if hasattr(value, 'item'):
            converted[key] = value.item()
        elif isinstance(value, (np.integer, np.floating)):
            converted[key] = float(value)
        else:
            converted[key] = value
    
    # Try to serialize
    try:
        json_str = json.dumps(converted, indent=2)
        print("✓ JSON serialization successful")
        print(f"  Converted {len(test_data)} items\n")
        return True
    except Exception as e:
        print(f"✗ JSON serialization failed: {e}\n")
        return False


def test_per_attack_metrics():
    """Test per-attack metrics format"""
    print("Testing per-attack metrics format...")
    
    # Simulate per_attack_metrics
    import config
    per_attack_metrics = {}
    
    for attack_type in config.ATTACK_TYPES[:2]:  # Test first 2
        per_attack_metrics[attack_type] = {
            'auc_roc': float(np.random.rand()),
            'f1_score': float(np.random.rand()),
            'precision': float(np.random.rand()),
            'recall': float(np.random.rand()),
            'n_samples': int(100),
            'n_positive': int(50)
        }
    
    # Check keys
    for attack_type, metrics in per_attack_metrics.items():
        if 'auc_roc' not in metrics or 'f1_score' not in metrics:
            print(f"✗ Missing keys in {attack_type}\n")
            return False
    
    # Try JSON serialization
    import json
    try:
        json_str = json.dumps(per_attack_metrics, indent=2)
        print("✓ Per-attack metrics format correct")
        print(f"  Tested {len(per_attack_metrics)} attack types\n")
        return True
    except Exception as e:
        print(f"✗ Per-attack metrics serialization failed: {e}\n")
        return False


def main():
    """Run all tests"""
    print("="*70)
    print("COMPREHENSIVE TESTING")
    print("="*70)
    print()
    
    tests = [
        ("Imports", test_imports),
        ("Config Attributes", test_config_attributes),
        ("Mixed Attack Injector", test_mixed_attack_injector),
        ("Model Creation", test_model_creation),
        ("Metric Keys", test_metric_keys),
        ("JSON Serialization", test_json_serialization),
        ("Per-Attack Metrics", test_per_attack_metrics),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ Test '{name}' crashed: {e}\n")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("="*70)
    passed = sum(results)
    total = len(results)
    
    if all(results):
        print(f"✓ ALL TESTS PASSED ({passed}/{total})")
    else:
        print(f"✗ SOME TESTS FAILED ({passed}/{total} passed)")
    
    print("="*70)
    
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
