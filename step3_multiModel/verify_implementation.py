"""
Implementation Verification Script

Verifies that all required files and components are in place.
"""

import os
from pathlib import Path


def check_file_exists(filepath, description):
    """Check if a file exists and print status."""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"  ✅ {description:40s} ({size:,} bytes)")
        return True
    else:
        print(f"  ❌ {description:40s} MISSING!")
        return False


def verify_implementation():
    """Verify all components are in place."""
    
    print("="*80)
    print("STEP 3 IMPLEMENTATION VERIFICATION")
    print("="*80)
    
    base_dir = Path(__file__).parent
    all_ok = True
    
    # Core implementation files
    print("\n1. Core Implementation Files:")
    all_ok &= check_file_exists(base_dir / 'model.py', 'model.py (7 model architectures)')
    all_ok &= check_file_exists(base_dir / 'main.py', 'main.py (experiment runner)')
    all_ok &= check_file_exists(base_dir / 'compare_models.py', 'compare_models.py (analysis)')
    all_ok &= check_file_exists(base_dir / 'test_models.py', 'test_models.py (validation)')
    all_ok &= check_file_exists(base_dir / 'quick_test.py', 'quick_test.py (demo)')
    
    # Configuration and injection
    print("\n2. Configuration Files (from step2):")
    all_ok &= check_file_exists(base_dir / 'config_step3.py', 'config_step3.py')
    all_ok &= check_file_exists(base_dir / 'multi_attack_injector.py', 'multi_attack_injector.py')
    
    # Data processing (from step1)
    print("\n3. Data Processing Files (from step1):")
    all_ok &= check_file_exists(base_dir / 'data_loader.py', 'data_loader.py')
    all_ok &= check_file_exists(base_dir / 'labeling.py', 'labeling.py')
    all_ok &= check_file_exists(base_dir / 'feature_engineering.py', 'feature_engineering.py')
    all_ok &= check_file_exists(base_dir / 'window_creation.py', 'window_creation.py')
    
    # Training and evaluation (from step1)
    print("\n4. Training/Evaluation Files (from step1):")
    all_ok &= check_file_exists(base_dir / 'training.py', 'training.py')
    all_ok &= check_file_exists(base_dir / 'evaluation.py', 'evaluation.py')
    
    # Documentation
    print("\n5. Documentation:")
    all_ok &= check_file_exists(base_dir / 'README.md', 'README.md')
    all_ok &= check_file_exists(base_dir / 'IMPLEMENTATION_SUMMARY.md', 'IMPLEMENTATION_SUMMARY.md')
    
    # Verify model implementations
    print("\n6. Model Implementations:")
    try:
        from model import create_model
        
        models = ['cnn', 'lstm', 'bilstm', 'gru', 'cnn_lstm', 'tcn', 'transformer']
        for model_type in models:
            try:
                model = create_model(model_type, n_features=13, window_size=50, dropout=0.3)
                num_params = sum(p.numel() for p in model.parameters())
                print(f"  ✅ {model_type:15s} model ({num_params:,} params)")
            except Exception as e:
                print(f"  ❌ {model_type:15s} model - ERROR: {e}")
                all_ok = False
    except Exception as e:
        print(f"  ❌ Could not import create_model: {e}")
        all_ok = False
    
    # Summary
    print("\n" + "="*80)
    if all_ok:
        print("✅ ALL CHECKS PASSED - Implementation is complete!")
        print("\nNext steps:")
        print("  1. Run: python test_models.py")
        print("  2. Run: python quick_test.py (demo - 4 experiments)")
        print("  3. Run: python main.py (full - 56 experiments)")
        print("  4. Run: python compare_models.py (analysis)")
    else:
        print("❌ SOME CHECKS FAILED - Please fix the issues above")
    print("="*80)
    
    return all_ok


if __name__ == '__main__':
    import sys
    success = verify_implementation()
    sys.exit(0 if success else 1)
