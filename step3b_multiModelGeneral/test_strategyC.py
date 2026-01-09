"""
Quick validation test for Strategy C implementation
"""
import sys
import os

print("="*80)
print("STRATEGY C - QUICK VALIDATION TEST")
print("="*80)

passed = []
failed = []

def test(name, func):
    try:
        func()
        passed.append(name)
        print(f"✓ {name}")
        return True
    except Exception as e:
        failed.append((name, str(e)))
        print(f"✗ {name}: {e}")
        return False

# Test 1: Config
def test_config():
    import config
    assert hasattr(config, 'USE_FLIGHT_REUSE')
    assert hasattr(config, 'MIN_TEST_FLIGHTS_PER_ATTACK')
    assert config.TRAIN_RATIO == 0.60
    assert config.VAL_RATIO == 0.20
    assert config.TEST_RATIO == 0.20

test("Config updated", test_config)

# Test 2: Stratified split
def test_split():
    from stratified_split import stratified_split_with_reuse
    import pandas as pd
    
    df = pd.DataFrame({'flight': range(209)})
    train, val, test, allocation = stratified_split_with_reuse(
        df, ['step', 'drift_ramp'], min_test_per_attack=2, random_seed=42
    )
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
    assert 'step' in allocation
    assert 'drift_ramp' in allocation

test("Stratified split", test_split)

# Test 3: Flight expansion
def test_expansion():
    from stratified_split import expand_flights_with_all_attacks
    
    flights = [1, 2, 3]
    attacks = ['step', 'drift_ramp']
    expanded = expand_flights_with_all_attacks(flights, attacks)
    
    # Should have 3 flights × 3 versions (2 attacks + 1 normal)
    assert len(expanded) == 9
    assert (1, 'none') in expanded
    assert (1, 'step') in expanded
    assert (1, 'drift_ramp') in expanded

test("Flight expansion", test_expansion)

# Test 4: Flight reuse injector
def test_injector():
    from flight_reuse_injector import FlightReuseInjector
    
    injector = FlightReuseInjector(random_seed=42)
    assert injector is not None

test("Flight reuse injector", test_injector)

# Test 5: Main script exists
def test_main_script():
    assert os.path.exists('main_strategyC.py')
    assert os.path.exists('run_strategyC.py')

test("Main scripts exist", test_main_script)

# Test 6: All imports work
def test_imports():
    import config
    from stratified_split import stratified_split_with_reuse, expand_flights_with_all_attacks
    from flight_reuse_injector import FlightReuseInjector
    from data_loader import load_flights_data
    from model import create_model

test("All imports work", test_imports)

print("\n" + "="*80)
print(f"VALIDATION RESULTS: {len(passed)}/{len(passed)+len(failed)} tests passed")
print("="*80)

if failed:
    print("\nFailed tests:")
    for name, error in failed:
        print(f"  ✗ {name}")
        print(f"    {error}")
    sys.exit(1)
else:
    print("\n✓ All validation tests passed!")
    print("\nYou can now run:")
    print("  python run_strategyC.py")
    print("\nOr directly:")
    print("  python main_strategyC.py")
    sys.exit(0)
