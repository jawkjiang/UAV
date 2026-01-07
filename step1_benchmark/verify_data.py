"""
Quick verification script to check data loading and basic pipeline components.
"""
import config
from data_loader import load_flights_data, split_flights, compute_delta_t

def verify_data():
    """Verify that data can be loaded and has expected structure."""
    
    print("="*60)
    print("Data Verification Script")
    print("="*60)
    
    # Load data
    print(f"\n1. Loading data from: {config.DATA_PATH}")
    try:
        df = load_flights_data()
        print(f"   ✓ Successfully loaded {len(df)} rows")
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        return False
    
    # Check required columns
    print("\n2. Checking required columns...")
    required_cols = [
        'flight', 'time', 
        'position_x', 'position_y',
        'velocity_x', 'velocity_y',
        'linear_acceleration_x', 'linear_acceleration_y'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"   ✗ Missing columns: {missing_cols}")
        return False
    else:
        print(f"   ✓ All required columns present")
    
    # Check data types
    print("\n3. Checking data characteristics...")
    print(f"   - Number of flights: {df['flight'].nunique()}")
    print(f"   - Flight IDs: {sorted(df['flight'].unique())[:10]}...")
    print(f"   - Time range: {df['time'].min():.2f} to {df['time'].max():.2f}")
    print(f"   - Position range X: {df['position_x'].min():.2f} to {df['position_x'].max():.2f}")
    print(f"   - Position range Y: {df['position_y'].min():.2f} to {df['position_y'].max():.2f}")
    
    # Test splitting
    print("\n4. Testing flight split...")
    try:
        train_flights, val_flights, test_flights = split_flights(df)
        total = len(train_flights) + len(val_flights) + len(test_flights)
        print(f"   ✓ Split successful: {len(train_flights)}/{len(val_flights)}/{len(test_flights)} = {total} flights")
    except Exception as e:
        print(f"   ✗ Error splitting: {e}")
        return False
    
    # Test delta_t computation
    print("\n5. Testing delta_t computation...")
    try:
        df = compute_delta_t(df)
        print(f"   ✓ Delta_t computed")
        print(f"   - Delta_t range: {df['delta_t'].min():.4f} to {df['delta_t'].max():.4f}")
        print(f"   - Mean delta_t: {df['delta_t'].mean():.4f}")
    except Exception as e:
        print(f"   ✗ Error computing delta_t: {e}")
        return False
    
    # Check for irregular time sampling
    print("\n6. Verifying irregular time sampling...")
    delta_t_values = df[df['delta_t'] > 0]['delta_t'].values
    delta_t_std = delta_t_values.std()
    print(f"   - Delta_t std deviation: {delta_t_std:.4f}")
    if delta_t_std > 0.01:
        print(f"   ✓ Confirmed irregular sampling (Δt varies)")
    else:
        print(f"   ⚠ Warning: Time sampling appears regular (Δt constant)")
    
    print("\n" + "="*60)
    print("Verification complete! Data is ready for processing.")
    print("="*60)
    print("\nNext steps:")
    print("  1. Review config.py for parameter settings")
    print("  2. Run: python main.py")
    print("="*60)
    
    return True


if __name__ == '__main__':
    verify_data()
