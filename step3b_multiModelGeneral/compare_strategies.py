"""
Visualize the difference between flight reuse strategies
"""
import matplotlib.pyplot as plt
import numpy as np

# Configuration
attack_types = ['step', 'drift_ramp', 'drift_sigmoid', 'delay', 'takeover_step', 'takeover_ramp']
total_flights = 209

print('='*80)
print('FLIGHT REUSE STRATEGY COMPARISON')
print('='*80)

# ============================================================================
# Current Strategy
# ============================================================================
print('\n' + '='*80)
print('CURRENT STRATEGY (Random Split)')
print('='*80)

current_train = 146
current_val = 31
current_test = 32

print(f'\nData Split:')
print(f'  Train: {current_train} flights')
print(f'  Val: {current_val} flights')
print(f'  Test: {current_test} flights')

print(f'\nTest Set Attack Distribution (actual):')
print(f'  step: 1 flight')
print(f'  drift_ramp: 1 flight')
print(f'  drift_sigmoid: 1 flight')
print(f'  delay: 0 flights ❌')
print(f'  takeover_step: 0 flights ❌')
print(f'  takeover_ramp: 0 flights ❌')
print(f'  none: 29 flights')

# ============================================================================
# Strategy A: Full Reuse (NOT RECOMMENDED)
# ============================================================================
print('\n' + '='*80)
print('STRATEGY A: Full Reuse (⚠️  NOT RECOMMENDED)')
print('='*80)

a_train = 146 * 7  # 6 attacks + 1 normal per flight
a_val = 31 * 7
a_test = 32 * 7

print(f'\nData after reuse:')
print(f'  Train: {a_train} samples ({current_train} flights × 7 versions)')
print(f'  Val: {a_val} samples ({current_val} flights × 7 versions)')
print(f'  Test: {a_test} samples ({current_test} flights × 7 versions)')

print(f'\nTest set per attack: ~{32} samples')

print(f'\n❌ CRITICAL ISSUE: Data Leakage!')
print(f'   - Test samples share same underlying flights as training')
print(f'   - Model may memorize flight-specific patterns')
print(f'   - Performance metrics will be inflated')
print(f'   - Cannot validate generalization to new flights')

# ============================================================================
# Strategy B: Stratified Reuse (ACCEPTABLE)
# ============================================================================
print('\n' + '='*80)
print('STRATEGY B: Stratified Reuse Within Groups (⚠️  ACCEPTABLE)')
print('='*80)

b_train_flights = 146
b_val_flights = 31
b_test_flights = 32

b_train = b_train_flights * 7
b_val = b_val_flights * 7
b_test = b_test_flights * 7

print(f'\nStep 1: Split flights (same as current)')
print(f'  Train: {b_train_flights} flights')
print(f'  Val: {b_val_flights} flights')
print(f'  Test: {b_test_flights} flights')

print(f'\nStep 2: Reuse WITHIN each group')
print(f'  Train: {b_train_flights} × 7 = {b_train} samples')
print(f'  Val: {b_val_flights} × 7 = {b_val} samples')
print(f'  Test: {b_test_flights} × 7 = {b_test} samples')

print(f'\nTest set per attack: ~{b_test_flights} samples')

print(f'\n✅ ADVANTAGES:')
print(f'   + All attack types in test set')
print(f'   + Sufficient samples per attack type')
print(f'   + No cross-group leakage')

print(f'\n⚠️  LIMITATIONS:')
print(f'   - Cannot validate generalization to NEW flights')
print(f'   - May overfit to specific flight patterns')
print(f'   - Test performance reflects "cross-attack" generalization only')

# ============================================================================
# Strategy C: Conservative Hybrid (RECOMMENDED)
# ============================================================================
print('\n' + '='*80)
print('STRATEGY C: Conservative Hybrid (✅ RECOMMENDED)')
print('='*80)

# Adjust ratios for better test coverage
c_train_flights = int(total_flights * 0.50)  # 105 flights
c_val_flights = int(total_flights * 0.25)    # 52 flights  
c_test_flights = int(total_flights * 0.25)   # 52 flights

c_train = c_train_flights * 7   # Reuse for training
c_val = c_val_flights * 7       # Reuse for validation
c_test = c_test_flights         # NO reuse for testing

print(f'\nStep 1: Adjust split ratios for larger test set')
print(f'  Train flights: {c_train_flights} (50%)')
print(f'  Val flights: {c_val_flights} (25%)')
print(f'  Test flights: {c_test_flights} (25%)')

print(f'\nStep 2: Reuse only for train/val, NOT for test')
print(f'  Train: {c_train_flights} × 7 = {c_train} samples (reused)')
print(f'  Val: {c_val_flights} × 7 = {c_val} samples (reused)')
print(f'  Test: {c_test_flights} flights (NOT reused)')

print(f'\nStep 3: Use stratified sampling for test set')
min_per_attack = 8
print(f'  Ensure >= {min_per_attack} flights per attack type in test')
print(f'  Total test flights: ~{c_test_flights}')
print(f'  Per attack: ~{min_per_attack} flights')
print(f'  Plus normal: ~{min_per_attack} flights')

print(f'\n✅ ADVANTAGES:')
print(f'   + Large training set ({c_train} samples)')
print(f'   + All attack types well-represented in test')
print(f'   + Test set maintains flight-level independence')
print(f'   + Can validate generalization to new flights')
print(f'   + Realistic performance metrics')

print(f'\n⚠️  CONSIDERATIONS:')
print(f'   - Training set reuse may cause some overfitting')
print(f'   - Need regularization (dropout, weight decay)')
print(f'   - Need to verify no memorization of flight patterns')

# ============================================================================
# Strategy D: No Reuse with Stratified Sampling (MOST CONSERVATIVE)
# ============================================================================
print('\n' + '='*80)
print('STRATEGY D: No Reuse + Stratified Sampling (🛡️  MOST CONSERVATIVE)')
print('='*80)

d_train_flights = int(total_flights * 0.60)  # 125 flights
d_val_flights = int(total_flights * 0.15)    # 31 flights
d_test_flights = int(total_flights * 0.25)   # 52 flights

print(f'\nAdjusted split for better test coverage (NO reuse anywhere):')
print(f'  Train: {d_train_flights} flights')
print(f'  Val: {d_val_flights} flights')
print(f'  Test: {d_test_flights} flights')

print(f'\nUse stratified sampling:')
print(f'  Ensure each attack type has >= {min_per_attack} flights in test')

print(f'\n✅ ADVANTAGES:')
print(f'   + Complete independence between sets')
print(f'   + Most realistic evaluation')
print(f'   + Highest confidence in generalization')
print(f'   + No risk of memorization')

print(f'\n⚠️  DISADVANTAGES:')
print(f'   - Smaller training set ({d_train_flights} vs {c_train})')
print(f'   - May need more regularization')
print(f'   - Training may be slower to converge')

# ============================================================================
# Recommendation
# ============================================================================
print('\n' + '='*80)
print('RECOMMENDATION')
print('='*80)

print(f'''
For your current problem (incomplete test set coverage):

PRIMARY RECOMMENDATION: Strategy C (Conservative Hybrid)
├─ Reuse for training/validation (more data = better learning)
├─ No reuse for testing (realistic evaluation)
├─ Stratified sampling ensures all attacks represented
└─ Best balance between data efficiency and evaluation quality

ALTERNATIVE: Strategy D (Most Conservative)
├─ If you want absolute confidence in results
├─ If you have concerns about flight memorization
└─ Trade-off: smaller training set

NOT RECOMMENDED: Strategy A or B
├─ Strategy A: Severe data leakage
└─ Strategy B: Cannot validate flight-level generalization

IMPLEMENTATION ORDER:
1. First try Strategy C (recommended)
2. Compare with current results
3. If overfitting suspected, try Strategy D
4. Report both results with clear methodology description
''')

print('\n' + '='*80)
print('NEXT STEPS')
print('='*80)
print('''
Would you like me to implement:
1. Strategy C (Conservative Hybrid) - RECOMMENDED
2. Strategy D (No Reuse + Stratified Sampling)
3. Both for comparison

Just let me know which one you prefer!
''')
