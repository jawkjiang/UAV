"""
Strategy C Training Launcher

This script trains models using Conservative Hybrid strategy:
- Train/Val sets use flight reuse (more data)
- Test set uses stratified sampling (realistic evaluation)
"""
import subprocess
import sys
import os

print("="*80)
print("STRATEGY C: CONSERVATIVE HYBRID")
print("="*80)
print("""
This will train models using:
- Train set: ~714 samples (102 flights × 7 versions)
- Val set: ~357 samples (51 flights × 7 versions)
- Test set: ~56 flights (NO reuse, stratified sampling)

Test set will include:
- Each attack type: 8 independent flights
- Normal flights: 8 independent flights

Expected training time: ~2-3 hours for all 7 models
""")

response = input("\nDo you want to proceed? (yes/no): ")

if response.lower() not in ['yes', 'y']:
    print("Training cancelled.")
    sys.exit(0)

print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)

# Remove old output
if os.path.exists('./output'):
    print("Removing old output directory...")
    import shutil
    shutil.rmtree('./output')

# Run main_strategyC.py
print("\nLaunching main_strategyC.py...")
result = subprocess.run(
    [sys.executable, 'main_strategyC.py'],
    cwd=os.path.dirname(os.path.abspath(__file__))
)

sys.exit(result.returncode)
