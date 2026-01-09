"""
Quick script to check if window_metadata.csv has been generated
"""
import os
from pathlib import Path

output_dir = Path("output")
models = ['cnn', 'lstm', 'bilstm', 'gru', 'cnn_lstm', 'tcn', 'transformer']

print("=" * 80)
print("Checking for window_metadata.csv files")
print("=" * 80)

for model in models:
    metadata_file = output_dir / model / "window_metadata.csv"
    if metadata_file.exists():
        size = metadata_file.stat().st_size
        print(f"✓ {model}: Found ({size:,} bytes)")
    else:
        print(f"✗ {model}: Not found")

print("=" * 80)
