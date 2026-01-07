"""Quick check of validation progress"""
import os
import json

validation_dir = './validation_output'

if os.path.exists(os.path.join(validation_dir, 'validation_report.json')):
    with open(os.path.join(validation_dir, 'validation_report.json'), 'r') as f:
        results = json.load(f)
    
    print("Current validation results:")
    print(json.dumps(results, indent=2))
else:
    print("No validation results yet")

if os.path.exists(os.path.join(validation_dir, 'validation_summary.md')):
    with open(os.path.join(validation_dir, 'validation_summary.md'), 'r') as f:
        print("\n" + "="*80)
        print(f.read())
