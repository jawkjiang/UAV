import pandas as pd
import json
import numpy as np

# Load flight splits
with open('output/flight_splits.json', 'r') as f:
    splits = json.load(f)

print('='*70)
print('FLIGHT SPLITS')
print('='*70)
print(f'  Train: {len(splits["train"])} flights')
print(f'  Val: {len(splits["val"])} flights')
print(f'  Test: {len(splits["test"])} flights')

# Load attack info
train_info = pd.read_csv('output/cnn/train_attack_info.csv')
val_info = pd.read_csv('output/cnn/val_attack_info.csv')
test_info = pd.read_csv('output/cnn/test_attack_info.csv')

print('\n' + '='*70)
print('TRAIN SET ATTACK DISTRIBUTION')
print('='*70)
print(train_info['attack_type'].value_counts())
print(f'\nTotal attacked: {train_info["attacked"].sum()}/{len(train_info)} flights ({train_info["attacked"].mean()*100:.1f}%)')

print('\n' + '='*70)
print('VAL SET ATTACK DISTRIBUTION')
print('='*70)
print(val_info['attack_type'].value_counts())
print(f'\nTotal attacked: {val_info["attacked"].sum()}/{len(val_info)} flights ({val_info["attacked"].mean()*100:.1f}%)')

print('\n' + '='*70)
print('TEST SET ATTACK DISTRIBUTION')
print('='*70)
print(test_info['attack_type'].value_counts())
print(f'\nTotal attacked: {test_info["attacked"].sum()}/{len(test_info)} flights ({test_info["attacked"].mean()*100:.1f}%)')

# Check for data leakage
train_flights = set(train_info['flight'])
val_flights = set(val_info['flight'])
test_flights = set(test_info['flight'])

overlap_train_test = train_flights & test_flights
overlap_val_test = val_flights & test_flights
overlap_train_val = train_flights & val_flights

print('\n' + '='*70)
print('DATA LEAKAGE CHECK')
print('='*70)
print(f'Train-Test overlap: {len(overlap_train_test)} flights')
print(f'Val-Test overlap: {len(overlap_val_test)} flights')
print(f'Train-Val overlap: {len(overlap_train_val)} flights')

if len(overlap_train_test) > 0:
    print(f'\n⚠️  WARNING: Train and Test sets have overlapping flights!')
    print(f'   This causes data leakage and inflated metrics!')

# Check test metrics
with open('output/cnn/test_metrics.json', 'r') as f:
    metrics = json.load(f)

print('\n' + '='*70)
print('TEST SET PERFORMANCE (CNN)')
print('='*70)
print(f'ROC-AUC: {metrics["roc_auc"]:.4f}')
print(f'PR-AUC: {metrics["pr_auc"]:.4f}')
print(f'Precision: {metrics["precision"]:.4f}')
print(f'Recall: {metrics["recall"]:.4f}')
print(f'F1: {metrics["f1"]:.4f}')
print(f'\nConfusion Matrix:')
print(f'  TN: {metrics["true_negatives"]}, FP: {metrics["false_positives"]}')
print(f'  FN: {metrics["false_negatives"]}, TP: {metrics["true_positives"]}')

# Analyze window-level metrics
print('\n' + '='*70)
print('WINDOW-LEVEL ANALYSIS')
print('='*70)
total_windows = metrics["true_negatives"] + metrics["false_positives"] + metrics["false_negatives"] + metrics["true_positives"]
positive_windows = metrics["true_positives"] + metrics["false_negatives"]
print(f'Total test windows: {total_windows}')
print(f'Positive windows: {positive_windows} ({positive_windows/total_windows*100:.2f}%)')
print(f'Negative windows: {total_windows - positive_windows} ({(total_windows - positive_windows)/total_windows*100:.2f}%)')

# Check training history
with open('output/cnn/training_history.json', 'r') as f:
    history = json.load(f)

print('\n' + '='*70)
print('TRAINING HISTORY (CNN)')
print('='*70)
print(f'Epochs trained: {len(history["train_loss"])}')
print(f'Final train loss: {history["train_loss"][-1]:.6f}')
print(f'Final val loss: {history["val_loss"][-1]:.6f}')
print(f'Final val PR-AUC: {history["val_pr_auc"][-1]:.6f}')
print(f'\nLoss progression (last 5 epochs):')
for i in range(max(0, len(history["train_loss"])-5), len(history["train_loss"])):
    print(f'  Epoch {i+1:2d}: Train={history["train_loss"][i]:.6f}, Val={history["val_loss"][i]:.6f}, Val PR-AUC={history["val_pr_auc"][i]:.6f}')

# Check if model is overfitting
if len(history["train_loss"]) > 0:
    train_loss_trend = history["train_loss"][-1] - history["train_loss"][0]
    val_loss_trend = history["val_loss"][-1] - history["val_loss"][0]
    
    print('\n' + '='*70)
    print('OVERFITTING ANALYSIS')
    print('='*70)
    print(f'Train loss: {history["train_loss"][0]:.6f} → {history["train_loss"][-1]:.6f} (Δ={train_loss_trend:.6f})')
    print(f'Val loss: {history["val_loss"][0]:.6f} → {history["val_loss"][-1]:.6f} (Δ={val_loss_trend:.6f})')
    
    if history["val_loss"][-1] < 0.01 and metrics["roc_auc"] >= 0.999:
        print('\n⚠️  SUSPICIOUSLY PERFECT PERFORMANCE!')
        print('   Possible causes:')
        print('   1. Data leakage (train/test overlap)')
        print('   2. Task is too easy (attacks are very obvious)')
        print('   3. Window balancing creates artificial patterns')
        print('   4. Test set is not representative')
