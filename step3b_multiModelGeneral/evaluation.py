"""
Evaluation Module

Evaluate GPS spoofing detection model on test set.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    precision_recall_curve, auc, roc_curve, roc_auc_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import config


def compute_recall_at_fpr(targets, predictions, fpr_threshold=0.01):
    """
    Compute recall at a fixed false positive rate.
    
    Args:
        targets: Ground truth labels
        predictions: Predicted probabilities
        fpr_threshold: Target FPR threshold
    
    Returns:
        Recall at specified FPR
    """
    fpr, tpr, thresholds = roc_curve(targets, predictions)
    
    # Find threshold that achieves closest FPR to target
    idx = np.argmin(np.abs(fpr - fpr_threshold))
    recall = tpr[idx]
    actual_fpr = fpr[idx]
    threshold = thresholds[idx]
    
    return recall, actual_fpr, threshold


def compute_avg_false_positives_per_flight(targets, predictions, flight_ids, threshold=0.5):
    """
    Compute average number of false positives per flight.
    
    Args:
        targets: Ground truth labels
        predictions: Predicted probabilities
        flight_ids: Flight ID for each window
        threshold: Classification threshold
    
    Returns:
        Average false positives per flight
    """
    # Binary predictions
    pred_labels = (predictions >= threshold).astype(int)
    
    # Count false positives per flight
    false_positives = {}
    
    for flight_id in np.unique(flight_ids):
        flight_mask = flight_ids == flight_id
        flight_targets = targets[flight_mask]
        flight_preds = pred_labels[flight_mask]
        
        # False positive: predicted 1, actual 0
        fp_count = ((flight_preds == 1) & (flight_targets == 0)).sum()
        false_positives[flight_id] = fp_count
    
    avg_fp = np.mean(list(false_positives.values()))
    
    return avg_fp, false_positives


def evaluate_model(model, test_loader, flight_ids, device='cuda'):
    """
    Comprehensive evaluation of model on test set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        flight_ids: Flight IDs for test windows
        device: Device to use
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            
            all_predictions.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    predictions = np.concatenate(all_predictions).flatten()
    targets = np.concatenate(all_targets).flatten()
    
    # Compute metrics
    metrics = {}
    
    # PR-AUC
    precision, recall, pr_thresholds = precision_recall_curve(targets, predictions)
    pr_auc = auc(recall, precision)
    metrics['pr_auc'] = pr_auc
    
    # ROC-AUC
    roc_auc = roc_auc_score(targets, predictions)
    metrics['roc_auc'] = roc_auc
    
    # Recall @ fixed FPR
    for fpr_threshold in config.FPR_THRESHOLDS:
        recall_at_fpr, actual_fpr, threshold = compute_recall_at_fpr(
            targets, predictions, fpr_threshold
        )
        metrics[f'recall@fpr{fpr_threshold:.2f}'] = recall_at_fpr
        metrics[f'actual_fpr@fpr{fpr_threshold:.2f}'] = actual_fpr
        metrics[f'threshold@fpr{fpr_threshold:.2f}'] = threshold
    
    # Average false positives per flight
    # Use threshold from FPR=0.01
    threshold_001 = metrics.get('threshold@fpr0.01', 0.5)
    avg_fp, fp_dict = compute_avg_false_positives_per_flight(
        targets, predictions, flight_ids, threshold_001
    )
    metrics['avg_fp_per_flight'] = avg_fp
    
    # Classification metrics at threshold 0.5
    pred_labels = (predictions >= 0.5).astype(int)
    cm = confusion_matrix(targets, pred_labels)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        metrics['true_positives'] = tp
        
        # Precision, Recall, F1
        precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score) \
                   if (precision_score + recall_score) > 0 else 0
        
        metrics['precision'] = precision_score
        metrics['recall'] = recall_score
        metrics['f1'] = f1_score
    
    return metrics, predictions, targets


def print_evaluation_report(metrics):
    """
    Print formatted evaluation report.
    
    Args:
        metrics: Dictionary of evaluation metrics
    """
    print("\n" + "="*60)
    print("GPS Spoofing Detection - Test Set Evaluation")
    print("="*60)
    
    print(f"\n1. Area Under Curves:")
    print(f"   PR-AUC:  {metrics['pr_auc']:.4f}")
    print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
    
    print(f"\n2. Recall @ Fixed FPR:")
    for fpr_threshold in config.FPR_THRESHOLDS:
        recall = metrics[f'recall@fpr{fpr_threshold:.2f}']
        actual_fpr = metrics[f'actual_fpr@fpr{fpr_threshold:.2f}']
        threshold = metrics[f'threshold@fpr{fpr_threshold:.2f}']
        print(f"   FPR={fpr_threshold:.2f}: Recall={recall:.4f} "
              f"(actual FPR={actual_fpr:.4f}, threshold={threshold:.4f})")
    
    print(f"\n3. False Positives:")
    print(f"   Avg FP per flight: {metrics['avg_fp_per_flight']:.2f}")
    
    if 'precision' in metrics:
        print(f"\n4. Classification Metrics (threshold=0.5):")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1']:.4f}")
        
        print(f"\n5. Confusion Matrix:")
        print(f"   TN={metrics['true_negatives']}, FP={metrics['false_positives']}")
        print(f"   FN={metrics['false_negatives']}, TP={metrics['true_positives']}")
    
    print("="*60 + "\n")


def plot_evaluation_curves(targets, predictions, save_path=None):
    """
    Plot PR and ROC curves.
    
    Args:
        targets: Ground truth labels
        predictions: Predicted probabilities
        save_path: Path to save plot (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(targets, predictions)
    pr_auc = auc(recall, precision)
    
    axes[0].plot(recall, precision, label=f'PR-AUC = {pr_auc:.4f}')
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].set_title('Precision-Recall Curve')
    axes[0].legend()
    axes[0].grid(True)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(targets, predictions)
    roc_auc = auc(fpr, tpr)
    
    axes[1].plot(fpr, tpr, label=f'ROC-AUC = {roc_auc:.4f}')
    axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved evaluation plots to {save_path}")
    
    plt.close()


import torch


if __name__ == '__main__':
    import os
    import json
    import pickle
    from torch.utils.data import DataLoader
    from model import GPSSpoofingDetector
    from window_creation import WindowDataset
    
    print("="*80)
    print("Model Evaluation")
    print("="*80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load windows data
    print("\n[1] Loading test data...")
    with open(os.path.join(config.OUTPUT_DIR, 'windows_data.pkl'), 'rb') as f:
        windows_data = pickle.load(f)
    
    test_windows = windows_data['test']['windows']
    test_labels = windows_data['test']['labels']
    test_flight_ids = windows_data['test']['flight_ids']
    
    print(f"    Test: {len(test_labels)} windows from {len(np.unique(test_flight_ids))} flights")
    print(f"    Positive ratio: {test_labels.mean():.2%}")
    
    # Create dataset and loader
    test_dataset = WindowDataset(test_windows, test_labels)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Load model
    print("\n[2] Loading trained model...")
    n_features = test_windows.shape[2]
    model = GPSSpoofingDetector(n_features=n_features)
    
    checkpoint_path = os.path.join(config.OUTPUT_DIR, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Model checkpoint not found at {checkpoint_path}")
        print("Please run training.py first!")
        exit(1)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"    ✓ Loaded model from epoch {checkpoint['epoch']}")
    print(f"    ✓ Validation PR-AUC: {checkpoint['val_pr_auc']:.4f}")
    
    # Evaluate
    print("\n[3] Evaluating on test set...")
    metrics, predictions, targets = evaluate_model(
        model, test_loader, test_flight_ids, device
    )
    
    # Print results
    print_evaluation_report(metrics)
    
    # Save metrics
    print("\n[4] Saving evaluation results...")
    
    # Convert numpy types to Python types for JSON serialization
    metrics_json = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer, np.floating)):
            metrics_json[key] = float(value)
        else:
            metrics_json[key] = value
    
    with open(os.path.join(config.OUTPUT_DIR, 'test_metrics.json'), 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    print(f"    ✓ Saved metrics to {config.OUTPUT_DIR}/test_metrics.json")
    
    # Save predictions
    predictions_data = {
        'predictions': predictions.tolist(),
        'targets': targets.tolist(),
        'flight_ids': test_flight_ids.tolist()
    }
    
    with open(os.path.join(config.OUTPUT_DIR, 'test_predictions.json'), 'w') as f:
        json.dump(predictions_data, f)
    
    print(f"    ✓ Saved predictions to {config.OUTPUT_DIR}/test_predictions.json")
    
    # Plot curves
    print("\n[5] Generating evaluation plots...")
    plot_evaluation_curves(
        targets, predictions,
        save_path=os.path.join(config.OUTPUT_DIR, 'evaluation_curves.png')
    )
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)
