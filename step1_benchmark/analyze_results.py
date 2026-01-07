"""
Results Analysis Script

Analyze and visualize results after training.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config


def analyze_results(output_dir=config.OUTPUT_DIR):
    """
    Analyze and visualize training results.
    
    Args:
        output_dir: Directory containing output files
    """
    print("="*80)
    print("Results Analysis")
    print("="*80)
    
    # ========================================================================
    # 1. Load and display test metrics
    # ========================================================================
    metrics_path = os.path.join(output_dir, 'test_metrics.json')
    
    if os.path.exists(metrics_path):
        print("\n[1] Test Set Performance Metrics")
        print("-" * 80)
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        print(f"\nPrimary Metrics:")
        print(f"  PR-AUC:  {metrics.get('pr_auc', 0):.4f}")
        print(f"  ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
        
        print(f"\nRecall @ Fixed FPR:")
        for fpr in config.FPR_THRESHOLDS:
            key = f'recall@fpr{fpr:.2f}'
            if key in metrics:
                print(f"  FPR={fpr:.2%}: Recall={metrics[key]:.4f}")
        
        print(f"\nFalse Positives:")
        print(f"  Avg per flight: {metrics.get('avg_fp_per_flight', 0):.2f}")
        
        if 'precision' in metrics:
            print(f"\nClassification @ threshold=0.5:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1']:.4f}")
    else:
        print(f"\n[1] Test metrics not found at {metrics_path}")
    
    # ========================================================================
    # 2. Load and plot training history
    # ========================================================================
    history_path = os.path.join(output_dir, 'training_history.json')
    
    if os.path.exists(history_path):
        print("\n[2] Training History")
        print("-" * 80)
        
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        axes[0].plot(epochs, history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(epochs, history['val_loss'], label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # PR-AUC plot
        axes[1].plot(epochs, history['val_pr_auc'], label='Val PR-AUC', marker='o', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('PR-AUC')
        axes[1].set_title('Validation PR-AUC')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n  Saved training history plot to {plot_path}")
        plt.close()
        
        print(f"\n  Training completed in {len(epochs)} epochs")
        print(f"  Best Val PR-AUC: {max(history['val_pr_auc']):.4f} at epoch {np.argmax(history['val_pr_auc'])+1}")
    else:
        print(f"\n[2] Training history not found at {history_path}")
    
    # ========================================================================
    # 3. Analyze attack injection statistics
    # ========================================================================
    print("\n[3] Attack Injection Statistics")
    print("-" * 80)
    
    for split_name in ['train', 'val', 'test']:
        attack_info_path = os.path.join(output_dir, f'{split_name}_attack_info.csv')
        
        if os.path.exists(attack_info_path):
            attack_df = pd.read_csv(attack_info_path)
            
            n_attacked = attack_df['attacked'].sum()
            n_total = len(attack_df)
            
            print(f"\n  {split_name.upper()} Set:")
            print(f"    Total flights: {n_total}")
            print(f"    Attacked flights: {n_attacked} ({n_attacked/n_total*100:.1f}%)")
            
            if n_attacked > 0:
                attacked_flights = attack_df[attack_df['attacked'] == True]
                
                print(f"    Attack magnitude distribution:")
                for mag in config.ATTACK_MAGNITUDES:
                    count = (attacked_flights['attack_magnitude'] == mag).sum()
                    print(f"      {mag}m: {count} flights")
    
    # ========================================================================
    # 4. Load and analyze predictions
    # ========================================================================
    predictions_path = os.path.join(output_dir, 'test_predictions.npz')
    
    if os.path.exists(predictions_path):
        print("\n[4] Prediction Analysis")
        print("-" * 80)
        
        data = np.load(predictions_path)
        predictions = data['predictions']
        targets = data['targets']
        flight_ids = data['flight_ids']
        
        print(f"\n  Total windows: {len(predictions)}")
        print(f"  Positive windows: {targets.sum():.0f} ({targets.mean()*100:.1f}%)")
        
        print(f"\n  Prediction statistics:")
        print(f"    Mean prediction: {predictions.mean():.4f}")
        print(f"    Std prediction:  {predictions.std():.4f}")
        print(f"    Min prediction:  {predictions.min():.4f}")
        print(f"    Max prediction:  {predictions.max():.4f}")
        
        # Per-flight analysis
        unique_flights = np.unique(flight_ids)
        flight_accuracies = []
        
        for flight_id in unique_flights:
            flight_mask = flight_ids == flight_id
            flight_preds = predictions[flight_mask]
            flight_targets = targets[flight_mask]
            
            if len(flight_targets) > 0:
                flight_acc = ((flight_preds >= 0.5) == flight_targets).mean()
                flight_accuracies.append(flight_acc)
        
        print(f"\n  Per-flight accuracy (threshold=0.5):")
        print(f"    Mean: {np.mean(flight_accuracies):.4f}")
        print(f"    Std:  {np.std(flight_accuracies):.4f}")
        print(f"    Min:  {np.min(flight_accuracies):.4f}")
        print(f"    Max:  {np.max(flight_accuracies):.4f}")
    
    # ========================================================================
    # 5. Display flight splits
    # ========================================================================
    splits_path = os.path.join(output_dir, 'flight_splits.json')
    
    if os.path.exists(splits_path):
        print("\n[5] Flight Splits")
        print("-" * 80)
        
        with open(splits_path, 'r') as f:
            splits = json.load(f)
        
        print(f"\n  Random seed: {splits['random_seed']}")
        print(f"  Train flights: {len(splits['train'])} flights")
        print(f"  Val flights:   {len(splits['val'])} flights")
        print(f"  Test flights:  {len(splits['test'])} flights")
        print(f"\n  First 10 test flights: {splits['test'][:10]}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    analyze_results()
