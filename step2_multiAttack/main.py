"""
Main Execution Script for Multi-Attack GPS Spoofing Detection

Runs all attack types and compares detection performance:
- Drift Spoofing (ramp & sigmoid)
- Delay (fixed delay)
- Replay (segment replay)
- Consistent Takeover
"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import json
import pandas as pd

# Add step1_benchmark to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'step1_benchmark'))

# Import from step1_benchmark
from data_loader import (
    load_flights_data, split_flights, get_flight_subset,
    compute_delta_t, convert_to_local_coordinates
)
from labeling import generate_point_labels
from feature_engineering import (
    compute_all_features, get_feature_columns, normalize_features
)
from window_creation import (
    create_windows_from_dataset, balance_windows, WindowDataset
)
from model import create_model
from training import train_model
from evaluation import evaluate_model
import config  # step1's config

# Import step2's own config and injector
import config_step2
from multi_attack_injector import MultiAttackInjector


def set_random_seeds(seed=config_step2.RANDOM_SEED):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_single_attack_experiment(attack_type: str,
                                 base_df: pd.DataFrame,
                                 train_flights: list,
                                 val_flights: list,
                                 test_flights: list,
                                 output_dir: str,
                                 experiment_name: str = None,
                                 **attack_params):
    """
    Run complete pipeline for a single attack type.
    
    Args:
        experiment_name: Unique name for this experiment. If None, uses attack_type.
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Use experiment_name if provided, otherwise use attack_type
    exp_name = experiment_name if experiment_name else attack_type
    
    print("\n" + "="*80)
    print(f"RUNNING EXPERIMENT: {exp_name.upper()}")
    print("="*80)
    
    # Create attack-specific output directory
    attack_output_dir = os.path.join(output_dir, exp_name)
    os.makedirs(attack_output_dir, exist_ok=True)
    
    # Check if we should skip this experiment
    model_path = os.path.join(attack_output_dir, 'best_model.pth')
    if config_step2.SKIP_EXISTING and os.path.exists(model_path):
        print(f"\n[{exp_name}] Skipping - best_model.pth already exists")
        
        # Load existing metrics if available
        metrics_path = os.path.join(attack_output_dir, 'test_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            print(f"  Loaded existing metrics:")
            print(f"    AUC-ROC: {metrics['roc_auc']:.4f}")
            print(f"    AUC-PR: {metrics['pr_auc']:.4f}")
            print(f"    F1 Score: {metrics['f1']:.4f}")
            
            return {
                'attack_type': exp_name,
                'metrics': metrics,
                'output_dir': attack_output_dir,
                'skipped': True
            }
        else:
            print(f"  Warning: Model exists but metrics not found. Re-running experiment.")
    
    # Initialize injector
    injector = MultiAttackInjector(random_seed=config_step2.RANDOM_SEED)
    
    # ========================================================================
    # Inject attacks to each split
    # ========================================================================
    print(f"\n[{exp_name}] Injecting attacks...")
    
    # Training set: ~50% flights attacked
    train_df = get_flight_subset(base_df, train_flights)
    train_df, train_attack_info = injector.inject_attacks_to_dataset(
        train_df, attack_type=attack_type, attack_ratio=0.5, **attack_params
    )
    train_attack_info.to_csv(
        os.path.join(attack_output_dir, 'train_attack_info.csv'), index=False
    )
    
    # Validation set: ~10% flights attacked (aligned with step1)
    val_df = get_flight_subset(base_df, val_flights)
    val_df, val_attack_info = injector.inject_attacks_to_dataset(
        val_df, attack_type=attack_type, attack_ratio=0.1, **attack_params
    )
    val_attack_info.to_csv(
        os.path.join(attack_output_dir, 'val_attack_info.csv'), index=False
    )
    
    # Test set: ~10% flights attacked (aligned with step1)
    test_df = get_flight_subset(base_df, test_flights)
    test_df, test_attack_info = injector.inject_attacks_to_dataset(
        test_df, attack_type=attack_type, attack_ratio=0.1, **attack_params
    )
    test_attack_info.to_csv(
        os.path.join(attack_output_dir, 'test_attack_info.csv'), index=False
    )
    
    # ========================================================================
    # Generate labels
    # ========================================================================
    print(f"\n[{exp_name}] Generating point labels...")
    
    train_df = generate_point_labels(train_df, train_attack_info)
    val_df = generate_point_labels(val_df, val_attack_info)
    test_df = generate_point_labels(test_df, test_attack_info)
    
    # ========================================================================
    # Feature engineering
    # ========================================================================
    print(f"\n[{exp_name}] Computing features...")
    
    train_df = compute_all_features(train_df)
    val_df = compute_all_features(val_df)
    test_df = compute_all_features(test_df)
    
    # Normalize features
    feature_cols = get_feature_columns()
    train_df, val_df, test_df, normalization_stats = normalize_features(
        train_df, val_df, test_df, feature_cols
    )
    
    # Save normalization stats
    with open(os.path.join(attack_output_dir, 'normalization_stats.json'), 'w') as f:
        json.dump(normalization_stats, f, indent=2)
    
    # ========================================================================
    # Create windows
    # ========================================================================
    print(f"\n[{exp_name}] Creating windows...")
    
    train_windows, train_labels, train_flight_ids = create_windows_from_dataset(
        train_df, feature_cols
    )
    val_windows, val_labels, val_flight_ids = create_windows_from_dataset(
        val_df, feature_cols
    )
    test_windows, test_labels, test_flight_ids = create_windows_from_dataset(
        test_df, feature_cols
    )
    
    # Balance training windows (target ~30-40% positive)
    train_windows, train_labels, train_flight_ids = balance_windows(
        train_windows, train_labels, train_flight_ids,
        target_pos_ratio=0.35, random_seed=config_step2.RANDOM_SEED
    )
    
    # Balance validation/test windows (target ~1-5% positive)
    val_windows, val_labels, val_flight_ids = balance_windows(
        val_windows, val_labels, val_flight_ids,
        target_pos_ratio=0.03, random_seed=config_step2.RANDOM_SEED
    )
    test_windows, test_labels, test_flight_ids = balance_windows(
        test_windows, test_labels, test_flight_ids,
        target_pos_ratio=0.03, random_seed=config_step2.RANDOM_SEED
    )
    
    print(f"Train windows: {len(train_windows)}")
    print(f"Val windows: {len(val_windows)}")
    print(f"Test windows: {len(test_windows)}")
    
    # ========================================================================
    # Create dataloaders
    # ========================================================================
    train_dataset = WindowDataset(train_windows, train_labels)
    val_dataset = WindowDataset(val_windows, val_labels)
    test_dataset = WindowDataset(test_windows, test_labels)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config_step2.BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config_step2.BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config_step2.BATCH_SIZE, shuffle=False
    )
    
    # ========================================================================
    # Train model
    # ========================================================================
    print(f"\n[{exp_name}] Training model...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = create_model(
        model_type='cnn',
        n_features=len(feature_cols),
        window_size=config_step2.WINDOW_SIZE,
        dropout=0.3
    )
    
    pos_weight = train_dataset.get_positive_weight()
    print(f"Positive sample weight for loss: {pos_weight:.2f}")
    
    model_path = os.path.join(attack_output_dir, 'best_model.pth')
    
    history = train_model(
        model, train_loader, val_loader,
        pos_weight=pos_weight,
        num_epochs=config_step2.NUM_EPOCHS,
        learning_rate=config_step2.LEARNING_RATE,
        patience=config_step2.EARLY_STOPPING_PATIENCE,
        device=device,
        save_path=model_path
    )
    
    # Save training history
    with open(os.path.join(attack_output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # ========================================================================
    # Evaluate on test set
    # ========================================================================
    print(f"\n[{exp_name}] Evaluating on test set...")
    
    # Load best model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    
    # Evaluate
    metrics, predictions, targets = evaluate_model(
        model, test_loader, test_flight_ids, device=device
    )
    
    # Save metrics
    with open(os.path.join(attack_output_dir, 'test_metrics.json'), 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        metrics_serializable = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                               for k, v in metrics.items()}
        json.dump(metrics_serializable, f, indent=2)
    
    # Save predictions
    np.savez(os.path.join(attack_output_dir, 'test_predictions.npz'),
             predictions=predictions,
             targets=targets,
             flight_ids=test_flight_ids)
    
    print(f"\n[{exp_name}] Results:")
    print(f"  AUC-ROC: {metrics['roc_auc']:.4f}")
    print(f"  AUC-PR: {metrics['pr_auc']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    
    return {
        'attack_type': exp_name,
        'metrics': metrics,
        'output_dir': attack_output_dir,
        'skipped': False
    }


def main():
    """Main execution pipeline for all attack types."""
    
    print("="*80)
    print("Multi-Attack GPS Spoofing Detection - Complete Pipeline")
    print("="*80)
    
    # Set random seeds
    set_random_seeds(config_step2.RANDOM_SEED)
    
    # Create output directories
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    
    # ========================================================================
    # Step 1: Load and split data (shared across all experiments)
    # ========================================================================
    print("\n[Step 1] Loading and splitting data...")
    
    df = load_flights_data(data_path=config_step2.DATA_PATH)
    train_flights, val_flights, test_flights = split_flights(df)
    
    # Save flight splits
    flight_splits = {
        'train': train_flights,
        'val': val_flights,
        'test': test_flights,
        'random_seed': config_step2.RANDOM_SEED
    }
    
    with open(os.path.join(output_dir, 'flight_splits.json'), 'w') as f:
        json.dump(flight_splits, f, indent=2)
    
    # ========================================================================
    # Step 2: Preprocess base data
    # ========================================================================
    print("\n[Step 2] Preprocessing base data...")
    
    df = compute_delta_t(df)
    df = convert_to_local_coordinates(df)
    
    # ========================================================================
    # Step 3: Run experiments for each attack type
    # ========================================================================
    
    all_results = []
    
    # Baseline: Step Attack (from step1_benchmark)
    result = run_single_attack_experiment(
        'step',
        df, train_flights, val_flights, test_flights,
        output_dir
    )
    all_results.append(result)
    
    # Attack A: Drift (ramp)
    result = run_single_attack_experiment(
        'drift',
        df, train_flights, val_flights, test_flights,
        output_dir,
        experiment_name='drift_ramp',
        profile='ramp'
    )
    all_results.append(result)
    
    # Attack A: Drift (sigmoid)
    result = run_single_attack_experiment(
        'drift',
        df, train_flights, val_flights, test_flights,
        output_dir,
        experiment_name='drift_sigmoid',
        profile='sigmoid'
    )
    all_results.append(result)
    
    # Attack B1: Delay
    result = run_single_attack_experiment(
        'delay',
        df, train_flights, val_flights, test_flights,
        output_dir
    )
    all_results.append(result)
    
    # Attack B2: Replay (same flight)
    result = run_single_attack_experiment(
        'replay',
        df, train_flights, val_flights, test_flights,
        output_dir,
        experiment_name='replay_same_hard',
        donor_source='same_flight_earlier',
        stitching='hard'
    )
    all_results.append(result)
    
    # Attack B2: Replay (other flight, soft)
    result = run_single_attack_experiment(
        'replay',
        df, train_flights, val_flights, test_flights,
        output_dir,
        experiment_name='replay_other_soft',
        donor_source='same_route_other_flight',
        stitching='soft'
    )
    all_results.append(result)
    
    # Attack C: Takeover (step)
    result = run_single_attack_experiment(
        'takeover',
        df, train_flights, val_flights, test_flights,
        output_dir,
        experiment_name='takeover_step',
        offset_profile='step'
    )
    all_results.append(result)
    
    # Attack C: Takeover (ramp)
    result = run_single_attack_experiment(
        'takeover',
        df, train_flights, val_flights, test_flights,
        output_dir,
        experiment_name='takeover_ramp',
        offset_profile='ramp'
    )
    all_results.append(result)
    
    # ========================================================================
    # Step 4: Save comparison results
    # ========================================================================
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    comparison_df = pd.DataFrame([
        {
            'attack_type': r['attack_type'],
            'auc_roc': r['metrics']['roc_auc'],
            'auc_pr': r['metrics']['pr_auc'],
            'f1_score': r['metrics']['f1'],
            'precision': r['metrics']['precision'],
            'recall': r['metrics']['recall'],
        }
        for r in all_results
    ])
    
    comparison_df = comparison_df.sort_values('auc_roc', ascending=False)
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_df.to_csv(os.path.join(output_dir, 'attack_comparison.csv'), index=False)
    
    print(f"\n✓ All results saved to: {output_dir}")
    print(f"✓ Comparison saved to: {os.path.join(output_dir, 'attack_comparison.csv')}")


if __name__ == '__main__':
    main()
