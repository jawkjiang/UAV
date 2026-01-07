"""
Multi-Model Multi-Attack GPS Spoofing Detection

Tests all model architectures on all attack types.
Experiment matrix: 8 attacks × 7 models = 56 experiments
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import json
import pandas as pd
from itertools import product

# Import from current directory
import config_step3 as config
from multi_attack_injector import MultiAttackInjector
from model import create_model
from data_loader import load_flights_data, split_flights, get_flight_subset, compute_delta_t, convert_to_local_coordinates
from labeling import generate_point_labels
from feature_engineering import compute_all_features, get_feature_columns, normalize_features
from window_creation import create_windows_from_dataset, balance_windows, WindowDataset
from training import train_model
from evaluation import evaluate_model


# ============================================================================
# Experiment Configuration
# ============================================================================

# Define experiment matrix
ATTACK_TYPES = [
    'step',
    'drift_ramp',
    'drift_sigmoid',
    'delay',
    'replay_same_hard',
    'replay_other_soft',
    'takeover_step',
    'takeover_ramp'
]

MODEL_TYPES = [
    'cnn',
    'lstm',
    'bilstm',
    'gru',
    'cnn_lstm',
    'tcn',
    'transformer'
]

# Attack-specific parameters
ATTACK_PARAMS = {
    'step': {},
    'drift_ramp': {'profile': 'ramp'},
    'drift_sigmoid': {'profile': 'sigmoid'},
    'delay': {},
    'replay_same_hard': {'donor_source': 'same_flight_earlier', 'stitching': 'hard'},
    'replay_other_soft': {'donor_source': 'same_route_other_flight', 'stitching': 'soft'},
    'takeover_step': {'offset_profile': 'step'},
    'takeover_ramp': {'offset_profile': 'ramp'}
}

# Map attack names to injection types
ATTACK_TYPE_MAP = {
    'step': 'step',
    'drift_ramp': 'drift',
    'drift_sigmoid': 'drift',
    'delay': 'delay',
    'replay_same_hard': 'replay',
    'replay_other_soft': 'replay',
    'takeover_step': 'takeover',
    'takeover_ramp': 'takeover'
}


# ============================================================================
# Single Experiment Runner
# ============================================================================

def run_single_experiment(attack_name, model_type, base_df, train_flights, val_flights, test_flights, output_dir):
    """
    Run a single attack-model combination.
    
    CRITICAL: This function MUST NOT modify any data processing, training, or evaluation logic.
    ONLY the model creation is different.
    """
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {attack_name} + {model_type.upper()}")
    print(f"{'='*80}")
    
    # Create output directory
    exp_output_dir = os.path.join(output_dir, attack_name, model_type)
    os.makedirs(exp_output_dir, exist_ok=True)
    
    # Check if experiment already done
    model_path = os.path.join(exp_output_dir, 'best_model.pth')
    if config.SKIP_EXISTING and os.path.exists(model_path):
        print(f"Skipping - already completed")
        metrics_path = os.path.join(exp_output_dir, 'test_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            return {'attack': attack_name, 'model': model_type, 'metrics': metrics, 'skipped': True}
        else:
            print(f"Warning: Model exists but metrics not found, re-running...")
    
    # Initialize injector
    injector = MultiAttackInjector(random_seed=config.RANDOM_SEED)
    
    # Inject attacks
    print(f"Injecting {attack_name} attacks...")
    attack_type = ATTACK_TYPE_MAP[attack_name]
    attack_params = ATTACK_PARAMS[attack_name]
    
    train_df = get_flight_subset(base_df, train_flights)
    train_df, train_attack_info = injector.inject_attacks_to_dataset(
        train_df, attack_type=attack_type, attack_ratio=0.5, **attack_params
    )
    train_attack_info.to_csv(os.path.join(exp_output_dir, 'train_attack_info.csv'), index=False)
    
    val_df = get_flight_subset(base_df, val_flights)
    val_df, val_attack_info = injector.inject_attacks_to_dataset(
        val_df, attack_type=attack_type, attack_ratio=0.1, **attack_params
    )
    val_attack_info.to_csv(os.path.join(exp_output_dir, 'val_attack_info.csv'), index=False)
    
    test_df = get_flight_subset(base_df, test_flights)
    test_df, test_attack_info = injector.inject_attacks_to_dataset(
        test_df, attack_type=attack_type, attack_ratio=0.1, **attack_params
    )
    test_attack_info.to_csv(os.path.join(exp_output_dir, 'test_attack_info.csv'), index=False)
    
    # Generate labels
    print("Generating labels...")
    train_df = generate_point_labels(train_df, train_attack_info)
    val_df = generate_point_labels(val_df, val_attack_info)
    test_df = generate_point_labels(test_df, test_attack_info)
    
    # Feature engineering
    print("Computing features...")
    train_df = compute_all_features(train_df)
    val_df = compute_all_features(val_df)
    test_df = compute_all_features(test_df)
    
    feature_cols = get_feature_columns()
    train_df, val_df, test_df, normalization_stats = normalize_features(
        train_df, val_df, test_df, feature_cols
    )
    
    with open(os.path.join(exp_output_dir, 'normalization_stats.json'), 'w') as f:
        json.dump(normalization_stats, f, indent=2)
    
    # Create windows
    print("Creating windows...")
    train_windows, train_labels, train_flight_ids = create_windows_from_dataset(train_df, feature_cols)
    val_windows, val_labels, val_flight_ids = create_windows_from_dataset(val_df, feature_cols)
    test_windows, test_labels, test_flight_ids = create_windows_from_dataset(test_df, feature_cols)
    
    train_windows, train_labels, train_flight_ids = balance_windows(
        train_windows, train_labels, train_flight_ids,
        target_pos_ratio=0.35, random_seed=config.RANDOM_SEED
    )
    val_windows, val_labels, val_flight_ids = balance_windows(
        val_windows, val_labels, val_flight_ids,
        target_pos_ratio=0.03, random_seed=config.RANDOM_SEED
    )
    test_windows, test_labels, test_flight_ids = balance_windows(
        test_windows, test_labels, test_flight_ids,
        target_pos_ratio=0.03, random_seed=config.RANDOM_SEED
    )
    
    # Create dataloaders
    train_dataset = WindowDataset(train_windows, train_labels)
    val_dataset = WindowDataset(val_windows, val_labels)
    test_dataset = WindowDataset(test_windows, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Train model - ONLY DIFFERENCE IS MODEL TYPE
    print(f"Training {model_type} model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = create_model(
        model_type=model_type,  # <-- ONLY THIS CHANGES
        n_features=len(feature_cols),
        window_size=config.WINDOW_SIZE,
        dropout=0.3
    )
    
    pos_weight = train_dataset.get_positive_weight()
    
    history = train_model(
        model, train_loader, val_loader,
        pos_weight=pos_weight,
        num_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        patience=config.EARLY_STOPPING_PATIENCE,
        device=device,
        save_path=model_path
    )
    
    with open(os.path.join(exp_output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Evaluate
    print("Evaluating...")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    metrics, predictions, targets = evaluate_model(model, test_loader, test_flight_ids, device=device)
    
    with open(os.path.join(exp_output_dir, 'test_metrics.json'), 'w') as f:
        metrics_serializable = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                               for k, v in metrics.items()}
        json.dump(metrics_serializable, f, indent=2)
    
    np.savez(os.path.join(exp_output_dir, 'test_predictions.npz'),
             predictions=predictions, targets=targets, flight_ids=test_flight_ids)
    
    print(f"Results: AUC-ROC={metrics['roc_auc']:.4f}, F1={metrics['f1']:.4f}")
    
    return {'attack': attack_name, 'model': model_type, 'metrics': metrics, 'skipped': False}


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run full experiment matrix."""
    
    print("="*80)
    print("Multi-Model Multi-Attack GPS Spoofing Detection")
    print("="*80)
    print(f"Attack types: {len(ATTACK_TYPES)}")
    print(f"Model types: {len(MODEL_TYPES)}")
    print(f"Total experiments: {len(ATTACK_TYPES) * len(MODEL_TYPES)}")
    print("="*80)
    
    # Setup
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and split data (ONCE, shared across all experiments)
    print("\nLoading and preprocessing data...")
    df = load_flights_data(data_path=config.DATA_PATH)
    train_flights, val_flights, test_flights = split_flights(df)
    
    print(f"Train flights: {len(train_flights)}")
    print(f"Val flights: {len(val_flights)}")
    print(f"Test flights: {len(test_flights)}")
    
    with open(os.path.join(output_dir, 'flight_splits.json'), 'w') as f:
        json.dump({
            'train': train_flights,
            'val': val_flights,
            'test': test_flights,
            'random_seed': config.RANDOM_SEED
        }, f, indent=2)
    
    df = compute_delta_t(df)
    df = convert_to_local_coordinates(df)
    
    # Run experiment matrix
    all_results = []
    total_experiments = len(ATTACK_TYPES) * len(MODEL_TYPES)
    current_experiment = 0
    
    for attack_name, model_type in product(ATTACK_TYPES, MODEL_TYPES):
        current_experiment += 1
        print(f"\n\n{'#'*80}")
        print(f"# EXPERIMENT {current_experiment}/{total_experiments}: {attack_name} + {model_type}")
        print(f"{'#'*80}")
        
        try:
            result = run_single_experiment(
                attack_name, model_type,
                df, train_flights, val_flights, test_flights,
                output_dir
            )
            all_results.append(result)
        except Exception as e:
            print(f"ERROR in experiment {attack_name} + {model_type}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'attack': attack_name, 
                'model': model_type, 
                'metrics': None, 
                'error': str(e)
            })
    
    # Generate comparison tables
    print("\n\n" + "="*80)
    print("Generating comparison tables...")
    print("="*80)
    
    # Overall comparison
    comparison_data = []
    for result in all_results:
        if result.get('metrics') is not None:
            comparison_data.append({
                'attack_type': result['attack'],
                'model_type': result['model'],
                'auc_roc': result['metrics']['roc_auc'],
                'auc_pr': result['metrics']['pr_auc'],
                'f1_score': result['metrics']['f1'],
                'precision': result['metrics']['precision'],
                'recall': result['metrics']['recall']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(output_dir, 'overall_comparison.csv'), index=False)
    
    # Print summary
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {len([r for r in all_results if r.get('metrics') is not None])}")
    print(f"Failed: {len([r for r in all_results if r.get('metrics') is None])}")
    
    if len(comparison_data) > 0:
        print("\nTop 5 configurations by AUC-ROC:")
        top5 = comparison_df.nlargest(5, 'auc_roc')
        print(top5.to_string(index=False))
        
        # Per-model average
        print("\n\nAverage performance by model:")
        model_avg = comparison_df.groupby('model_type')[['auc_roc', 'f1_score']].mean()
        model_avg = model_avg.sort_values('auc_roc', ascending=False)
        print(model_avg.to_string())
        
        # Per-attack average
        print("\n\nAverage performance by attack:")
        attack_avg = comparison_df.groupby('attack_type')[['auc_roc', 'f1_score']].mean()
        attack_avg = attack_avg.sort_values('auc_roc', ascending=False)
        print(attack_avg.to_string())
    
    print("\n\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)
    print(f"\nResults saved to: {os.path.abspath(output_dir)}")
    print(f"Overall comparison: {os.path.join(output_dir, 'overall_comparison.csv')}")


if __name__ == '__main__':
    main()
