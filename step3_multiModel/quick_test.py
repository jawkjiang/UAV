"""
Quick Test Run - Demo Experiment

Runs a small subset of experiments to verify the system works correctly.
Tests: 2 attacks × 2 models = 4 experiments (~1 hour)
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


# Quick test configuration - only 4 experiments
ATTACK_TYPES = ['step', 'drift_ramp']  # Just 2 attacks
MODEL_TYPES = ['cnn', 'lstm']  # Just 2 models

# Attack parameters (same as main.py)
ATTACK_PARAMS = {
    'step': {},
    'drift_ramp': {'profile': 'ramp'},
}

ATTACK_TYPE_MAP = {
    'step': 'step',
    'drift_ramp': 'drift',
}


def run_single_experiment(attack_name, model_type, base_df, train_flights, val_flights, test_flights, output_dir):
    """Run a single experiment (same as main.py but simplified logging)."""
    
    print(f"\n{'='*60}")
    print(f"Running: {attack_name} + {model_type.upper()}")
    print(f"{'='*60}")
    
    # Create output directory
    exp_output_dir = os.path.join(output_dir, attack_name, model_type)
    os.makedirs(exp_output_dir, exist_ok=True)
    
    # Check if already done
    model_path = os.path.join(exp_output_dir, 'best_model.pth')
    if config.SKIP_EXISTING and os.path.exists(model_path):
        print(f"⏭ Skipping - already completed")
        metrics_path = os.path.join(exp_output_dir, 'test_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            return {'attack': attack_name, 'model': model_type, 'metrics': metrics, 'skipped': True}
    
    # Initialize injector
    injector = MultiAttackInjector(random_seed=config.RANDOM_SEED)
    
    # Inject attacks
    attack_type = ATTACK_TYPE_MAP[attack_name]
    attack_params = ATTACK_PARAMS[attack_name]
    
    print("Injecting attacks...")
    train_df = get_flight_subset(base_df, train_flights)
    train_df, train_attack_info = injector.inject_attacks_to_dataset(
        train_df, attack_type=attack_type, attack_ratio=0.5, **attack_params
    )
    
    val_df = get_flight_subset(base_df, val_flights)
    val_df, val_attack_info = injector.inject_attacks_to_dataset(
        val_df, attack_type=attack_type, attack_ratio=0.1, **attack_params
    )
    
    test_df = get_flight_subset(base_df, test_flights)
    test_df, test_attack_info = injector.inject_attacks_to_dataset(
        test_df, attack_type=attack_type, attack_ratio=0.1, **attack_params
    )
    
    # Generate labels
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
    
    # Train model
    print(f"Training {model_type} model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = create_model(
        model_type=model_type,
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
    
    print(f"✓ Results: AUC-ROC={metrics['roc_auc']:.4f}, F1={metrics['f1']:.4f}")
    
    return {'attack': attack_name, 'model': model_type, 'metrics': metrics, 'skipped': False}


def main():
    """Run quick test experiments."""
    
    print("="*60)
    print("QUICK TEST RUN - Demo Experiment")
    print("="*60)
    print(f"Attack types: {ATTACK_TYPES}")
    print(f"Model types: {MODEL_TYPES}")
    print(f"Total experiments: {len(ATTACK_TYPES) * len(MODEL_TYPES)}")
    print("="*60)
    
    # Setup
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and split data
    print("\nLoading data...")
    df = load_flights_data(data_path=config.DATA_PATH)
    train_flights, val_flights, test_flights = split_flights(df)
    
    df = compute_delta_t(df)
    df = convert_to_local_coordinates(df)
    
    # Run experiments
    all_results = []
    total = len(ATTACK_TYPES) * len(MODEL_TYPES)
    current = 0
    
    for attack_name, model_type in product(ATTACK_TYPES, MODEL_TYPES):
        current += 1
        print(f"\n{'#'*60}")
        print(f"# Experiment {current}/{total}")
        print(f"{'#'*60}")
        
        try:
            result = run_single_experiment(
                attack_name, model_type,
                df, train_flights, val_flights, test_flights,
                output_dir
            )
            all_results.append(result)
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate quick summary
    print("\n" + "="*60)
    print("QUICK TEST RESULTS")
    print("="*60)
    
    comparison_data = []
    for result in all_results:
        if result.get('metrics') is not None:
            comparison_data.append({
                'attack_type': result['attack'],
                'model_type': result['model'],
                'auc_roc': result['metrics']['roc_auc'],
                'f1_score': result['metrics']['f1']
            })
    
    if len(comparison_data) > 0:
        df_results = pd.DataFrame(comparison_data)
        print("\nResults Summary:")
        print(df_results.to_string(index=False))
        
        # Save
        df_results.to_csv(os.path.join(output_dir, 'quick_test_results.csv'), index=False)
        print(f"\nResults saved to: {os.path.join(output_dir, 'quick_test_results.csv')}")
    
    print("\n✓ Quick test completed successfully!")
    print("\nTo run full experiments (56 total), use: python main.py")


if __name__ == '__main__':
    main()
