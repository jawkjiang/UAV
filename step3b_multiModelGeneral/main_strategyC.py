"""
Multi-Model Training with Strategy C (Conservative Hybrid)

Strategy C: Conservative Hybrid with Flight Reuse
- Train/Val: Use flight reuse (each flight × 7 versions: 6 attacks + 1 normal)
- Test: No reuse, stratified sampling ensures all attack types represented

This provides:
- Large training set for better learning
- Realistic test evaluation (flight-level independence)
- All attack types properly tested
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import json
import pandas as pd
from datetime import datetime

# Import modules
import config
from stratified_split import stratified_split_with_reuse, expand_flights_with_all_attacks, assign_attacks_to_test_flights
from flight_reuse_injector import FlightReuseInjector
from model import create_model
from data_loader import load_flights_data, get_flight_subset, compute_delta_t, convert_to_local_coordinates
from labeling import generate_point_labels
from feature_engineering import compute_all_features, get_feature_columns, normalize_features
from window_creation import create_windows_from_dataset, balance_windows, WindowDataset
from training import train_model
from evaluation import evaluate_model


def train_single_model(model_type, base_df, train_pairs, val_pairs, test_flights, 
                      test_allocation, output_dir):
    """
    Train a single model using Strategy C.
    
    Args:
        model_type: Model architecture name
        base_df: Base DataFrame with all flights
        train_pairs: List of (flight_id, attack_type) tuples for training
        val_pairs: List of (flight_id, attack_type) tuples for validation
        test_flights: List of test flight IDs (no reuse)
        test_allocation: Dict mapping attack_type -> flight_ids for test
        output_dir: Output directory for this model
    
    Returns:
        Dictionary with test metrics
    """
    
    print(f"\n{'='*80}")
    print(f"TRAINING MODEL: {model_type.upper()}")
    print(f"{'='*80}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if already trained
    model_path = os.path.join(output_dir, 'best_model.pth')
    if config.SKIP_EXISTING and os.path.exists(model_path):
        print(f"Model already trained, loading results...")
        metrics_path = os.path.join(output_dir, 'test_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            return {'model': model_type, 'metrics': metrics, 'skipped': True}
    
    # Initialize injector
    injector = FlightReuseInjector(random_seed=config.RANDOM_SEED)
    
    # ========================================================================
    # INJECT ATTACKS WITH FLIGHT REUSE
    # ========================================================================
    
    print(f"\n{'='*60}")
    print(f"TRAINING SET - Flight Reuse Injection")
    print(f"{'='*60}")
    train_df, train_attack_info = injector.inject_with_flight_reuse(
        base_df, 
        train_pairs,
        verbose=True
    )
    train_attack_info.to_csv(os.path.join(output_dir, 'train_attack_info.csv'), index=False)
    
    print(f"\n{'='*60}")
    print(f"VALIDATION SET - Flight Reuse Injection")
    print(f"{'='*60}")
    val_df, val_attack_info = injector.inject_with_flight_reuse(
        base_df,
        val_pairs,
        verbose=True
    )
    val_attack_info.to_csv(os.path.join(output_dir, 'val_attack_info.csv'), index=False)
    
    print(f"\n{'='*60}")
    print(f"TEST SET - Stratified Injection (No Reuse)")
    print(f"{'='*60}")
    test_df = get_flight_subset(base_df, test_flights)
    test_df, test_attack_info = injector.inject_stratified_attacks_to_test(
        test_df,
        test_allocation,
        verbose=True
    )
    test_attack_info.to_csv(os.path.join(output_dir, 'test_attack_info.csv'), index=False)
    
    # ========================================================================
    # PREPROCESSING
    # ========================================================================
    
    print(f"\n{'='*60}")
    print(f"Preprocessing")
    print(f"{'='*60}")
    
    # Compute features for all sets
    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"\n{name} set:")
        df = compute_delta_t(df)
        df = convert_to_local_coordinates(df)
        df = generate_point_labels(df, train_attack_info if name == 'Train' 
                                   else val_attack_info if name == 'Val' 
                                   else test_attack_info)
        df = compute_all_features(df)
        
        if name == 'Train':
            train_df = df
        elif name == 'Val':
            val_df = df
        else:
            test_df = df
    
    # Get feature columns
    feature_cols = get_feature_columns()
    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")
    
    # Normalize features (all three sets at once using train statistics)
    train_df, val_df, test_df, norm_stats = normalize_features(
        train_df, val_df, test_df, feature_cols
    )
    
    # Save normalization stats
    with open(os.path.join(output_dir, 'normalization_stats.json'), 'w') as f:
        json.dump(norm_stats, f, indent=2)
    
    # ========================================================================
    # CREATE WINDOWS
    # ========================================================================
    
    print(f"\n{'='*60}")
    print(f"Creating Windows")
    print(f"{'='*60}")
    
    # Create windows
    train_windows, train_labels, train_flight_ids = create_windows_from_dataset(
        train_df, feature_cols, window_size=config.WINDOW_SIZE
    )
    val_windows, val_labels, val_flight_ids = create_windows_from_dataset(
        val_df, feature_cols, window_size=config.WINDOW_SIZE
    )
    test_windows, test_labels, test_flight_ids = create_windows_from_dataset(
        test_df, feature_cols, window_size=config.WINDOW_SIZE
    )
    
    # Balance windows
    train_windows, train_labels, train_flight_ids = balance_windows(
        train_windows, train_labels, train_flight_ids,
        target_pos_ratio=config.TRAIN_POS_RATIO,
        random_seed=config.RANDOM_SEED
    )
    val_windows, val_labels, val_flight_ids = balance_windows(
        val_windows, val_labels, val_flight_ids,
        target_pos_ratio=config.VAL_POS_RATIO,
        random_seed=config.RANDOM_SEED
    )
    test_windows, test_labels, test_flight_ids = balance_windows(
        test_windows, test_labels, test_flight_ids,
        target_pos_ratio=config.TEST_POS_RATIO,
        random_seed=config.RANDOM_SEED
    )
    
    # Create datasets
    train_dataset = WindowDataset(train_windows, train_labels)
    val_dataset = WindowDataset(val_windows, val_labels)
    test_dataset = WindowDataset(test_windows, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"\nDatasets created:")
    print(f"  Train: {len(train_dataset)} windows")
    print(f"  Val: {len(val_dataset)} windows")
    print(f"  Test: {len(test_dataset)} windows")
    
    # ========================================================================
    # TRAIN MODEL
    # ========================================================================
    
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()}")
    print(f"{'='*60}")
    
    model = create_model(
        model_type=model_type,
        n_features=len(feature_cols),
        window_size=config.WINDOW_SIZE,
        dropout=0.3
    )
    
    pos_weight = train_dataset.get_positive_weight()
    
    print(f"\nModel architecture: {model_type}")
    print(f"Positive weight: {pos_weight:.4f}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.MAX_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        pos_weight=pos_weight,
        patience=config.EARLY_STOPPING_PATIENCE,
        save_path=model_path
    )
    
    # Save training history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # ========================================================================
    # EVALUATE MODEL
    # ========================================================================
    
    print(f"\n{'='*60}")
    print(f"Evaluating {model_type.upper()}")
    print(f"{'='*60}")
    
    # Load best model
    checkpoint = torch.load(model_path)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Overall evaluation
    test_metrics, test_predictions, test_targets = evaluate_model(
        model=model,
        test_loader=test_loader,
        flight_ids=test_flight_ids,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Save overall results
    # Convert numpy types to Python native types for JSON serialization
    test_metrics_serializable = {}
    for key, value in test_metrics.items():
        if hasattr(value, 'item'):  # numpy scalar
            test_metrics_serializable[key] = value.item()
        elif isinstance(value, (np.integer, np.floating)):
            test_metrics_serializable[key] = float(value)
        else:
            test_metrics_serializable[key] = value
    
    with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics_serializable, f, indent=2)
    
    np.savez(
        os.path.join(output_dir, 'test_predictions.npz'),
        predictions=test_predictions,
        labels=test_targets,
        probabilities=test_predictions
    )
    
    # ========================================================================
    # PER-ATTACK EVALUATION
    # ========================================================================
    
    print(f"\n{'='*60}")
    print(f"Per-Attack Type Evaluation")
    print(f"{'='*60}")
    
    per_attack_metrics = evaluate_per_attack_type(
        model=model,
        test_windows=test_windows,
        test_labels=test_labels,
        test_flight_ids=test_flight_ids,
        test_attack_info=test_attack_info,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        feature_cols=feature_cols
    )
    
    with open(os.path.join(output_dir, 'per_attack_metrics.json'), 'w') as f:
        json.dump(per_attack_metrics, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Model {model_type.upper()} - Summary")
    print(f"{'='*60}")
    print(f"Overall ROC-AUC: {test_metrics_serializable['roc_auc']:.4f}")
    print(f"Overall F1 Score: {test_metrics_serializable['f1']:.4f}")
    print(f"\nPer-Attack Performance:")
    for attack_type in config.ATTACK_TYPES:
        if attack_type in per_attack_metrics:
            auc = per_attack_metrics[attack_type]['auc_roc']
            f1 = per_attack_metrics[attack_type]['f1_score']
            print(f"  {attack_type:20s}: AUC={auc:.4f}, F1={f1:.4f}")
    
    return {
        'model': model_type,
        'metrics': test_metrics_serializable,  # Use serializable version
        'per_attack_metrics': per_attack_metrics,
        'skipped': False
    }


def evaluate_per_attack_type(model, test_windows, test_labels, test_flight_ids,
                             test_attack_info, device, feature_cols):
    """Evaluate model performance on each attack type separately."""
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    
    model.eval()
    per_attack_metrics = {}
    
    # Create mapping from flight_id to attack_type
    flight_to_attack = dict(zip(test_attack_info['flight'], test_attack_info['attack_type']))
    
    for attack_type in config.ATTACK_TYPES:
        # Find windows belonging to this attack type
        attack_mask = np.array([flight_to_attack.get(fid, 'none') == attack_type 
                               for fid in test_flight_ids])
        
        if attack_mask.sum() == 0:
            print(f"  {attack_type:20s}: No samples found")
            continue
        
        # Get windows and labels for this attack
        attack_windows = test_windows[attack_mask]
        attack_labels = test_labels[attack_mask]
        
        # Create dataset and loader
        attack_dataset = WindowDataset(attack_windows, attack_labels)
        attack_loader = DataLoader(attack_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        # Predict
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch_windows, batch_labels in attack_loader:
                batch_windows = batch_windows.to(device)
                outputs = model(batch_windows)
                probs = outputs.cpu().numpy().flatten()
                all_probs.extend(probs)
                all_labels.extend(batch_labels.numpy())
        
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # Compute metrics
        if len(np.unique(all_labels)) < 2:
            print(f"  {attack_type:20s}: Only one class present, skipping")
            continue
        
        auc = roc_auc_score(all_labels, all_probs)
        preds = (all_probs >= 0.5).astype(int)
        f1 = f1_score(all_labels, preds)
        precision = precision_score(all_labels, preds, zero_division=0)
        recall = recall_score(all_labels, preds, zero_division=0)
        
        per_attack_metrics[attack_type] = {
            'auc_roc': float(auc),
            'f1_score': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'n_samples': int(attack_mask.sum()),
            'n_positive': int(all_labels.sum())
        }
        
        print(f"  {attack_type:20s}: AUC={auc:.4f}, F1={f1:.4f}, N={attack_mask.sum()}")
    
    return per_attack_metrics


# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("="*80)
    print("MULTI-MODEL TRAINING - STRATEGY C (Conservative Hybrid)")
    print("="*80)
    print(f"Models to train: {len(config.MODEL_TYPES)}")
    print(f"Attack types: {len(config.ATTACK_TYPES)}")
    print(f"Attack types: {', '.join(config.ATTACK_TYPES)}")
    print(f"Strategy: Flight Reuse for Train/Val, Stratified Test")
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # ========================================================================
    # LOAD AND SPLIT DATA
    # ========================================================================
    
    base_df = load_flights_data(config.DATA_PATH)
    print(f"\nLoaded {len(base_df)} samples from {len(base_df['flight'].unique())} flights")
    
    # Use stratified split with reuse support
    print(f"\n{'='*80}")
    print("SPLITTING DATA WITH STRATIFIED SAMPLING")
    print(f"{'='*80}")
    
    train_flights, val_flights, test_flights, test_allocation = stratified_split_with_reuse(
        df=base_df,
        attack_types=config.ATTACK_TYPES,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        min_test_per_attack=config.MIN_TEST_FLIGHTS_PER_ATTACK,
        random_seed=config.RANDOM_SEED
    )
    
    # Expand train and val with all attacks (flight reuse)
    print(f"\n{'='*80}")
    print("EXPANDING TRAIN/VAL WITH FLIGHT REUSE")
    print(f"{'='*80}")
    
    train_pairs = expand_flights_with_all_attacks(train_flights, config.ATTACK_TYPES)
    val_pairs = expand_flights_with_all_attacks(val_flights, config.ATTACK_TYPES)
    
    print(f"Train: {len(train_flights)} flights → {len(train_pairs)} samples")
    print(f"Val: {len(val_flights)} flights → {len(val_pairs)} samples")
    print(f"Test: {len(test_flights)} flights (no reuse)")
    
    # Save flight splits (convert numpy types to Python native types for JSON serialization)
    flight_splits = {
        'train': [int(f) for f in train_flights],
        'val': [int(f) for f in val_flights],
        'test': [int(f) for f in test_flights],
        'test_allocation': {k: [int(f) for f in v] for k, v in test_allocation.items()},
        'train_samples': int(len(train_pairs)),
        'val_samples': int(len(val_pairs)),
        'strategy': 'Conservative Hybrid (C)',
        'random_seed': int(config.RANDOM_SEED)
    }
    
    with open(os.path.join(config.OUTPUT_DIR, 'flight_splits.json'), 'w') as f:
        json.dump(flight_splits, f, indent=2)
    
    # ========================================================================
    # TRAIN ALL MODELS
    # ========================================================================
    
    all_results = []
    
    for i, model_type in enumerate(config.MODEL_TYPES, 1):
        print(f"\n\n")
        print("="*80)
        print(f"PROGRESS: Model {i}/{len(config.MODEL_TYPES)}")
        print("="*80)
        
        output_dir = os.path.join(config.OUTPUT_DIR, model_type)
        
        result = train_single_model(
            model_type=model_type,
            base_df=base_df,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            test_flights=test_flights,
            test_allocation=test_allocation,
            output_dir=output_dir
        )
        
        all_results.append(result)
    
    # Save summary
    print(f"\n\n")
    print("="*80)
    print("ALL MODELS TRAINED")
    print("="*80)
    
    summary = []
    for result in all_results:
        summary.append({
            'model': result['model'],
            'roc_auc': result['metrics']['roc_auc'],
            'f1': result['metrics']['f1'],
            'precision': result['metrics']['precision'],
            'recall': result['metrics']['recall'],
            'skipped': result.get('skipped', False)
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values('roc_auc', ascending=False)
    summary_df.to_csv(os.path.join(config.OUTPUT_DIR, 'overall_summary.csv'), index=False)
    
    print("\nOverall Summary:")
    print(summary_df.to_string(index=False))
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {config.OUTPUT_DIR}")
    print(f"{'='*80}")
    
    print("\nNext steps:")
    print("1. Run: python compare_models.py")
    print("2. Analyze per-attack performance")
    print("3. Compare with original results (without flight reuse)")


if __name__ == '__main__':
    main()
