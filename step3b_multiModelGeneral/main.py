"""
Multi-Model Training with Mixed Attack Types

Trains 7 different model architectures on a dataset with mixed attack types.
Each model sees all 6 attack types during training for better generalization.

Experiment: 7 models × 1 mixed dataset = 7 experiments
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
from mixed_attack_injector import MixedAttackInjector
from model import create_model
from data_loader import load_flights_data, split_flights, get_flight_subset, compute_delta_t, convert_to_local_coordinates
from labeling import generate_point_labels
from feature_engineering import compute_all_features, get_feature_columns, normalize_features
from window_creation import create_windows_from_dataset, balance_windows, WindowDataset
from training import train_model
from evaluation import evaluate_model


# ============================================================================
# Single Model Training
# ============================================================================

def train_single_model(model_type, base_df, train_flights, val_flights, test_flights, 
                      flight_splits_dict, output_dir):
    """
    Train a single model on mixed attack dataset.
    
    Args:
        model_type: Model architecture name
        base_df: Base DataFrame with all flights
        train_flights: List of training flight IDs
        val_flights: List of validation flight IDs
        test_flights: List of test flight IDs
        flight_splits_dict: Dictionary with flight split info (to save)
        output_dir: Output directory for this model
    
    Returns:
        Dictionary with test metrics
    """
    
    print(f"\n{'='*80}")
    print(f"TRAINING MODEL: {model_type.upper()}")
    print(f"{'='*80}")
    
    # Create output directory
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
        else:
            print(f"Warning: Model exists but metrics not found, re-training...")
    
    # Initialize injector
    injector = MixedAttackInjector(random_seed=config.RANDOM_SEED)
    
    # ========================================================================
    # INJECT MIXED ATTACKS
    # ========================================================================
    
    print(f"\n{'='*60}")
    print(f"TRAINING SET - Injecting Mixed Attacks")
    print(f"{'='*60}")
    train_df = get_flight_subset(base_df, train_flights)
    train_df, train_attack_info = injector.inject_mixed_attacks_to_dataset(
        train_df,
        attack_ratio=config.TRAIN_ATTACK_RATIO,
        attack_types=config.ATTACK_TYPES
    )
    train_attack_info.to_csv(os.path.join(output_dir, 'train_attack_info.csv'), index=False)
    
    print(f"\n{'='*60}")
    print(f"VALIDATION SET - Injecting Mixed Attacks")
    print(f"{'='*60}")
    val_df = get_flight_subset(base_df, val_flights)
    val_df, val_attack_info = injector.inject_mixed_attacks_to_dataset(
        val_df,
        attack_ratio=config.VAL_ATTACK_RATIO,
        attack_types=config.ATTACK_TYPES
    )
    val_attack_info.to_csv(os.path.join(output_dir, 'val_attack_info.csv'), index=False)
    
    print(f"\n{'='*60}")
    print(f"TEST SET - Injecting Mixed Attacks (with coverage guarantee)")
    print(f"{'='*60}")
    test_df = get_flight_subset(base_df, test_flights)
    # Use guaranteed coverage for test set to ensure all attack types are tested
    test_df, test_attack_info = injector.inject_mixed_attacks_with_guarantee(
        test_df,
        attack_ratio=config.TEST_ATTACK_RATIO,
        attack_types=config.ATTACK_TYPES,
        min_per_attack=config.MIN_TEST_FLIGHTS_PER_ATTACK if hasattr(config, 'MIN_TEST_FLIGHTS_PER_ATTACK') else 2
    )
    test_attack_info.to_csv(os.path.join(output_dir, 'test_attack_info.csv'), index=False)
    
    # ========================================================================
    # GENERATE LABELS
    # ========================================================================
    
    print(f"\nGenerating labels...")
    train_df = generate_point_labels(train_df, train_attack_info)
    val_df = generate_point_labels(val_df, val_attack_info)
    test_df = generate_point_labels(test_df, test_attack_info)
    
    # ========================================================================
    # FEATURE ENGINEERING
    # ========================================================================
    
    print(f"\nComputing features...")
    train_df = compute_all_features(train_df)
    val_df = compute_all_features(val_df)
    test_df = compute_all_features(test_df)
    
    feature_cols = get_feature_columns()
    train_df, val_df, test_df, normalization_stats = normalize_features(
        train_df, val_df, test_df, feature_cols
    )
    
    with open(os.path.join(output_dir, 'normalization_stats.json'), 'w') as f:
        json.dump(normalization_stats, f, indent=2)
    
    # ========================================================================
    # CREATE WINDOWS
    # ========================================================================
    
    print(f"\nCreating windows...")
    train_windows, train_labels, train_flight_ids = create_windows_from_dataset(train_df, feature_cols)
    val_windows, val_labels, val_flight_ids = create_windows_from_dataset(val_df, feature_cols)
    test_windows, test_labels, test_flight_ids = create_windows_from_dataset(test_df, feature_cols)
    
    print(f"Balancing windows...")
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
    
    # Create dataloaders
    train_dataset = WindowDataset(train_windows, train_labels)
    val_dataset = WindowDataset(val_windows, val_labels)
    test_dataset = WindowDataset(test_windows, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # ========================================================================
    # TRAIN MODEL
    # ========================================================================
    
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} Model")
    print(f"{'='*60}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = create_model(
        model_type=model_type,
        n_features=len(feature_cols),
        window_size=config.WINDOW_SIZE,
        dropout=0.3
    )
    
    pos_weight = train_dataset.get_positive_weight()
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=config.MAX_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        pos_weight=pos_weight,
        patience=config.EARLY_STOPPING_PATIENCE,
        save_path=model_path
    )
    
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # ========================================================================
    # EVALUATE MODEL
    # ========================================================================
    
    print(f"\n{'='*60}")
    print(f"Evaluating {model_type.upper()} Model")
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
        device=device
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
    
    # 保存完整的窗口元数据用于时间感知评估
    np.savez(
        os.path.join(output_dir, 'test_predictions.npz'),
        predictions=test_predictions,
        labels=test_targets,
        probabilities=test_predictions,
        # 窗口元数据（新增）
        window_indices=np.arange(len(test_targets)),
        flight_ids=test_flight_ids,
        # 配置参数（新增）
        window_size=config.WINDOW_SIZE,
        step_size=config.STEP_SIZE,
        sampling_rate=config.SAMPLING_RATE
    )
    
    # 生成详细的窗口元数据文件
    print(f"\n  Generating window metadata file...")
    window_metadata_list = []
    for idx in range(len(test_targets)):
        flight_id = test_flight_ids[idx]
        flight_info = test_attack_info[test_attack_info['flight'] == flight_id]
        
        if len(flight_info) > 0:
            flight_info = flight_info.iloc[0]
            attacked = flight_info['attacked']
            attack_type = flight_info['attack_type'] if attacked else 'normal'
            attack_start_time = flight_info['attack_start_time'] if attacked else np.nan
        else:
            attacked = False
            attack_type = 'normal'
            attack_start_time = np.nan
        
        window_metadata_list.append({
            'window_index': idx,
            'flight_id': flight_id,
            'label': int(test_targets[idx]),
            'attacked': attacked,
            'attack_type': attack_type,
            'attack_start_time': attack_start_time
        })
    
    window_metadata_df = pd.DataFrame(window_metadata_list)
    window_metadata_df.to_csv(os.path.join(output_dir, 'window_metadata.csv'), index=False)
    print(f"    ✓ Saved window metadata: {len(window_metadata_df)} windows with attack timing info")
    
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
        device=device,
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
    """
    Evaluate model performance on each attack type separately.
    
    Returns:
        Dictionary with metrics for each attack type
    """
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
    """Main training loop for all models."""
    
    print("="*80)
    print("Multi-Model Training with Mixed Attack Types")
    print("="*80)
    print(f"Models to train: {len(config.MODEL_TYPES)}")
    print(f"Attack types: {len(config.ATTACK_TYPES)}")
    print(f"Attack types: {', '.join(config.ATTACK_TYPES)}")
    print("="*80)
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Load base data
    print("\nLoading flight data...")
    base_df = load_flights_data(config.DATA_PATH)
    base_df = compute_delta_t(base_df)
    base_df = convert_to_local_coordinates(base_df)
    
    print(f"Loaded {len(base_df)} data points from {base_df['flight'].nunique()} flights")
    
    # Split flights
    print("\nSplitting flights...")
    train_flights, val_flights, test_flights = split_flights(
        base_df,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        random_seed=config.RANDOM_SEED
    )
    
    print(f"Train: {len(train_flights)} flights")
    print(f"Val:   {len(val_flights)} flights")
    print(f"Test:  {len(test_flights)} flights")
    
    # Save flight splits
    flight_splits = {
        'train': train_flights,
        'val': val_flights,
        'test': test_flights,
        'random_seed': config.RANDOM_SEED
    }
    
    with open(os.path.join(config.OUTPUT_DIR, 'flight_splits.json'), 'w') as f:
        json.dump(flight_splits, f, indent=2)
    
    # Train each model
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
            train_flights=train_flights,
            val_flights=val_flights,
            test_flights=test_flights,
            flight_splits_dict=flight_splits,
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
    print("2. Check visualizations in output/visualizations/")


if __name__ == '__main__':
    main()
