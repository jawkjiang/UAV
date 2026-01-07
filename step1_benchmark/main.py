"""
Main Execution Script

Complete pipeline for GPS spoofing detection:
1. Load and split data
2. Inject attacks
3. Create windows
4. Train model
5. Evaluate on test set
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import json
import config
from data_loader import (
    load_flights_data, split_flights, get_flight_subset,
    compute_delta_t, convert_to_local_coordinates
)
from attack_injector import GPSSpoofingInjector
from labeling import generate_point_labels
from feature_engineering import (
    compute_all_features, get_feature_columns, normalize_features
)
from window_creation import (
    create_windows_from_dataset, balance_windows, WindowDataset
)
from model import create_model
from training import train_model
from evaluation import evaluate_model, print_evaluation_report, plot_evaluation_curves


def set_random_seeds(seed=config.RANDOM_SEED):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main execution pipeline."""
    
    print("="*80)
    print("GPS Spoofing Detection - Complete Pipeline")
    print("="*80)
    
    # Set random seeds
    set_random_seeds(config.RANDOM_SEED)
    
    # Create output directories
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # ========================================================================
    # Step 1: Load and split data
    # ========================================================================
    print("\n[Step 1] Loading and splitting data...")
    
    df = load_flights_data()
    train_flights, val_flights, test_flights = split_flights(df)
    
    # Save flight splits for reproducibility
    flight_splits = {
        'train': train_flights,
        'val': val_flights,
        'test': test_flights,
        'random_seed': config.RANDOM_SEED
    }
    
    with open(os.path.join(config.OUTPUT_DIR, 'flight_splits.json'), 'w') as f:
        json.dump(flight_splits, f, indent=2)
    
    print(f"Saved flight splits to {config.OUTPUT_DIR}/flight_splits.json")
    
    # ========================================================================
    # Step 2: Preprocess data
    # ========================================================================
    print("\n[Step 2] Preprocessing data...")
    
    df = compute_delta_t(df)
    df = convert_to_local_coordinates(df)
    
    # ========================================================================
    # Step 3: Inject attacks
    # ========================================================================
    print("\n[Step 3] Injecting GPS spoofing attacks...")
    
    injector = GPSSpoofingInjector(random_seed=config.RANDOM_SEED)
    
    # Training set: inject to achieve 30-40% positive windows
    # Start with ~50% flights attacked, then adjust windows
    print("\nProcessing training set...")
    train_df = get_flight_subset(df, train_flights)
    train_df, train_attack_info = injector.inject_attacks_to_dataset(
        train_df, attack_ratio=0.5
    )
    train_df = generate_point_labels(train_df, train_attack_info)
    
    # Validation set: inject to achieve 1-5% positive windows
    print("\nProcessing validation set...")
    val_df = get_flight_subset(df, val_flights)
    val_df, val_attack_info = injector.inject_attacks_to_dataset(
        val_df, attack_ratio=0.1
    )
    val_df = generate_point_labels(val_df, val_attack_info)
    
    # Test set: inject to achieve 1-5% positive windows
    print("\nProcessing test set...")
    test_df = get_flight_subset(df, test_flights)
    test_df, test_attack_info = injector.inject_attacks_to_dataset(
        test_df, attack_ratio=0.1
    )
    test_df = generate_point_labels(test_df, test_attack_info)
    
    # Save attack info
    train_attack_info.to_csv(os.path.join(config.OUTPUT_DIR, 'train_attack_info.csv'), index=False)
    val_attack_info.to_csv(os.path.join(config.OUTPUT_DIR, 'val_attack_info.csv'), index=False)
    test_attack_info.to_csv(os.path.join(config.OUTPUT_DIR, 'test_attack_info.csv'), index=False)
    
    # ========================================================================
    # Step 4: Feature engineering
    # ========================================================================
    print("\n[Step 4] Computing features...")
    
    train_df = compute_all_features(train_df)
    val_df = compute_all_features(val_df)
    test_df = compute_all_features(test_df)
    
    feature_columns = get_feature_columns()
    print(f"Using {len(feature_columns)} features: {feature_columns}")
    
    # Normalize features
    train_df, val_df, test_df, norm_stats = normalize_features(
        train_df, val_df, test_df, feature_columns
    )
    
    with open(os.path.join(config.OUTPUT_DIR, 'normalization_stats.json'), 'w') as f:
        json.dump(norm_stats, f, indent=2)
    
    # ========================================================================
    # Step 5: Create windows
    # ========================================================================
    print("\n[Step 5] Creating windows...")
    
    print("\nCreating training windows...")
    train_windows, train_labels, train_flight_ids = create_windows_from_dataset(
        train_df, feature_columns
    )
    
    print("\nCreating validation windows...")
    val_windows, val_labels, val_flight_ids = create_windows_from_dataset(
        val_df, feature_columns
    )
    
    print("\nCreating test windows...")
    test_windows, test_labels, test_flight_ids = create_windows_from_dataset(
        test_df, feature_columns
    )
    
    # Balance windows to achieve target positive ratios
    print("\nBalancing training windows (target: 30-40% positive)...")
    train_windows, train_labels, train_flight_ids = balance_windows(
        train_windows, train_labels, train_flight_ids,
        target_pos_ratio=0.35, random_seed=config.RANDOM_SEED
    )
    
    print("\nBalancing validation windows (target: 1-5% positive)...")
    val_windows, val_labels, val_flight_ids = balance_windows(
        val_windows, val_labels, val_flight_ids,
        target_pos_ratio=0.03, random_seed=config.RANDOM_SEED
    )
    
    print("\nBalancing test windows (target: 1-5% positive)...")
    test_windows, test_labels, test_flight_ids = balance_windows(
        test_windows, test_labels, test_flight_ids,
        target_pos_ratio=0.03, random_seed=config.RANDOM_SEED
    )
    
    # ========================================================================
    # Step 6: Create PyTorch datasets
    # ========================================================================
    print("\n[Step 6] Creating PyTorch datasets...")
    
    train_dataset = WindowDataset(train_windows, train_labels)
    val_dataset = WindowDataset(val_windows, val_labels)
    test_dataset = WindowDataset(test_windows, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    pos_weight = train_dataset.get_positive_weight()
    print(f"Positive sample weight for loss: {pos_weight:.2f}")
    
    # ========================================================================
    # Step 7: Train model
    # ========================================================================
    print("\n[Step 7] Training model...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = create_model(
        model_type='cnn',
        n_features=len(feature_columns),
        window_size=config.WINDOW_SIZE,
        dropout=0.3
    )
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    save_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
    
    history = train_model(
        model, train_loader, val_loader,
        pos_weight=pos_weight,
        num_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        patience=config.EARLY_STOPPING_PATIENCE,
        device=device,
        save_path=save_path
    )
    
    # Save training history
    with open(os.path.join(config.OUTPUT_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # ========================================================================
    # Step 8: Load best model and evaluate on test set
    # ========================================================================
    print("\n[Step 8] Evaluating on test set...")
    
    # Load best model
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    
    # Evaluate
    metrics, predictions, targets = evaluate_model(
        model, test_loader, test_flight_ids, device=device
    )
    
    # Print results
    print_evaluation_report(metrics)
    
    # Save metrics
    with open(os.path.join(config.OUTPUT_DIR, 'test_metrics.json'), 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        metrics_serializable = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                               for k, v in metrics.items()}
        json.dump(metrics_serializable, f, indent=2)
    
    # Plot curves
    plot_evaluation_curves(
        targets, predictions,
        save_path=os.path.join(config.OUTPUT_DIR, 'evaluation_curves.png')
    )
    
    # Save predictions
    np.savez(os.path.join(config.OUTPUT_DIR, 'test_predictions.npz'),
             predictions=predictions,
             targets=targets,
             flight_ids=test_flight_ids)
    
    print("\n" + "="*80)
    print("Pipeline completed successfully!")
    print(f"Results saved to: {config.OUTPUT_DIR}")
    print(f"Model saved to: {save_path}")
    print("="*80)


if __name__ == '__main__':
    main()
