"""evaluation.py

High-level evaluation functions to compute time-aware metrics for all models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import os

import config
import data_loader_simple as data_loader
import time_aware_metrics as tam


def evaluate_single_model(model_name: str, data: Dict[str, np.ndarray]) -> Dict:
    """
    Evaluate a single model using time-aware metrics.
    
    Args:
        model_name: Name of the model
        data: Dictionary with y_true, y_pred, timestamps, flight_ids, attack_types
    
    Returns:
        Dictionary with all evaluation results
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name.upper()}")
    print(f"{'='*60}")
    
    # Extract data
    y_true = data['y_true']
    y_pred = data['y_pred']
    timestamps = data['timestamps']
    flight_ids = data['flight_ids']
    attack_types = data['attack_types']
    
    # Compute all metrics
    metrics = tam.compute_all_metrics(
        y_true, y_pred, timestamps, flight_ids, attack_types
    )
    
    # Print summary
    print(f"\nOverall Metrics:")
    print(f"  Total attacks: {metrics['n_attacks']}")
    print(f"  Detected: {metrics['n_detected']} ({metrics['n_detected']/max(metrics['n_attacks'],1)*100:.1f}%)")
    add_str = f"{metrics['ADD']:.3f}s" if metrics['ADD'] is not None else "N/A"
    print(f"  ADD: {add_str}")
    mtbfa_str = f"{metrics['MTBFA']:.1f}h" if not np.isinf(metrics['MTBFA']) else "inf"
    print(f"  MTBFA: {mtbfa_str}")
    print(f"  False alarms: {metrics['n_false_alarms']}")
    
    print(f"\nDetection Rate at different time thresholds:")
    for dt in config.DELTA_T_VALUES:
        dr = metrics['DR'][dt]
        print(f"  DR@{dt}s: {dr:.3f}")
    
    print(f"\nPer-Attack Performance:")
    for attack_type, attack_metrics in metrics['per_attack'].items():
        print(f"  {attack_type}:")
        print(f"    Instances: {attack_metrics['n_instances']}")
        add_str = f"{attack_metrics['ADD']:.3f}s" if attack_metrics['ADD'] is not None else "N/A"
        print(f"    ADD: {add_str}")
        print(f"    DR@5s: {attack_metrics['DR@5s']:.3f}")
        print(f"    DR@10s: {attack_metrics['DR@10s']:.3f}")
    
    return metrics


def evaluate_all_models(save_results: bool = True) -> Dict[str, Dict]:
    """
    Evaluate all models and optionally save results.
    
    Args:
        save_results: Whether to save results to CSV files
    
    Returns:
        Dictionary mapping model_name -> evaluation_metrics
    """
    print("\n" + "="*70)
    print("TIME-AWARE EVALUATION - ALL MODELS")
    print("="*70)
    
    # Load all model data
    print("\n[1/3] Loading data...")
    all_data = data_loader.load_all_models_simple()
    
    # Evaluate each model
    print("\n[2/3] Computing time-aware metrics...")
    all_metrics = {}
    
    for model_name, data in all_data.items():
        try:
            metrics = evaluate_single_model(model_name, data)
            all_metrics[model_name] = metrics
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            continue
    
    # Save results
    if save_results:
        print("\n[3/3] Saving results...")
        save_evaluation_results(all_metrics)
    
    print("\n" + "="*70)
    print(f"Evaluation complete! Evaluated {len(all_metrics)} models.")
    print("="*70)
    
    return all_metrics


def save_evaluation_results(all_metrics: Dict[str, Dict]):
    """
    Save evaluation results to CSV files.
    
    Args:
        all_metrics: Dictionary mapping model_name -> evaluation_metrics
    """
    config.ensure_output_dir()
    
    # 1. Overall metrics CSV
    overall_rows = []
    for model_name, metrics in all_metrics.items():
        row = {
            'model': model_name,
            'n_attacks': metrics['n_attacks'],
            'n_detected': metrics['n_detected'],
            'detection_rate': metrics['n_detected'] / max(metrics['n_attacks'], 1),
            'ADD': metrics['ADD'],
            'MTBFA': metrics['MTBFA'],
            'n_false_alarms': metrics['n_false_alarms']
        }
        
        # Add DR@Δt columns
        for dt in config.DELTA_T_VALUES:
            row[f'DR@{dt}s'] = metrics['DR'][dt]
        
        overall_rows.append(row)
    
    overall_df = pd.DataFrame(overall_rows)
    overall_csv = os.path.join(config.OUTPUT_DIR, config.OVERALL_METRICS_CSV)
    overall_df.to_csv(overall_csv, index=False)
    print(f"  Saved overall metrics: {overall_csv}")
    
    # 2. Per-attack metrics for each model
    for model_name, metrics in all_metrics.items():
        per_attack_rows = []
        
        for attack_type, attack_metrics in metrics['per_attack'].items():
            row = {
                'model': model_name,
                'attack_type': attack_type,
                'n_instances': attack_metrics['n_instances'],
                'ADD': attack_metrics['ADD'],
                'DR@5s': attack_metrics['DR@5s'],
                'DR@10s': attack_metrics['DR@10s']
            }
            per_attack_rows.append(row)
        
        if per_attack_rows:
            per_attack_df = pd.DataFrame(per_attack_rows)
            per_attack_csv = os.path.join(
                config.OUTPUT_DIR, 
                config.PER_ATTACK_METRICS_CSV_TEMPLATE.format(model=model_name)
            )
            per_attack_df.to_csv(per_attack_csv, index=False)
            print(f"  Saved per-attack metrics for {model_name}: {per_attack_csv}")
    
    # 3. Detailed delays CSV
    detailed_rows = []
    for model_name, metrics in all_metrics.items():
        for seg, delay in zip(metrics['segments'], metrics['delays']):
            row = {
                'model': model_name,
                'flight_id': seg['flight_id'],
                'attack_type': seg['attack_type'],
                'start_time': seg['t_start'],
                'duration': seg['t_end'] - seg['t_start'],
                'delay': delay if delay is not None else np.nan,
                'detected': delay is not None
            }
            detailed_rows.append(row)
    
    if detailed_rows:
        detailed_df = pd.DataFrame(detailed_rows)
        detailed_csv = os.path.join(config.OUTPUT_DIR, config.DETAILED_DELAYS_CSV)
        detailed_df.to_csv(detailed_csv, index=False)
        print(f"  Saved detailed delays: {detailed_csv}")


def compare_models(all_metrics: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create a comparison table of all models.
    
    Args:
        all_metrics: Dictionary mapping model_name -> evaluation_metrics
    
    Returns:
        DataFrame with model comparison
    """
    rows = []
    
    for model_name, metrics in all_metrics.items():
        row = {
            'Model': model_name.upper(),
            'ADD (s)': f"{metrics['ADD']:.2f}",
            'MTBFA (h)': f"{metrics['MTBFA']:.1f}",
            'DR@1s': f"{metrics['DR'][1]:.3f}",
            'DR@5s': f"{metrics['DR'][5]:.3f}",
            'DR@10s': f"{metrics['DR'][10]:.3f}",
            'False Alarms': metrics['n_false_alarms']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    # Run full evaluation
    all_metrics = evaluate_all_models(save_results=True)
    
    # Print comparison table
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    comparison = compare_models(all_metrics)
    print(comparison.to_string(index=False))
