"""
Main pipeline: load MATLAB data → train 7 models × 5 seeds → evaluate → save.

Usage:
    cd ~/projects/UAV/matlab_pipeline
    python main.py
"""
import os, json, pickle
import numpy as np
import torch

import config
import data_loader
from models import MODEL_REGISTRY
from training import train_multi_seed
from evaluation import evaluate_all_models
from statistical_tests import aggregate_metrics, pairwise_comparisons, print_summary_table

os.makedirs(config.OUTPUT_DIR, exist_ok=True)


def main():
    print('\n=== UAV GPS Spoofing — MATLAB Pipeline ===')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print('\n[1/4] Loading MATLAB simulation data...')
    train_df, val_df, test_df = data_loader.load_all_splits()

    print('\n[1/4] Creating sliding windows...')
    X_tr, y_tr, m_tr = data_loader.make_windows(train_df)
    X_va, y_va, m_va = data_loader.make_windows(val_df)
    X_te, y_te, m_te = data_loader.make_windows(test_df)
    print(f'  Windows — train:{X_tr.shape}  val:{X_va.shape}  test:{X_te.shape}')
    print(f'  Label balance — train:{y_tr.mean():.1%}  val:{y_va.mean():.1%}  test:{y_te.mean():.1%}')

    # ── 2. Train all models ───────────────────────────────────────────────────
    print('\n[2/4] Training 7 models × 5 seeds...')
    model_results = {}
    for model_name, model_class in MODEL_REGISTRY.items():
        results = train_multi_seed(
            model_class, model_name,
            X_tr, y_tr, X_va, y_va,
            seeds=config.SEEDS,
            n_features=config.N_FEATURES,
            window_size=config.WINDOW_SIZE,
        )
        model_results[model_name] = results

    # Save checkpoints are already written by training.py
    # Save metadata (excluding model objects) for quick reload
    meta_path = os.path.join(config.OUTPUT_DIR, 'training_metadata.json')
    meta = {name: [{'seed': r['seed'], 'best_epoch': r['best_epoch'],
                    'final_val_loss': min(r['val_loss_history'])}
                   for r in results]
            for name, results in model_results.items()}
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'  Training metadata saved to {meta_path}')

    # ── 3. Evaluate ───────────────────────────────────────────────────────────
    print('\n[3/4] Evaluating on test set...')
    all_metrics = evaluate_all_models(model_results, X_te, m_te, device=device)

    print_summary_table(all_metrics)

    # Statistical tests: pairwise DR@5s comparisons
    print('\n--- Pairwise Bonferroni t-tests (DR@5s) ---')
    comparisons = pairwise_comparisons(all_metrics, metric_key='dr_at_5s')
    for (ma, mb), res in comparisons.items():
        sig = '✓' if res['significant'] else '✗'
        print(f"  {ma} vs {mb}: p={res['p_value']:.4f} {sig}")

    # ── 4. Save results ───────────────────────────────────────────────────────
    print('\n[4/4] Saving results...')
    results_path = os.path.join(config.OUTPUT_DIR, 'all_metrics.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(all_metrics, f)

    # Also save aggregated (CI) results as JSON for easy inspection
    agg_path = os.path.join(config.OUTPUT_DIR, 'aggregated_metrics.json')
    agg_all = {name: aggregate_metrics(ms) for name, ms in all_metrics.items()}
    # JSON-serialise (replace inf/nan with strings)
    def _clean(obj):
        if isinstance(obj, float):
            if obj != obj: return 'nan'
            if obj == float('inf'): return 'inf'
            if obj == float('-inf'): return '-inf'
        if isinstance(obj, dict): return {k: _clean(v) for k, v in obj.items()}
        return obj
    with open(agg_path, 'w') as f:
        json.dump(_clean(agg_all), f, indent=2)

    # Save test metadata for cross-domain eval later
    meta_te_path = os.path.join(config.OUTPUT_DIR, 'test_meta.pkl')
    with open(meta_te_path, 'wb') as f:
        pickle.dump({'X_te': X_te, 'y_te': y_te, 'm_te': m_te}, f)

    print(f'\nAll results saved to {config.OUTPUT_DIR}/')
    print('\n=== Pipeline complete ===\n')


if __name__ == '__main__':
    main()
