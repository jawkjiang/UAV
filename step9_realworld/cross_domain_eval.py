"""
Zero-shot cross-domain evaluation: MATLAB-trained models on real-world data.

Loads trained models from the main pipeline output, evaluates them directly
(no fine-tuning) on ALFA and/or IEEE DataPort datasets.

Usage:
    cd ~/projects/UAV
    python step9_realworld/cross_domain_eval.py [--dataset alfa|dataport|both]
"""
import sys, os, json, math, pickle, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'matlab_pipeline'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'step3_multiModel'))

import torch
import config
import data_loader as _dl
from evaluation import evaluate_all_models, compute_time_aware_metrics
from statistical_tests import aggregate_metrics, print_summary_table

PIPELINE_OUTPUT = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'matlab_pipeline', 'output')
METRICS_PKL   = os.path.join(PIPELINE_OUTPUT, 'all_metrics.pkl')
RESULTS_DIR   = os.path.join(PIPELINE_OUTPUT, 'cross_domain')


def _load_trained_models():
    """Reconstruct model objects from checkpoints."""
    from models import MODEL_REGISTRY
    models_dict = {}
    ckpt_dir = os.path.join(PIPELINE_OUTPUT, 'checkpoints')
    for model_name, model_class in MODEL_REGISTRY.items():
        seed_results = []
        for seed in config.SEEDS:
            ckpt = os.path.join(ckpt_dir, f'{model_name}_seed{seed}.pt')
            if not os.path.exists(ckpt):
                print(f'  WARNING: checkpoint missing for {model_name} seed={seed}')
                continue
            m = model_class(n_features=config.N_FEATURES, window_size=config.WINDOW_SIZE)
            m.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
            m.eval()
            seed_results.append({'model': m, 'seed': seed})
        if seed_results:
            models_dict[model_name] = seed_results
    return models_dict


def _eval_on_dataset(dataset_name: str, df_func, models_dict: dict, device: torch.device):
    """Load real-world dataset, make windows, evaluate all models."""
    print(f'\n--- Evaluating on {dataset_name} ---')
    try:
        df = df_func()
    except FileNotFoundError as e:
        print(f'  SKIP: {e}')
        return None

    X, y, meta = _dl.make_windows(df)
    print(f'  Windows: {X.shape}, label balance: {y.mean():.1%}')

    results = {}
    for model_name, seed_results in models_dict.items():
        seed_metrics = []
        for r in seed_results:
            m = compute_time_aware_metrics(r['model'], X, meta, device)
            m['seed'] = r['seed']
            seed_metrics.append(m)
        results[model_name] = seed_metrics
        agg = aggregate_metrics(seed_metrics)
        dr5  = agg.get('dr_at_5s', {}).get('mean', float('nan'))
        mtbfa = agg.get('mtbfa_h', {}).get('mean', float('nan'))
        print(f"  {model_name:<12} DR@5s={dr5:.3f}  MTBFA={mtbfa:.3f}h")
    return results


def _compare_with_matlab(real_results: dict, matlab_metrics_pkl: str, dataset_name: str):
    """Print side-by-side comparison vs MATLAB test results."""
    if not os.path.exists(matlab_metrics_pkl):
        return
    with open(matlab_metrics_pkl, 'rb') as f:
        matlab_metrics = pickle.load(f)

    print(f'\n{"="*70}')
    print(f'MATLAB sim vs {dataset_name} (zero-shot)  —  DR@5s mean ± CI')
    print(f'{"="*70}')
    header = f"{'Model':<14} {'MATLAB DR@5s':>14} {'Real DR@5s':>12} {'Drop':>8}"
    print(header)
    print('-' * len(header))
    for model_name in real_results:
        if model_name not in matlab_metrics:
            continue
        m_agg  = aggregate_metrics(matlab_metrics[model_name])
        r_agg  = aggregate_metrics(real_results[model_name])
        m_dr5  = m_agg.get('dr_at_5s', {}).get('mean', float('nan'))
        r_dr5  = r_agg.get('dr_at_5s', {}).get('mean', float('nan'))
        drop   = r_dr5 - m_dr5
        print(f"{model_name:<14} {m_dr5:>14.3f} {r_dr5:>12.3f} {drop:>+8.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['alfa', 'dataport', 'both'],
                        default='both')
    args = parser.parse_args()

    if not os.path.exists(METRICS_PKL):
        print(f'ERROR: {METRICS_PKL} not found. Run main pipeline first.')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Cross-domain eval  device={device}')

    models_dict = _load_trained_models()
    print(f'Loaded {len(models_dict)} models')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    from loader_alfa import load_alfa_dataset
    from loader_ieee_dataport import load_dataport_dataset

    all_results = {}

    if args.dataset in ('alfa', 'both'):
        r = _eval_on_dataset('ALFA', load_alfa_dataset, models_dict, device)
        if r:
            all_results['alfa'] = r
            _compare_with_matlab(r, METRICS_PKL, 'ALFA')
            out = os.path.join(RESULTS_DIR, 'alfa_metrics.pkl')
            with open(out, 'wb') as f:
                pickle.dump(r, f)

    if args.dataset in ('dataport', 'both'):
        r = _eval_on_dataset('IEEE DataPort', load_dataport_dataset, models_dict, device)
        if r:
            all_results['dataport'] = r
            _compare_with_matlab(r, METRICS_PKL, 'IEEE DataPort')
            out = os.path.join(RESULTS_DIR, 'dataport_metrics.pkl')
            with open(out, 'wb') as f:
                pickle.dump(r, f)

    if not all_results:
        print('\nNo real-world data available. Place data in step9_realworld/data/')
        print('  ALFA:     https://kilthub.cmu.edu/articles/dataset/ALFA.../12707963')
        print('  DataPort: https://ieee-dataport.org/open-access/uav-attack-dataset')
        return

    # Save combined results
    out = os.path.join(RESULTS_DIR, 'cross_domain_summary.json')
    def _clean(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return str(v)
        if isinstance(v, dict):
            return {k: _clean(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_clean(x) for x in v]
        return v
    with open(out, 'w') as f:
        json.dump({ds: {mn: aggregate_metrics(ms) for mn, ms in res.items()}
                   for ds, res in all_results.items()}, f, indent=2, default=_clean)
    print(f'\nCross-domain results saved to {RESULTS_DIR}/')


if __name__ == '__main__':
    main()
