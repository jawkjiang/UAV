"""
Ablation: window size × step size sensitivity sweep.

Trains the best baseline (GRU) across L ∈ {30,50,80} and S ∈ {3,5,10}.
Reports DR@5s / ADD / MTBFA for each (L,S) combination.

Usage:
    cd ~/projects/UAV
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python step10_ablation/window_step_sweep.py
"""
import sys, os, json, math
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'matlab_pipeline'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'step3_multiModel'))

import config as base_config
import data_loader as _dl
from model import GRUDetector
from training import train_one_model
from evaluation import compute_time_aware_metrics

WINDOW_SIZES = [30, 50, 80]
STEP_SIZES   = [3, 5, 10]
SEEDS        = [0, 1, 2]          # 3 seeds for ablation (faster)
OUTPUT_PATH  = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'matlab_pipeline', 'output', 'ablation_window_step.json')


def _make_windows(df, window_size: int, step_size: int):
    """Thin wrapper that overrides window/step without touching config."""
    return _dl.make_windows(df, window_size=window_size, step_size=step_size)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Ablation: window×step sweep  device={device}')

    print('Loading raw splits (no windowing yet)...')
    train_df, val_df, test_df = _dl.load_all_splits()

    results = {}
    for L in WINDOW_SIZES:
        for S in STEP_SIZES:
            key = f'L{L}_S{S}'
            print(f'\n{"="*50}')
            print(f'Window={L}  Step={S}')
            print(f'{"="*50}')

            X_tr, y_tr, _   = _make_windows(train_df, L, S)
            X_va, y_va, _   = _make_windows(val_df,   L, S)
            X_te, y_te, m_te = _make_windows(test_df,  L, S)
            print(f'  Windows: train={X_tr.shape} val={X_va.shape} test={X_te.shape}')

            seed_metrics = []
            for seed in SEEDS:
                result = train_one_model(
                    GRUDetector, f'GRU_L{L}_S{S}',
                    X_tr, y_tr, X_va, y_va,
                    seed=seed,
                    n_features=base_config.N_FEATURES,
                    window_size=L,
                    device=device,
                )
                m = compute_time_aware_metrics(result['model'], X_te, m_te, device)
                m['seed'] = seed
                seed_metrics.append({
                    'dr_at_5s': m['dr_at_5s'],
                    'add_s': m['add_s'],
                    'mtbfa_h': m['mtbfa_h'] if not math.isinf(m['mtbfa_h']) else None,
                    'f1': m['f1'],
                    'seed': seed,
                })
                print(f"  s={seed}  DR@5s={m['dr_at_5s']:.3f}  ADD={m['add_s']:.2f}s  "
                      f"F1={m['f1']:.3f}")
            results[key] = {'window': L, 'step': S, 'seeds': seed_metrics}

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nAblation results saved to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
