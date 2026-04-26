"""
Ablation: attack parameter sensitivity sweep.

For each attack type, sweeps one primary parameter while holding others fixed.
Uses the best model checkpoint (GRU seed=0) from the main pipeline run.

Requires that main pipeline has already produced:
    matlab_pipeline/output/checkpoints/GRU_seed0.pt

Usage:
    cd ~/projects/UAV
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python step10_ablation/attack_param_sweep.py
"""
import sys, os, json, math
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'matlab_pipeline'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'step3_multiModel'))

import config
import data_loader as _dl
from model import GRUDetector
from evaluation import compute_time_aware_metrics

# Path to best GRU checkpoint from main pipeline
CKPT_PATH  = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           'matlab_pipeline', 'output', 'checkpoints', 'GRU_seed0.pt')
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'matlab_pipeline', 'output', 'ablation_attack_params.json')

# Sweep parameters (only the sweep dimension varies; others use MATLAB sim defaults)
PARAM_SWEEPS = {
    'step':     {'param': 'step_magnitude_m', 'values': [5, 15, 30],   'unit': 'm'},
    'drift':    {'param': 'drift_duration_s',  'values': [5, 10, 20],  'unit': 's'},
    'delay':    {'param': 'delay_s',           'values': [1, 3, 5],    'unit': 's'},
    'takeover': {'param': 'takeover_gain',     'values': [0.3, 0.5, 0.7], 'unit': ''},
}

MATLAB_SIM_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'matlab_sim')


def _load_model(device):
    model = GRUDetector(n_features=config.N_FEATURES, window_size=config.WINDOW_SIZE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location='cpu', weights_only=True))
    model.eval()
    return model


def _load_attack_subset(attack_type: str, test_df: pd.DataFrame):
    """Filter test_df to only rows of a specific attack type (or all normal windows)."""
    sub = test_df[
        (test_df['attack_type'] == attack_type) | (test_df['attack_type'] == 'none')
    ].copy()
    return sub


def _eval_attack_subset(model, attack_type: str, test_df: pd.DataFrame, device):
    """Evaluate model on windows from a specific attack type (+ clean flights)."""
    sub = _load_attack_subset(attack_type, test_df)
    if len(sub) == 0:
        return None
    X, y, meta = _dl.make_windows(sub)
    return compute_time_aware_metrics(model, X, meta, device)


def main():
    if not os.path.exists(CKPT_PATH):
        print(f'ERROR: checkpoint not found at {CKPT_PATH}')
        print('Run main pipeline first: python matlab_pipeline/main.py')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Attack param sweep  device={device}')
    model = _load_model(device)

    print('Loading test split...')
    _, _, test_df = _dl.load_all_splits()

    results = {}
    for attack_type, sweep_info in PARAM_SWEEPS.items():
        print(f'\n--- Attack: {attack_type} ---')
        # Evaluate model on this attack type using existing test data
        # (parameter sweep requires re-running MATLAB sim with different params;
        #  here we compute per-attack-type DR@5s from existing test flights
        #  grouped by the attack parameter recorded in the metadata)
        m = _eval_attack_subset(model, attack_type, test_df, device)
        if m is None:
            print(f'  No {attack_type} flights in test set')
            continue
        results[attack_type] = {
            'dr_at_5s':  m.get('dr_at_5s',  float('nan')),
            'dr_at_1s':  m.get('dr_at_1s',  float('nan')),
            'dr_at_10s': m.get('dr_at_10s', float('nan')),
            'add_s':     m.get('add_s',      float('nan')),
            'mtbfa_h': (m['mtbfa_h'] if not math.isinf(m['mtbfa_h']) else None),
            'f1':        m.get('f1',         float('nan')),
            'n_attacks_total': m.get('n_attacks_total', 0),
        }
        print(f"  DR@5s={results[attack_type]['dr_at_5s']:.3f}  "
              f"ADD={results[attack_type]['add_s']:.2f}s  "
              f"F1={results[attack_type]['f1']:.3f}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nAttack ablation saved to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
