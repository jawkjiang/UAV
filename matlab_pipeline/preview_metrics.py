"""
Preview metrics from already-completed checkpoints, without waiting for full
pipeline. Loads any checkpoints present in output/checkpoints/ and prints a
side-by-side conventional vs time-aware comparison.

Usage on abee:
    cd ~/projects/UAV
    source ~/TSL/venv/bin/activate
    python matlab_pipeline/preview_metrics.py [--device cpu]
"""
import sys, os, glob, argparse, math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import data_loader
from models import MODEL_REGISTRY
from evaluation import compute_time_aware_metrics
from statistical_tests import aggregate_metrics

CHK_DIR = os.path.join(config.OUTPUT_DIR, 'checkpoints')


def _load_checkpoints():
    """Discover checkpoints by name → list of (seed, path)."""
    found = {}
    for path in sorted(glob.glob(os.path.join(CHK_DIR, '*.pt'))):
        fname = os.path.basename(path).replace('.pt', '')
        # filename pattern: <name>_seed<N>.pt
        if '_seed' not in fname:
            continue
        name, seed_part = fname.rsplit('_seed', 1)
        try:
            seed = int(seed_part)
        except ValueError:
            continue
        if name in MODEL_REGISTRY:
            found.setdefault(name, []).append((seed, path))
    for name in found:
        found[name].sort()
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu',
                    help='cpu (default, avoids GPU contention) or cuda')
    args = ap.parse_args()
    device = torch.device(args.device)

    # 1. Discover checkpoints
    found = _load_checkpoints()
    if not found:
        print(f'No checkpoints found in {CHK_DIR}')
        return
    print('Checkpoints found:')
    for n in MODEL_REGISTRY:
        if n in found:
            seeds = [s for s, _ in found[n]]
            print(f'  {n:<12s} seeds={seeds}')
    print()

    # 2. Load test data
    print('Loading test data...')
    _, _, test_df = data_loader.load_all_splits()
    X_te, y_te, m_te = data_loader.make_windows(test_df)
    print(f'  test windows: {X_te.shape}, label balance: {y_te.mean():.1%}\n')

    # 3. Build model + evaluate per seed
    all_metrics = {}
    for name in MODEL_REGISTRY:
        if name not in found:
            continue
        cls = MODEL_REGISTRY[name]
        per_seed = []
        for seed, path in found[name]:
            model = cls(n_features=config.N_FEATURES,
                        window_size=config.WINDOW_SIZE)
            sd = torch.load(path, map_location='cpu', weights_only=True)
            model.load_state_dict(sd)
            m = compute_time_aware_metrics(model, X_te, m_te, device)
            m['seed'] = seed
            per_seed.append(m)
            print(f'  [{name} s={seed}] '
                  f'P={m["precision"]:.3f} R={m["recall"]:.3f} F1={m["f1"]:.3f}  '
                  f'DR@5s={m["dr_at_5s"]:.3f} ADD={m["add_s"]:.2f}s '
                  f'MTBFA={m["mtbfa_h"]:.3f}h')
        all_metrics[name] = per_seed

    # 4. Aggregate and print summary table
    print('\n' + '=' * 100)
    print('SUMMARY (mean ± half-CI over available seeds)')
    print('=' * 100)
    hdr = f'{"Model":<12} {"Prec":>14} {"Recall":>14} {"F1":>14} ' \
          f'{"DR@5s":>14} {"ADD(s)":>12} {"MTBFA(h)":>14}'
    print(hdr); print('-' * 100)

    def _fmt(agg, key, fmt='{:.3f}', unit=''):
        m  = agg.get(key, {}).get('mean', float('nan'))
        lo = agg.get(key, {}).get('ci_lo', float('nan'))
        hi = agg.get(key, {}).get('ci_hi', float('nan'))
        if math.isnan(m): return '   ---'
        if math.isinf(m): return '    inf'
        err = max(m - lo, hi - m) if not math.isnan(lo) else 0
        if math.isnan(err): err = 0
        return f'{fmt.format(m)}±{fmt.format(err)}{unit}'

    for name in MODEL_REGISTRY:
        if name not in all_metrics:
            continue
        agg = aggregate_metrics(all_metrics[name])
        print(f'{name:<12} '
              f'{_fmt(agg, "precision"):>14} '
              f'{_fmt(agg, "recall"):>14} '
              f'{_fmt(agg, "f1"):>14} '
              f'{_fmt(agg, "dr_at_5s"):>14} '
              f'{_fmt(agg, "add_s", "{:.2f}"):>12} '
              f'{_fmt(agg, "mtbfa_h"):>14}')
    print('=' * 100)

    # 5. Differentiation diagnostic
    print('\nDIFFERENTIATION DIAGNOSTIC')
    print('-' * 60)
    for key, label in [('precision', 'Precision'),
                       ('recall', 'Recall'),
                       ('f1', 'F1'),
                       ('dr_at_5s', 'DR@5s'),
                       ('add_s', 'ADD'),
                       ('mtbfa_h', 'MTBFA')]:
        means = []
        for name in all_metrics:
            agg = aggregate_metrics(all_metrics[name])
            v = agg.get(key, {}).get('mean', float('nan'))
            if not math.isnan(v) and not math.isinf(v):
                means.append(v)
        if len(means) >= 2:
            spread = max(means) - min(means)
            ratio = max(means) / min(means) if min(means) > 0 else float('inf')
            print(f'  {label:<10}  range=[{min(means):.3f}, {max(means):.3f}]  '
                  f'spread={spread:.3f}  ratio={ratio:.2f}x')


if __name__ == '__main__':
    main()
