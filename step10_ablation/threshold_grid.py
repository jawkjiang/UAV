"""
Threshold sensitivity grid: DR@Δt vs MTBFA threshold matrix.

Using the best-performing model from main pipeline results, sweeps:
  - DR time threshold: {1, 3, 5, 10, 15} s
  - MTBFA threshold: {0.05, 0.1, 0.15, 0.2} h

Reports which (Δt, MTBFA_thresh) combinations pass the deployability standard.

Usage:
    cd ~/projects/UAV
    python step10_ablation/threshold_grid.py
"""
import sys, os, json, math, pickle
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'matlab_pipeline'))

OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'matlab_pipeline', 'output')
METRICS_PKL = os.path.join(OUTPUT_DIR, 'all_metrics.pkl')
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'ablation_threshold_grid.json')

DR_THRESHOLDS   = [1, 3, 5, 10, 15]      # seconds
MTBFA_THRESHOLDS = [0.05, 0.1, 0.15, 0.2] # hours
DR_PASS_MIN     = 0.85
DR_KEYS = {1: 'dr_at_1s', 3: 'dr_at_3s', 5: 'dr_at_5s', 10: 'dr_at_10s', 15: 'dr_at_15s'}


def _mean(vals):
    finite = [v for v in vals if v is not None and not math.isnan(float(v))
              and not math.isinf(float(v))]
    return float(np.mean(finite)) if finite else float('nan')


def main():
    if not os.path.exists(METRICS_PKL):
        print(f'ERROR: {METRICS_PKL} not found. Run main pipeline first.')
        return

    with open(METRICS_PKL, 'rb') as f:
        all_metrics = pickle.load(f)

    results = {}
    for model_name, seed_list in all_metrics.items():
        model_entry = {}
        for dt in DR_THRESHOLDS:
            dr_key = DR_KEYS.get(dt)
            if dr_key is None:
                continue
            dr_vals = [m.get(dr_key, float('nan')) for m in seed_list]
            mtbfa_vals = [m.get('mtbfa_h', float('nan')) for m in seed_list]
            dr_mean   = _mean(dr_vals)
            mtbfa_mean = _mean([v if not math.isinf(v) else float('nan') for v in mtbfa_vals])

            for mtbfa_thresh in MTBFA_THRESHOLDS:
                cell_key = f'dt{dt}_mtbfa{mtbfa_thresh}'
                passes = (dr_mean >= DR_PASS_MIN and
                          (math.isinf(mtbfa_mean) or mtbfa_mean >= mtbfa_thresh))
                model_entry[cell_key] = {
                    'dr_mean': dr_mean,
                    'mtbfa_mean': mtbfa_mean if not math.isinf(mtbfa_mean) else None,
                    'passes': passes,
                }
        results[model_name] = model_entry

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Threshold grid saved to {OUTPUT_PATH}')

    # Print summary table
    header = f"{'Model':<14}" + ''.join(f"  dt={dt}s" for dt in DR_THRESHOLDS)
    print('\nDeployability pass matrix (MTBFA≥0.1h, DR≥85%)')
    print(header)
    print('-' * len(header))
    for model_name, entry in results.items():
        row = f'{model_name:<14}'
        for dt in DR_THRESHOLDS:
            cell = entry.get(f'dt{dt}_mtbfa0.1', {})
            row += '     ✓' if cell.get('passes') else '     ✗'
        print(row)


if __name__ == '__main__':
    main()
