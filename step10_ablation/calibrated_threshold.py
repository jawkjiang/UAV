"""
Validation-calibrated threshold experiment.

For each (architecture, seed):
  1. Compute probabilities on val and test windows.
  2. Sweep probability thresholds p in [0.05 .. 0.99].
  3. Find smallest p* such that MTBFA(val, p*) >= 0.10 h
     (operational FA budget). If no such p exists, use p=0.99 as fallback.
  4. Apply p* on test set; report DR@5s, ADD, MTBFA(test) under p*.

Compare to default p=0.5 results from the main pipeline.

Output: matlab_pipeline/output/calibrated_threshold/calibrated_results.json
"""
import sys, os, json, math, pickle
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'matlab_pipeline'))
import config
import data_loader as _dl
from models import MODEL_REGISTRY

CHK_DIR  = os.path.join(config.OUTPUT_DIR, 'checkpoints')
OUT_DIR  = os.path.join(config.OUTPUT_DIR, 'calibrated_threshold')
os.makedirs(OUT_DIR, exist_ok=True)

MTBFA_TARGET = 0.10  # hours

# Sweep grid: 0.05 .. 0.99 in steps of 0.02 (48 thresholds)
THRESH_GRID = np.round(np.arange(0.05, 1.00, 0.02), 2)


def _probs(model, X, device):
    model.eval(); model.to(device)
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                        batch_size=config.BATCH_SIZE, shuffle=False)
    probs_list = []
    with torch.no_grad():
        for (batch,) in loader:
            probs_list.append(model(batch.to(device)).squeeze(-1).cpu().numpy())
    return np.concatenate(probs_list)


def _is_nan(v):
    if v is None: return True
    try: return math.isnan(float(v))
    except: return False


def _count_fp_events(preds):
    if len(preds) == 0: return 0
    prev = np.concatenate(([0], preds[:-1]))
    return int(np.sum((preds == 1) & (prev == 0)))


def _metrics_at_threshold(probs, meta_list, thresh):
    """Compute DR@5s, ADD, MTBFA for given probability threshold."""
    preds = (probs >= thresh).astype(np.int32)
    flights = defaultdict(list)
    for m, p in zip(meta_list, preds):
        flights[m['flight_id']].append((float(m['window_end_time']), int(p), m['onset_time']))
    for fid in flights:
        flights[fid].sort()

    detection_delays = []
    fp_events = 0
    normal_time_s = 0.0

    for fid, windows in flights.items():
        times = np.array([w[0] for w in windows])
        wred  = np.array([w[1] for w in windows], dtype=np.int32)
        onset = windows[0][2]
        attacked = not _is_nan(onset)
        if attacked:
            post = times >= float(onset)
            post_times = times[post]; post_preds = wred[post]
            det_idx = np.where(post_preds == 1)[0]
            delay = float(post_times[det_idx[0]] - float(onset)) if len(det_idx) > 0 else math.inf
            detection_delays.append(max(delay, 0.0))
            pre = times < float(onset)
            pre_times = times[pre]; pre_preds = wred[pre]
            if len(pre_times) > 1:
                normal_time_s += float(pre_times[-1] - pre_times[0])
            fp_events += _count_fp_events(pre_preds)
        else:
            if len(times) > 1:
                normal_time_s += float(times[-1] - times[0])
            fp_events += _count_fp_events(wred)

    n_total = len(detection_delays)
    dr5 = (sum(1 for d in detection_delays if d <= 5.0) / n_total) if n_total > 0 else float('nan')
    finite = [d for d in detection_delays if not math.isinf(d)]
    add_s = float(np.mean(finite)) if finite else float('nan')
    normal_time_h = normal_time_s / 3600.0
    mtbfa_h = math.inf if fp_events == 0 else normal_time_h / fp_events
    return {'dr_at_5s': dr5, 'add_s': add_s, 'mtbfa_h': mtbfa_h,
            'fp_events': fp_events, 'normal_time_h': normal_time_h}


def calibrate_one(model, X_va, m_va, X_te, m_te, device):
    p_va = _probs(model, X_va, device)
    p_te = _probs(model, X_te, device)
    # Find smallest threshold s.t. MTBFA(val) >= 0.10 h
    chosen_thresh = 0.99
    for t in THRESH_GRID:
        m = _metrics_at_threshold(p_va, m_va, t)
        mt = m['mtbfa_h']
        if (math.isinf(mt) or mt >= MTBFA_TARGET):
            chosen_thresh = float(t)
            break
    # Evaluate on test with chosen threshold
    test_metrics = _metrics_at_threshold(p_te, m_te, chosen_thresh)
    val_metrics  = _metrics_at_threshold(p_va, m_va, chosen_thresh)
    # Also report default p=0.5 baseline for reference
    test_at_05 = _metrics_at_threshold(p_te, m_te, 0.5)
    return {
        'p_star': chosen_thresh,
        'val_at_p_star':  val_metrics,
        'test_at_p_star': test_metrics,
        'test_at_p_05':   test_at_05,
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Calibrated-threshold experiment  device={device}')

    print('Loading data splits...')
    _, val_df, test_df = _dl.load_all_splits()
    X_va, _, m_va = _dl.make_windows(val_df)
    X_te, _, m_te = _dl.make_windows(test_df)
    print(f'  val windows: {X_va.shape}  test windows: {X_te.shape}')

    results = {}
    for model_name, cls in MODEL_REGISTRY.items():
        per_seed = []
        for seed in config.SEEDS:
            ckpt = os.path.join(CHK_DIR, f'{model_name}_seed{seed}.pt')
            if not os.path.exists(ckpt):
                continue
            m = cls(n_features=config.N_FEATURES, window_size=config.WINDOW_SIZE)
            m.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
            r = calibrate_one(m, X_va, m_va, X_te, m_te, device)
            r['seed'] = seed
            per_seed.append(r)
            print(f'  {model_name:<12} s={seed}  '
                  f"p*={r['p_star']:.2f}  "
                  f"val[MTBFA={r['val_at_p_star']['mtbfa_h']:.3f}h]  "
                  f"test[DR@5s={r['test_at_p_star']['dr_at_5s']:.3f} "
                  f"MTBFA={r['test_at_p_star']['mtbfa_h']:.3f}h] "
                  f"vs default[DR@5s={r['test_at_p_05']['dr_at_5s']:.3f} "
                  f"MTBFA={r['test_at_p_05']['mtbfa_h']:.3f}h]")
        if per_seed:
            results[model_name] = per_seed

    # Aggregate
    summary = {}
    for mn, seeds in results.items():
        ps = [s['p_star'] for s in seeds]
        dr5 = [s['test_at_p_star']['dr_at_5s'] for s in seeds]
        add = [s['test_at_p_star']['add_s'] for s in seeds if not math.isinf(s['test_at_p_star']['add_s'])]
        mt_test = [s['test_at_p_star']['mtbfa_h'] for s in seeds]
        mt_test_finite = [m for m in mt_test if not math.isinf(m)]
        dr5_05 = [s['test_at_p_05']['dr_at_5s'] for s in seeds]
        mt_05 = [s['test_at_p_05']['mtbfa_h'] for s in seeds if not math.isinf(s['test_at_p_05']['mtbfa_h'])]
        summary[mn] = {
            'p_star_mean':            float(np.mean(ps)),
            'p_star_std':             float(np.std(ps)),
            'calibrated_dr_at_5s':    float(np.mean(dr5)),
            'calibrated_dr_at_5s_std':float(np.std(dr5)),
            'calibrated_add_s':       float(np.mean(add)) if add else float('nan'),
            'calibrated_mtbfa_h':     float(np.mean(mt_test_finite)) if mt_test_finite else float('inf'),
            'default_dr_at_5s':       float(np.mean(dr5_05)),
            'default_mtbfa_h':        float(np.mean(mt_05)) if mt_05 else float('inf'),
            'dr_drop':                float(np.mean(dr5_05)) - float(np.mean(dr5)),
        }

    out_pkl  = os.path.join(OUT_DIR, 'calibrated_per_seed.pkl')
    out_json = os.path.join(OUT_DIR, 'calibrated_results.json')
    with open(out_pkl, 'wb') as f: pickle.dump(results, f)
    def _clean(v):
        if isinstance(v, float):
            if math.isnan(v): return 'nan'
            if math.isinf(v): return 'inf'
        if isinstance(v, dict): return {k: _clean(x) for k, x in v.items()}
        if isinstance(v, list): return [_clean(x) for x in v]
        return v
    with open(out_json, 'w') as f: json.dump(_clean(summary), f, indent=2)

    print('\n' + '=' * 100)
    print('Validation-calibrated threshold summary (target MTBFA >= 0.10 h on val)')
    print('=' * 100)
    print(f'{"Model":<12} {"p*":>6} {"def DR@5s":>11} {"cal DR@5s":>11} {"DR drop":>9} '
          f'{"def MTBFA":>10} {"cal MTBFA":>10}')
    print('-' * 100)
    for mn, s in summary.items():
        cal_mt = '   inf' if math.isinf(s['calibrated_mtbfa_h']) else f"{s['calibrated_mtbfa_h']:.3f}h"
        def_mt = '   inf' if math.isinf(s['default_mtbfa_h'])   else f"{s['default_mtbfa_h']:.3f}h"
        print(f"{mn:<12} {s['p_star_mean']:>6.2f} "
              f"{s['default_dr_at_5s']:>11.3f} {s['calibrated_dr_at_5s']:>11.3f} "
              f"{s['dr_drop']:>+9.3f} "
              f"{def_mt:>10} {cal_mt:>10}")
    print('=' * 100)
    print(f'\nResults: {out_json}')


if __name__ == '__main__':
    main()
