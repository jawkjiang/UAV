"""
Pareto sweep: for each (architecture, seed), evaluate (DR@5s, MTBFA) at
every probability threshold p in a fine grid (0.50 .. 0.99 by 0.01),
on both validation and test splits. Save the full curve so that:
  * Pareto front plots can show the (DR, MTBFA) trade-off shape
  * Multiple MTBFA targets ({0.05, 0.10, 0.20, 0.50}h) can be evaluated
    post-hoc without re-running.

Output: matlab_pipeline/output/calibrated_threshold/pareto_sweep.json
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

CHK_DIR = os.path.join(config.OUTPUT_DIR, 'checkpoints')
OUT_DIR = os.path.join(config.OUTPUT_DIR, 'calibrated_threshold')
os.makedirs(OUT_DIR, exist_ok=True)

THRESH_GRID = np.round(np.arange(0.50, 1.00, 0.01), 2)


def _probs(model, X, device):
    model.eval(); model.to(device)
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                        batch_size=config.BATCH_SIZE, shuffle=False)
    probs = []
    with torch.no_grad():
        for (b,) in loader:
            probs.append(model(b.to(device)).squeeze(-1).cpu().numpy())
    return np.concatenate(probs)


def _is_nan(v):
    if v is None: return True
    try: return math.isnan(float(v))
    except: return False


def _count_fp(p):
    if len(p) == 0: return 0
    prev = np.concatenate(([0], p[:-1]))
    return int(np.sum((p == 1) & (prev == 0)))


def _metrics_at(probs, meta_list, thresh):
    preds = (probs >= thresh).astype(np.int32)
    flights = defaultdict(list)
    for m, p in zip(meta_list, preds):
        flights[m['flight_id']].append((float(m['window_end_time']), int(p), m['onset_time']))
    for fid in flights:
        flights[fid].sort()
    delays = []; fp_events = 0; nt_s = 0.0
    for fid, w in flights.items():
        t = np.array([x[0] for x in w])
        r = np.array([x[1] for x in w], dtype=np.int32)
        onset = w[0][2]
        if not _is_nan(onset):
            post = t >= float(onset)
            pt = t[post]; pr = r[post]
            di = np.where(pr == 1)[0]
            d = float(pt[di[0]] - float(onset)) if len(di) > 0 else math.inf
            delays.append(max(d, 0.0))
            pre = t < float(onset)
            pt2 = t[pre]; pr2 = r[pre]
            if len(pt2) > 1: nt_s += float(pt2[-1] - pt2[0])
            fp_events += _count_fp(pr2)
        else:
            if len(t) > 1: nt_s += float(t[-1] - t[0])
            fp_events += _count_fp(r)
    n = len(delays)
    dr5 = (sum(1 for d in delays if d <= 5.0) / n) if n > 0 else float('nan')
    finite = [d for d in delays if not math.isinf(d)]
    add = float(np.mean(finite)) if finite else float('nan')
    nt_h = nt_s / 3600.0
    mt = math.inf if fp_events == 0 else nt_h / fp_events
    return {'dr_at_5s': dr5, 'add_s': add, 'mtbfa_h': mt, 'fp_events': fp_events}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Pareto sweep  device={device}  grid_size={len(THRESH_GRID)}')
    _, val_df, test_df = _dl.load_all_splits()
    X_va, _, m_va = _dl.make_windows(val_df)
    X_te, _, m_te = _dl.make_windows(test_df)
    print(f'  val:{X_va.shape}  test:{X_te.shape}')

    sweep = {}
    for mn, cls in MODEL_REGISTRY.items():
        per_seed = []
        for seed in config.SEEDS:
            ckpt = os.path.join(CHK_DIR, f'{mn}_seed{seed}.pt')
            if not os.path.exists(ckpt): continue
            m = cls(n_features=config.N_FEATURES, window_size=config.WINDOW_SIZE)
            m.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
            p_va = _probs(m, X_va, device)
            p_te = _probs(m, X_te, device)
            curve = []
            for t in THRESH_GRID:
                v = _metrics_at(p_va, m_va, t)
                te = _metrics_at(p_te, m_te, t)
                curve.append({'p': float(t),
                              'val_dr5': v['dr_at_5s'], 'val_mtbfa': v['mtbfa_h'],
                              'test_dr5': te['dr_at_5s'], 'test_mtbfa': te['mtbfa_h'],
                              'test_add': te['add_s']})
            per_seed.append({'seed': seed, 'curve': curve})
            print(f'  {mn:<12} s={seed} done ({len(curve)} thresholds)')
        if per_seed:
            sweep[mn] = per_seed

    # Aggregate to mean curve per model
    aggregated = {}
    for mn, seeds in sweep.items():
        n_thresh = len(seeds[0]['curve'])
        agg_curve = []
        for i in range(n_thresh):
            p = seeds[0]['curve'][i]['p']
            v_dr = [s['curve'][i]['val_dr5'] for s in seeds]
            v_mt = [s['curve'][i]['val_mtbfa'] for s in seeds]
            t_dr = [s['curve'][i]['test_dr5'] for s in seeds]
            t_mt = [s['curve'][i]['test_mtbfa'] for s in seeds]
            t_ad = [s['curve'][i]['test_add'] for s in seeds]
            def _safe_mean(vs):
                vs = [v for v in vs if not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))]
                return float(np.mean(vs)) if vs else float('nan')
            agg_curve.append({'p': p,
                              'val_dr5_mean': _safe_mean(v_dr),
                              'val_mtbfa_mean': _safe_mean(v_mt),
                              'test_dr5_mean': _safe_mean(t_dr),
                              'test_mtbfa_mean': _safe_mean(t_mt),
                              'test_add_mean': _safe_mean(t_ad)})
        aggregated[mn] = agg_curve

    out_pkl  = os.path.join(OUT_DIR, 'pareto_sweep_per_seed.pkl')
    out_json = os.path.join(OUT_DIR, 'pareto_sweep.json')
    with open(out_pkl, 'wb') as f: pickle.dump(sweep, f)
    def _clean(v):
        if isinstance(v, float):
            if math.isnan(v): return 'nan'
            if math.isinf(v): return 'inf'
        if isinstance(v, dict): return {k: _clean(x) for k, x in v.items()}
        if isinstance(v, list): return [_clean(x) for x in v]
        return v
    with open(out_json, 'w') as f: json.dump(_clean(aggregated), f, indent=2)
    print(f'\nWrote {out_json}')


if __name__ == '__main__':
    main()
