"""
Time-aware evaluation for the MATLAB-based UAV spoofing detection pipeline.

Metrics: DR@Δt, ADD, MTBFA, precision, recall, F1.
"""
import math
from collections import defaultdict
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import config

_DR_THRESHOLDS = [1.0, 3.0, 5.0, 10.0, 15.0]


def _predict(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    """Batch inference → binary predictions (threshold 0.5). Model returns probabilities."""
    model.eval()
    model.to(device)
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                        batch_size=config.BATCH_SIZE, shuffle=False)
    probs_list = []
    with torch.no_grad():
        for (batch,) in loader:
            # Model already outputs sigmoid probs
            probs_list.append(model(batch.to(device)).squeeze(-1).cpu().numpy())
    probs = np.concatenate(probs_list)
    return (probs >= 0.5).astype(np.int32)


def _is_nan(v: Any) -> bool:
    if v is None:
        return True
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def _count_fp_events(preds: np.ndarray) -> int:
    """Count rising-edge runs of 1s (FP events) in a binary sequence."""
    if len(preds) == 0:
        return 0
    prev = np.concatenate(([0], preds[:-1]))
    return int(np.sum((preds == 1) & (prev == 0)))


def _fp_event_durations(preds: np.ndarray, times: np.ndarray) -> list:
    """Return list of durations (seconds) for each FP run in a binary prediction sequence."""
    if len(preds) == 0:
        return []
    durations = []
    in_event = False
    t_start = 0.0
    for i, (p, t) in enumerate(zip(preds, times)):
        if p == 1 and not in_event:
            in_event = True
            t_start = float(t)
        elif p == 0 and in_event:
            in_event = False
            durations.append(float(t) - t_start)
    if in_event and len(times) > 0:
        durations.append(float(times[-1]) - t_start)
    return durations


def _prf(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
    return prec, rec, f1


def compute_time_aware_metrics(model: nn.Module, X_test: np.ndarray,
                               meta_list: list, device: torch.device = None) -> dict:
    """
    Compute DR@Δt, ADD, MTBFA and conventional metrics for one model.

    meta_list: list of dicts with keys
        flight_id, onset_time (NaN for clean), window_end_time, attack_type
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    preds = _predict(model, X_test, device)
    model.to('cpu')

    # Rebuild y_true from meta (end-point labelling)
    y_true = np.array(
        [1 if (not _is_nan(m['onset_time']) and m['window_end_time'] >= m['onset_time'])
         else 0 for m in meta_list], dtype=np.int32)

    # Group by flight
    flights = defaultdict(list)
    for m, p in zip(meta_list, preds):
        flights[m['flight_id']].append(
            (float(m['window_end_time']), int(p), m['onset_time']))
    for fid in flights:
        flights[fid].sort()

    detection_delays = []
    fp_events = 0
    fp_event_durations_s = []
    normal_time_s = 0.0

    for fid, windows in flights.items():
        times  = np.array([w[0] for w in windows])
        wred  = np.array([w[1] for w in windows], dtype=np.int32)
        onset  = windows[0][2]
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
            fp_event_durations_s.extend(_fp_event_durations(pre_preds, pre_times))
        else:
            if len(times) > 1:
                normal_time_s += float(times[-1] - times[0])
            fp_events += _count_fp_events(wred)
            fp_event_durations_s.extend(_fp_event_durations(wred, times))

    n_total    = len(detection_delays)
    n_detected = sum(1 for d in detection_delays if not math.isinf(d))

    dr = {}
    for th in _DR_THRESHOLDS:
        key = f'dr_at_{int(th)}s' if th == int(th) else f'dr_at_{th}s'
        dr[key] = (sum(1 for d in detection_delays if d <= th) / n_total
                   if n_total > 0 else float('nan'))

    finite = [d for d in detection_delays if not math.isinf(d)]
    add_s  = float(np.mean(finite)) if finite else float('nan')

    normal_time_h = normal_time_s / 3600.0
    mtbfa_h = math.inf if fp_events == 0 else normal_time_h / fp_events

    prec, rec, f1 = _prf(y_true, preds)

    fp_windows = int(np.sum((preds == 1) & (y_true == 0)))

    return {**dr,
            'add_s': add_s, 'mtbfa_h': mtbfa_h,
            'precision': prec, 'recall': rec, 'f1': f1,
            'n_attacks_detected': n_detected, 'n_attacks_total': n_total,
            'n_fp_events': fp_events, 'normal_time_h': normal_time_h,
            'fp_windows': fp_windows,
            'fp_events': fp_events,
            'fp_event_durations_s': fp_event_durations_s}


def evaluate_all_models(model_results_dict: dict, X_test: np.ndarray,
                        meta_list: list, device: torch.device = None) -> dict:
    """Evaluate every model across all seeds. Returns {model_name: [metric_dict, ...]}."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_metrics = {}
    for model_name, seed_results in model_results_dict.items():
        print(f'\nEvaluating {model_name} ({len(seed_results)} seeds)...')
        metrics_list = []
        for result in seed_results:
            m = compute_time_aware_metrics(result['model'], X_test, meta_list, device)
            m['seed'] = result.get('seed', '?')
            metrics_list.append(m)
            print(f"  s={m['seed']}  F1={m['f1']:.3f}  DR@5s={m['dr_at_5s']:.3f}  "
                  f"ADD={m['add_s']:.2f}s  MTBFA={m['mtbfa_h']:.3f}h")
        all_metrics[model_name] = metrics_list
    return all_metrics
