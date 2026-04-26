"""
Generate all paper figures from the MATLAB pipeline results.

Reads: matlab_pipeline/output/all_metrics.pkl
Writes: docs/figures/*.pdf

Usage:
    cd ~/projects/UAV
    python matlab_pipeline/visualize.py

Output filenames match the \\includegraphics references in docs/main.tex:
    fig_c1_dr_at_delta_t_curves.pdf
    fig_e1_pareto_frontier.pdf
    fig_b3_precision_mtbfa_paradox.pdf
    fig_b4_fp_windows_vs_events.pdf
    fig_d2_fp_duration_boxplot.pdf
    fp_aggregation_example.pdf
    overall_bars.pdf
    time_metrics.pdf
    per_model_dr5s.pdf
"""
import sys, os, pickle, json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from statistical_tests import aggregate_metrics

# ── Output paths ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR    = os.path.join(BASE_DIR, 'matlab_pipeline', 'output')
FIG_DIR    = os.path.join(BASE_DIR, 'docs', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3, 'lines.linewidth': 1.8,
})

MODEL_ORDER  = ['CNN', 'LSTM', 'BiLSTM', 'GRU', 'CNN-LSTM', 'TCN', 'Transformer']
MODEL_COLORS = {
    'CNN': '#4477AA', 'LSTM': '#EE6677', 'BiLSTM': '#228833',
    'GRU': '#CCBB44', 'CNN-LSTM': '#66CCEE', 'TCN': '#AA3377',
    'Transformer': '#BBBBBB',
}
DR_THRESHOLDS = [1, 3, 5, 10, 15]
DR_KEYS       = {t: f'dr_at_{t}s' for t in DR_THRESHOLDS}


def _savefig(fig, name: str):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches='tight')
    print(f'  → {path}')
    plt.close(fig)


def _get(agg: dict, key: str, field: str = 'mean', default: float = float('nan')):
    return agg.get(key, {}).get(field, default)


# ── Figure 1: DR@Δt curves with CI bands ─────────────────────────────────────
def fig_dr_curves(all_metrics: dict):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ts = DR_THRESHOLDS
    for mn in MODEL_ORDER:
        if mn not in all_metrics:
            continue
        agg = aggregate_metrics(all_metrics[mn])
        means = [_get(agg, DR_KEYS[t]) for t in ts]
        lo    = [_get(agg, DR_KEYS[t], 'ci_lo') for t in ts]
        hi    = [_get(agg, DR_KEYS[t], 'ci_hi') for t in ts]
        c = MODEL_COLORS[mn]
        ax.plot(ts, means, 'o-', color=c, label=mn, markersize=4)
        ax.fill_between(ts, lo, hi, color=c, alpha=0.12)

    ax.axhline(0.85, color='gray', ls='--', lw=1.2, label='DR threshold (85%)')
    ax.axvline(5,    color='gray', ls=':',  lw=1.0, label='5 s budget')
    ax.set_xlabel('Detection Time Budget Δt (s)')
    ax.set_ylabel('Detection Rate DR@Δt')
    ax.set_xlim([0, 16]); ax.set_ylim([0, 1.02])
    ax.set_xticks(ts)
    ax.legend(ncol=2, loc='lower right', framealpha=0.9)
    ax.set_title('DR@Δt for Seven Detectors (mean ± 95% CI, 5 seeds)')
    _savefig(fig, 'fig_c1_dr_at_delta_t_curves.pdf')


# ── Figure 2: Pareto (DR@5s vs MTBFA) with CI ellipses ───────────────────────
def fig_pareto(all_metrics: dict):
    fig, ax = plt.subplots(figsize=(6, 5))
    for mn in MODEL_ORDER:
        if mn not in all_metrics:
            continue
        agg = aggregate_metrics(all_metrics[mn])
        dr5   = _get(agg, 'dr_at_5s')
        mtbfa = _get(agg, 'mtbfa_h')
        dr5_lo  = _get(agg, 'dr_at_5s', 'ci_lo'); dr5_hi  = _get(agg, 'dr_at_5s', 'ci_hi')
        mt_lo   = _get(agg, 'mtbfa_h',  'ci_lo'); mt_hi   = _get(agg, 'mtbfa_h',  'ci_hi')

        c = MODEL_COLORS[mn]
        if not (math.isnan(dr5) or math.isnan(mtbfa) or
                math.isinf(mtbfa) or math.isinf(dr5)):
            ax.scatter(mtbfa, dr5, color=c, zorder=5, s=60)
            ax.annotate(mn, (mtbfa, dr5), textcoords='offset points',
                        xytext=(5, 3), fontsize=8, color=c)
            dr_range = max((dr5_hi - dr5_lo) / 2, 0.005)
            mt_range = max((mt_hi  - mt_lo)  / 2, 0.002)
            ell = Ellipse((mtbfa, dr5), width=mt_range*2, height=dr_range*2,
                          color=c, alpha=0.2, lw=0)
            ax.add_patch(ell)

    ax.axhline(0.85, color='gray', ls='--', lw=1.0)
    ax.axvline(0.10, color='gray', ls=':',  lw=1.0)
    ax.set_xlabel('Mean Time Between False Alarms (h)')
    ax.set_ylabel('Detection Rate DR@5s')
    ax.set_title('Pareto: DR@5s vs MTBFA (ellipses = 95% CI)')
    _savefig(fig, 'fig_e1_pareto_frontier.pdf')


# ── Figure 3: Overall metrics bar chart (mean ± CI) ──────────────────────────
def fig_overall_bars(all_metrics: dict):
    metrics = ['precision', 'recall', 'f1', 'dr_at_5s']
    labels  = ['Precision', 'Recall', 'F1', 'DR@5s']
    fig, axes = plt.subplots(1, 4, figsize=(12, 4), sharey=False)

    for ax, mkey, mlabel in zip(axes, metrics, labels):
        models = [mn for mn in MODEL_ORDER if mn in all_metrics]
        means, errs = [], []
        for mn in models:
            agg = aggregate_metrics(all_metrics[mn])
            m   = _get(agg, mkey)
            lo  = _get(agg, mkey, 'ci_lo')
            hi  = _get(agg, mkey, 'ci_hi')
            means.append(m)
            errs.append([(m - lo) if not math.isnan(lo) else 0,
                         (hi - m) if not math.isnan(hi) else 0])

        x = np.arange(len(models))
        bars = ax.bar(x, means, color=[MODEL_COLORS[mn] for mn in models],
                      alpha=0.8, width=0.6)
        if errs:
            err_lo = [e[0] for e in errs]
            err_hi = [e[1] for e in errs]
            ax.errorbar(x, means, yerr=[err_lo, err_hi], fmt='none',
                        color='black', capsize=3, lw=1.5)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(mlabel)
        ax.set_ylim([0, 1.05])
        if mkey == 'dr_at_5s':
            ax.axhline(0.85, color='gray', ls='--', lw=1.0)

    fig.suptitle('Overall Performance Metrics (mean ± 95% CI, 5 seeds)', y=1.02)
    fig.tight_layout()
    _savefig(fig, 'overall_bars.pdf')


# ── Figure 4: ADD and MTBFA bars ──────────────────────────────────────────────
def fig_time_metrics(all_metrics: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    models = [mn for mn in MODEL_ORDER if mn in all_metrics]
    x = np.arange(len(models))

    for ax, mkey, ylabel, thresh, thresh_dir in [
        (ax1, 'add_s', 'ADD (s)', None, None),
        (ax2, 'mtbfa_h', 'MTBFA (h)', 0.10, 'ge'),
    ]:
        means, lo_errs, hi_errs = [], [], []
        for mn in models:
            agg = aggregate_metrics(all_metrics[mn])
            m  = _get(agg, mkey)
            lo = _get(agg, mkey, 'ci_lo')
            hi = _get(agg, mkey, 'ci_hi')
            if math.isinf(m): m = 10.0
            if math.isinf(lo): lo = m
            if math.isinf(hi): hi = m
            means.append(m if not math.isnan(m) else 0)
            lo_errs.append(max(0, m - lo) if not math.isnan(lo) else 0)
            hi_errs.append(max(0, hi - m) if not math.isnan(hi) else 0)

        bars = ax.bar(x, means, color=[MODEL_COLORS[mn] for mn in models],
                      alpha=0.8, width=0.6)
        ax.errorbar(x, means, yerr=[lo_errs, hi_errs], fmt='none',
                    color='black', capsize=3, lw=1.5)
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(ylabel)
        if thresh:
            ax.axhline(thresh, color='gray', ls='--', lw=1.0,
                       label=f'Threshold ({thresh} h)')
            ax.legend(fontsize=8)

    fig.suptitle('Time-Aware Metrics (mean ± 95% CI)', y=1.02)
    fig.tight_layout()
    _savefig(fig, 'time_metrics.pdf')


# ── Figure 5: FP aggregation conceptual example ───────────────────────────────
def fig_fp_aggregation():
    fig, ax = plt.subplots(figsize=(8, 2.5))
    t = np.arange(0, 30, 1)
    preds = np.zeros(30, dtype=int)
    # Three separate FP bursts: windows 4-6, 12-14, 22-25
    for s, e in [(4, 7), (12, 15), (22, 26)]:
        preds[s:e] = 1

    ax.step(t, preds, where='post', color='#EE6677', lw=2, label='Window predictions')
    ax.fill_between(t, preds, step='post', alpha=0.15, color='#EE6677')

    # Mark FP events with arrows
    for i, (s, _) in enumerate([(4, 7), (12, 15), (22, 26)]):
        ax.annotate(f'FP event {i+1}', xy=(s, 1), xytext=(s+0.5, 1.25),
                    fontsize=8, color='#AA3377',
                    arrowprops=dict(arrowstyle='->', color='#AA3377', lw=1.2))

    ax.set_xlim([-0.5, 29.5]); ax.set_ylim([-0.15, 1.6])
    ax.set_xlabel('Time (window index)')
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Normal', 'FP'])
    ax.set_title('FP Window Aggregation: 9 FP windows → 3 FP events (MTBFA denominator = 3)')
    ax.grid(axis='x', alpha=0.3)
    _savefig(fig, 'fp_aggregation_example.pdf')


# ── Figure 6: Per-attack-type DR@5s ──────────────────────────────────────────
def fig_per_attack(all_metrics: dict):
    attack_types = ['step', 'drift', 'delay', 'takeover']
    attack_colors = {'step': '#e74c3c', 'drift': '#3498db',
                     'delay': '#f39c12', 'takeover': '#9b59b6'}

    # Collect per-attack metrics by looking at meta_list attack_type breakdown
    # The existing metrics are aggregated across all attacks;
    # we use the confusion-matrix-level metrics already computed per attack type
    # For now, show overall DR@Δt for each model, stratified if available
    fig, ax = plt.subplots(figsize=(8, 4.5))
    models = [mn for mn in MODEL_ORDER if mn in all_metrics]
    x = np.arange(len(models))
    width = 0.55

    # DR@5s aggregated
    means = []
    err_lo, err_hi = [], []
    for mn in models:
        agg = aggregate_metrics(all_metrics[mn])
        m  = _get(agg, 'dr_at_5s')
        lo = _get(agg, 'dr_at_5s', 'ci_lo')
        hi = _get(agg, 'dr_at_5s', 'ci_hi')
        means.append(m if not math.isnan(m) else 0)
        err_lo.append(max(0, m - lo) if not math.isnan(lo) else 0)
        err_hi.append(max(0, hi - m) if not math.isnan(hi) else 0)

    bars = ax.bar(x, means, width, color=[MODEL_COLORS[mn] for mn in models],
                  alpha=0.85)
    ax.errorbar(x, means, yerr=[err_lo, err_hi], fmt='none',
                color='black', capsize=4, lw=1.5)
    ax.axhline(0.85, color='gray', ls='--', lw=1.0, label='85% threshold')
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=0)
    ax.set_ylabel('DR@5s'); ax.set_ylim([0, 1.05])
    ax.legend(fontsize=8)
    ax.set_title('Overall DR@5s per Model (mean ± 95% CI)')
    _savefig(fig, 'per_model_dr5s.pdf')


# ── Figure: Precision vs MTBFA paradox scatter ───────────────────────────────
def fig_precision_mtbfa_paradox(all_metrics: dict):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for mn in MODEL_ORDER:
        if mn not in all_metrics:
            continue
        agg = aggregate_metrics(all_metrics[mn])
        prec  = _get(agg, 'precision')
        mtbfa = _get(agg, 'mtbfa_h')
        if math.isnan(prec) or math.isnan(mtbfa) or math.isinf(mtbfa):
            continue
        c = MODEL_COLORS[mn]
        ax.scatter(prec, mtbfa, color=c, zorder=5, s=80)
        ax.annotate(mn, (prec, mtbfa), textcoords='offset points',
                    xytext=(4, 3), fontsize=9, color=c)

    ax.axhline(0.10, color='gray', ls='--', lw=1.0, label='MTBFA threshold (0.1 h)')
    ax.axvline(0.85, color='gray', ls=':',  lw=1.0, label='Precision threshold (0.85)')
    ax.set_xlabel('Precision')
    ax.set_ylabel('MTBFA (h)')
    ax.set_title('Precision vs MTBFA: Non-Monotonic Relationship')
    ax.legend(fontsize=8)
    _savefig(fig, 'fig_b3_precision_mtbfa_paradox.pdf')


# ── Figure: FP windows vs FP events grouped bar chart ────────────────────────
def fig_fp_windows_vs_events(all_metrics: dict):
    models = [mn for mn in MODEL_ORDER if mn in all_metrics]
    x = np.arange(len(models))
    width = 0.35
    fp_windows_list, fp_events_list = [], []
    for mn in models:
        fw = [s.get('fp_windows', float('nan')) for s in all_metrics[mn]]
        fe = [s.get('fp_events',  float('nan')) for s in all_metrics[mn]]
        fp_windows_list.append(float(np.nanmean(fw)))
        fp_events_list.append(float(np.nanmean(fe)))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width/2, fp_windows_list, width,
           color=[MODEL_COLORS[mn] for mn in models], alpha=0.85, label='FP windows')
    ax.bar(x + width/2, fp_events_list, width,
           color=[MODEL_COLORS[mn] for mn in models], alpha=0.45,
           edgecolor='black', lw=0.8, label='FP events (aggregated)')
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=0)
    ax.set_ylabel('Count (mean over 5 seeds)')
    ax.set_title('FP Windows vs Aggregated FP Events per Model')
    ax.legend(fontsize=8)
    _savefig(fig, 'fig_b4_fp_windows_vs_events.pdf')


# ── Figure: FP event duration box plot ───────────────────────────────────────
def fig_fp_duration_boxplot(all_metrics: dict):
    models = [mn for mn in MODEL_ORDER if mn in all_metrics]
    durations = []
    for mn in models:
        dur_all = []
        for s in all_metrics[mn]:
            dur_all.extend(s.get('fp_event_durations_s', []))
        durations.append(dur_all if dur_all else [float('nan')])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bp = ax.boxplot(durations, labels=models, patch_artist=True,
                    flierprops=dict(marker='.', markersize=4, alpha=0.4))
    for patch, mn in zip(bp['boxes'], models):
        patch.set_facecolor(MODEL_COLORS[mn])
        patch.set_alpha(0.75)
    ax.set_ylabel('FP Event Duration (s)')
    ax.set_title('Distribution of False-Alarm Event Durations (all seeds pooled)')
    ax.grid(axis='y', alpha=0.3)
    _savefig(fig, 'fig_d2_fp_duration_boxplot.pdf')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    metrics_path = os.path.join(OUT_DIR, 'all_metrics.pkl')
    if not os.path.exists(metrics_path):
        print(f'ERROR: {metrics_path} not found. Run main pipeline first.')
        return

    with open(metrics_path, 'rb') as f:
        all_metrics = pickle.load(f)

    print(f'Loaded metrics for: {list(all_metrics.keys())}')
    print('Generating figures...')

    fig_fp_aggregation()
    fig_dr_curves(all_metrics)
    fig_pareto(all_metrics)
    fig_overall_bars(all_metrics)
    fig_time_metrics(all_metrics)
    fig_per_attack(all_metrics)
    fig_precision_mtbfa_paradox(all_metrics)
    fig_fp_windows_vs_events(all_metrics)
    fig_fp_duration_boxplot(all_metrics)

    print(f'\nAll figures saved to {FIG_DIR}')


if __name__ == '__main__':
    main()
