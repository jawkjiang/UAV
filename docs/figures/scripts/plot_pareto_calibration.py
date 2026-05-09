"""Generate fig_pareto_calibration_tradeoff.{pdf,png}

For each of 7 architectures, plot the (test MTBFA, test DR@5s) Pareto
trade-off curve obtained by sweeping the decision threshold p.
Highlight the operational target line MTBFA = 0.10 h, and mark each
architecture's calibrated operating point p* (smallest p such that
val MTBFA >= 0.10 h).

Reads:  matlab_pipeline/output/calibrated_threshold/pareto_sweep.json
"""
import json, os, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.dirname(_HERE)
DOCS = os.path.dirname(FIG_DIR)
ROOT = os.path.dirname(DOCS)
JSON_PATH = os.path.join(ROOT, 'matlab_pipeline', 'output',
                         'calibrated_threshold', 'pareto_sweep.json')

MODELS = ['CNN', 'LSTM', 'BiLSTM', 'GRU', 'CNN-LSTM', 'TCN', 'Transformer']
COLORS = {'CNN': '#4477AA', 'LSTM': '#EE6677', 'BiLSTM': '#228833',
          'GRU': '#CCBB44', 'CNN-LSTM': '#66CCEE', 'TCN': '#AA3377',
          'Transformer': '#777777'}
MTBFA_TARGET = 0.10  # h


def _safe(v):
    if v == 'inf':  return float('inf')
    if v == 'nan':  return float('nan')
    if v == '-inf': return float('-inf')
    return float(v)


def main():
    with open(JSON_PATH) as f:
        d = json.load(f)
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10,
                         'savefig.dpi': 300, 'savefig.bbox': 'tight'})
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    for m in MODELS:
        if m not in d: continue
        curve = d[m]
        # sort by mtbfa for monotone curve
        pts = []
        p_star_idx = None
        # Find smallest p such that val_mtbfa >= 0.10h (chosen on val, not test)
        for i, c in enumerate(curve):
            v_mt = _safe(c['val_mtbfa_mean'])
            if (math.isinf(v_mt) or v_mt >= MTBFA_TARGET) and p_star_idx is None:
                p_star_idx = i
            t_mt = _safe(c['test_mtbfa_mean'])
            t_dr = _safe(c['test_dr5_mean'])
            if math.isnan(t_mt) or math.isnan(t_dr): continue
            if math.isinf(t_mt): continue   # skip infinite (zero-FP) points for plot scale
            pts.append((t_mt, t_dr, c['p']))
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        c = COLORS[m]
        ax.plot(xs, ys, '-', color=c, lw=1.6, alpha=0.85, label=m)
        # mark p* operating point
        if p_star_idx is not None:
            cs = curve[p_star_idx]
            ts_mt = _safe(cs['test_mtbfa_mean']); ts_dr = _safe(cs['test_dr5_mean'])
            if not (math.isinf(ts_mt) or math.isnan(ts_mt) or math.isnan(ts_dr)):
                ax.plot([ts_mt], [ts_dr], 'o', color=c, markersize=10,
                        markeredgecolor='black', markeredgewidth=1.2, zorder=5)
                ax.annotate(f"$p^\\star={cs['p']:.2f}$", (ts_mt, ts_dr),
                            xytext=(8, -10), textcoords='offset points',
                            fontsize=8, color=c)

    ax.axvline(MTBFA_TARGET, color='gray', ls='--', lw=1.0, label='MTBFA target (0.10 h)')
    ax.axhline(0.85, color='gray', ls=':',  lw=1.0, label='DR@5 s threshold (0.85)')
    ax.set_xscale('log')
    ax.set_xlim([0.005, 1.0])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('MTBFA on test set (h, log scale)')
    ax.set_ylabel('DR@5 s on test set')
    ax.set_title('Calibration trade-off: per-architecture Pareto curve\n'
                 r'(sweep $p \in [0.50,0.99]$; markers = $p^\star$ chosen on validation)')
    ax.legend(loc='lower left', ncol=2, fontsize=8, framealpha=0.95)
    ax.grid(alpha=0.3, lw=0.5)
    fig.savefig(os.path.join(FIG_DIR, 'fig_pareto_calibration_tradeoff.pdf'))
    fig.savefig(os.path.join(FIG_DIR, 'fig_pareto_calibration_tradeoff.png'), dpi=200)
    plt.close(fig)
    print('Wrote fig_pareto_calibration_tradeoff.{pdf,png}')


if __name__ == '__main__':
    main()
