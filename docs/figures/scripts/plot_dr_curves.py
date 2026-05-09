"""Generate fig_c1_dr_at_delta_t_curves.{pdf,png}

DR@Δt curves for all 7 architectures at Δt ∈ {1, 3, 5, 10, 15} s,
mean ± 95% CI band over 5 seeds.

Reads:  matlab_pipeline/output/aggregated_metrics.json
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.dirname(_HERE)
DOCS = os.path.dirname(FIG_DIR)
ROOT = os.path.dirname(DOCS)
JSON_PATH = os.path.join(ROOT, 'matlab_pipeline', 'output', 'aggregated_metrics.json')

MODELS = ['CNN', 'LSTM', 'BiLSTM', 'GRU', 'CNN-LSTM', 'TCN', 'Transformer']
COLORS = {'CNN': '#4477AA', 'LSTM': '#EE6677', 'BiLSTM': '#228833',
          'GRU': '#CCBB44', 'CNN-LSTM': '#66CCEE', 'TCN': '#AA3377',
          'Transformer': '#777777'}
TS = [1, 5, 10, 15]


def main():
    with open(JSON_PATH) as f:
        d = json.load(f)
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10,
                         'savefig.dpi': 300, 'savefig.bbox': 'tight'})
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for m in MODELS:
        if m not in d: continue
        means = [d[m][f'dr_at_{t}s']['mean'] for t in TS]
        lo    = [d[m][f'dr_at_{t}s']['ci_lo'] for t in TS]
        hi    = [d[m][f'dr_at_{t}s']['ci_hi'] for t in TS]
        c = COLORS[m]
        ax.plot(TS, means, 'o-', color=c, label=m, lw=1.6, markersize=5)
        ax.fill_between(TS, lo, hi, color=c, alpha=0.15)
    ax.axhline(0.85, color='gray', ls='--', lw=1.0, label='DR threshold (0.85)')
    ax.axvline(5, color='gray', ls=':',  lw=1.0, label='5 s budget')
    ax.set_xlabel(r'Detection time budget $\Delta t$ (s)')
    ax.set_ylabel(r'DR@$\Delta t$')
    ax.set_xticks(TS); ax.set_xlim([0, 16]); ax.set_ylim([0, 1.05])
    ax.legend(ncol=2, loc='lower right', framealpha=0.95, fontsize=8)
    ax.set_title(r'DR@$\Delta t$ curves (mean $\pm$ 95\% CI, 5 seeds)')
    ax.grid(alpha=0.3, lw=0.5)
    fig.savefig(os.path.join(FIG_DIR, 'fig_c1_dr_at_delta_t_curves.pdf'))
    fig.savefig(os.path.join(FIG_DIR, 'fig_c1_dr_at_delta_t_curves.png'), dpi=200)
    plt.close(fig)
    print('Wrote fig_c1_dr_at_delta_t_curves.{pdf,png}')


if __name__ == '__main__':
    main()
