"""Generate fig_b3_precision_mtbfa_paradox.{pdf,png}

Scatter of (precision, MTBFA) for 7 architectures, illustrating that
precision and MTBFA are not monotonically related: high precision does
not imply high MTBFA.

Reads:  matlab_pipeline/output/aggregated_metrics.json
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
JSON_PATH = os.path.join(ROOT, 'matlab_pipeline', 'output', 'aggregated_metrics.json')

MODELS = ['CNN', 'LSTM', 'BiLSTM', 'GRU', 'CNN-LSTM', 'TCN', 'Transformer']
COLORS = {'CNN': '#4477AA', 'LSTM': '#EE6677', 'BiLSTM': '#228833',
          'GRU': '#CCBB44', 'CNN-LSTM': '#66CCEE', 'TCN': '#AA3377',
          'Transformer': '#777777'}


def main():
    with open(JSON_PATH) as f:
        d = json.load(f)
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10,
                         'savefig.dpi': 300, 'savefig.bbox': 'tight'})
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for m in MODELS:
        if m not in d: continue
        p   = d[m]['precision']['mean']
        plo = d[m]['precision']['ci_lo']; phi = d[m]['precision']['ci_hi']
        mt  = d[m]['mtbfa_h']['mean']
        mlo = d[m]['mtbfa_h']['ci_lo']; mhi = d[m]['mtbfa_h']['ci_hi']
        if math.isnan(p) or math.isnan(mt) or math.isinf(mt): continue
        c = COLORS[m]
        ax.errorbar([p], [mt], xerr=[[p-plo],[phi-p]], yerr=[[mt-mlo],[mhi-mt]],
                    fmt='o', color=c, markersize=8, capsize=3, lw=1.0)
        ax.annotate(m, (p, mt), textcoords='offset points', xytext=(7, 4),
                    fontsize=9, color=c, fontweight='bold')
    ax.axvline(0.85, color='gray', ls=':',  lw=1.0, label='Precision threshold (0.85)')
    ax.axhline(0.10, color='gray', ls='--', lw=1.0, label='MTBFA threshold (0.10 h)')
    ax.set_xlabel('Precision')
    ax.set_ylabel('MTBFA (h)')
    ax.set_title('Precision vs.\\ MTBFA: non-monotonic relationship\n'
                 '(error bars show 95\\% CI over 5 seeds)')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.95)
    ax.grid(alpha=0.3, lw=0.5)
    fig.savefig(os.path.join(FIG_DIR, 'fig_b3_precision_mtbfa_paradox.pdf'))
    fig.savefig(os.path.join(FIG_DIR, 'fig_b3_precision_mtbfa_paradox.png'), dpi=200)
    plt.close(fig)
    print('Wrote fig_b3_precision_mtbfa_paradox.{pdf,png}')


if __name__ == '__main__':
    main()
