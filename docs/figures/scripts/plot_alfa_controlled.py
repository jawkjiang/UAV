"""Generate fig_alfa_controlled_dr5.{pdf,png}

Bar chart of the controlled ALFA cross-distribution comparison.
For each architecture, shows DR@5s under three training regimes:
  - 5 m/s, 3000 flights (original sim)
  - 5 m/s, 600 flights (data-size control)
  - 20 m/s matched, 600 flights (platform-speed match)

Reads:
  matlab_pipeline/output/transfer_analysis/transfer_summary.json

Run:
  cd ~/projects/UAV
  python docs/figures/scripts/plot_alfa_controlled.py
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
JSON_PATH = os.path.join(ROOT, 'matlab_pipeline', 'output', 'transfer_analysis',
                         'transfer_summary.json')
OUT_DIR = FIG_DIR

MODELS = ['CNN', 'LSTM', 'BiLSTM', 'GRU', 'CNN-LSTM', 'TCN', 'Transformer']
REGIMES = [
    ('5 m/s, 3000 flights',      '5ms_3000',       '#4477AA'),
    ('5 m/s, 600 flights',        '5ms_600',        '#EE6677'),
    ('20 m/s matched, 600 flights','20ms_matched_600','#AA3377'),
]


def main():
    with open(JSON_PATH, 'r') as f:
        d = json.load(f)
    matrix = d['alfa_matrix']

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 10,
        'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 9, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })
    fig, ax = plt.subplots(figsize=(8, 4.2))

    n_models = len(MODELS)
    n_reg = len(REGIMES)
    width = 0.8 / n_reg
    x = np.arange(n_models)

    for i, (label, key, color) in enumerate(REGIMES):
        vals = [matrix.get(m, {}).get(key, float('nan')) for m in MODELS]
        ax.bar(x + (i - (n_reg - 1) / 2) * width, vals, width=width,
               color=color, edgecolor='white', linewidth=0.6, label=label)

    ax.axhline(0.85, color='gray', ls='--', lw=1.0, label='DR@5s threshold')
    ax.set_xticks(x); ax.set_xticklabels(MODELS, rotation=15, ha='right')
    ax.set_ylim([0, 1.05])
    ax.set_ylabel('DR@5 s on ALFA (zero-shot)')
    ax.set_title('Controlled ALFA cross-distribution comparison: '
                 'matched-platform speed worsens transfer')
    ax.legend(loc='upper right', ncol=1, framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, lw=0.5)

    out_pdf = os.path.join(OUT_DIR, 'fig_alfa_controlled_dr5.pdf')
    out_png = os.path.join(OUT_DIR, 'fig_alfa_controlled_dr5.png')
    fig.savefig(out_pdf); fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f'Wrote {out_pdf}')
    print(f'Wrote {out_png}')


if __name__ == '__main__':
    main()
