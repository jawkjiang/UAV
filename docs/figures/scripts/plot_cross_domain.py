"""Generate fig_cross_domain_dr5_original.{pdf,png}

Bar chart comparing DR@5s for 7 architectures across:
  - In-distribution MATLAB sim test set
  - IEEE DataPort live (HackRF spoofing)
  - IEEE DataPort SITL (multi-platform)
  - ALFA (injected step)

Reads:
  matlab_pipeline/output/aggregated_metrics.json   (sim test)
  matlab_pipeline/output/cross_domain/cross_domain_summary.json (3 cross-distribution sets)

Run:
  cd ~/projects/UAV
  python docs/figures/scripts/plot_cross_domain.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))   # .../docs/figures/scripts
FIG_DIR = os.path.dirname(_HERE)                      # .../docs/figures
DOCS = os.path.dirname(FIG_DIR)                       # .../docs
ROOT = os.path.dirname(DOCS)                          # project root
SIM_PATH = os.path.join(ROOT, 'matlab_pipeline', 'output', 'aggregated_metrics.json')
XD_PATH  = os.path.join(ROOT, 'matlab_pipeline', 'output', 'cross_domain',
                        'cross_domain_summary.json')
OUT_DIR  = FIG_DIR

MODELS = ['CNN', 'LSTM', 'BiLSTM', 'GRU', 'CNN-LSTM', 'TCN', 'Transformer']
DOMAINS = [
    ('Sim (in-dist.)',          SIM_PATH, None),
    ('DataPort Live (HackRF)',  XD_PATH,  'IEEE_DataPort_Live (HackRF spoofing)'),
    ('DataPort SITL',           XD_PATH,  'IEEE_DataPort_SITL (multi-platform)'),
    ('ALFA (injected)',         XD_PATH,  'ALFA (injected)'),
]
COLORS = ['#4477AA', '#EE6677', '#228833', '#CCBB44']


def _get(path, dataset_key, model):
    with open(path, 'r') as f:
        d = json.load(f)
    if dataset_key is not None:
        d = d.get(dataset_key, {})
    if model not in d:
        return float('nan')
    v = d[model].get('dr_at_5s', {}).get('mean', float('nan'))
    return float(v)


def main():
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 10,
        'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 9,  'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })
    fig, ax = plt.subplots(figsize=(8, 4.2))
    n_models = len(MODELS)
    n_dom = len(DOMAINS)
    width = 0.8 / n_dom
    x = np.arange(n_models)

    for i, (label, path, key) in enumerate(DOMAINS):
        values = [_get(path, key, m) for m in MODELS]
        ax.bar(x + (i - (n_dom - 1) / 2) * width, values, width=width,
               color=COLORS[i], edgecolor='white', linewidth=0.6, label=label)

    ax.axhline(0.85, color='gray', ls='--', lw=1.0, label='DR@5s threshold')
    ax.set_xticks(x); ax.set_xticklabels(MODELS, rotation=15, ha='right')
    ax.set_ylim([0, 1.05])
    ax.set_ylabel('DR@5 s')
    ax.set_title('Cross-distribution generalization: DR@5 s across four data sources')
    ax.legend(loc='lower left', ncol=2, framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, lw=0.5)

    out_pdf = os.path.join(OUT_DIR, 'fig_cross_domain_dr5_original.pdf')
    out_png = os.path.join(OUT_DIR, 'fig_cross_domain_dr5_original.png')
    fig.savefig(out_pdf); fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f'Wrote {out_pdf}')
    print(f'Wrote {out_png}')


if __name__ == '__main__':
    main()
