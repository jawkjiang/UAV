"""Generate fig_window_step_dr5_heatmap.{pdf,png}

3x3 heatmap of DR@5s as a function of window size L and step size S
for the GRU detector (3 seeds per cell).

Reads:
  matlab_pipeline/output/ablation_window_step.json
    -- keys are L{w}_S{s}, each with 'window', 'step', 'seeds':[{dr_at_5s,...}]

Run:
  cd ~/projects/UAV
  python docs/figures/scripts/plot_window_step.py
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
JSON_PATH = os.path.join(ROOT, 'matlab_pipeline', 'output', 'ablation_window_step.json')
OUT_DIR = FIG_DIR

WINDOW_SIZES = [30, 50, 80]
STEP_SIZES   = [3, 5, 10]


def main():
    with open(JSON_PATH, 'r') as f:
        d = json.load(f)

    grid = np.full((len(WINDOW_SIZES), len(STEP_SIZES)), float('nan'))
    for i, w in enumerate(WINDOW_SIZES):
        for j, s in enumerate(STEP_SIZES):
            key = f'L{w}_S{s}'
            cell = d.get(key)
            if cell is None:
                continue
            seeds = cell.get('seeds', [])
            vals = [seed.get('dr_at_5s', float('nan')) for seed in seeds]
            vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
            if vals:
                grid[i, j] = np.mean(vals)

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 10,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(grid, cmap='YlGnBu', vmin=0.85, vmax=1.0, aspect='auto')

    ax.set_xticks(np.arange(len(STEP_SIZES)));
    ax.set_yticks(np.arange(len(WINDOW_SIZES)))
    ax.set_xticklabels([f'S={s}' for s in STEP_SIZES])
    ax.set_yticklabels([f'L={w}' for w in WINDOW_SIZES])
    ax.set_xlabel('Step size (samples)')
    ax.set_ylabel('Window size (samples)')
    ax.set_title('DR@5 s sensitivity to sliding-window parameters\n'
                 '(GRU detector, mean over 3 seeds)')

    for i in range(len(WINDOW_SIZES)):
        for j in range(len(STEP_SIZES)):
            v = grid[i, j]
            if np.isnan(v):
                txt = '--'
            else:
                txt = f'{v:.3f}'
            tc = 'white' if v > 0.95 else 'black'
            ax.text(j, i, txt, ha='center', va='center', color=tc, fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('DR@5 s')

    out_pdf = os.path.join(OUT_DIR, 'fig_window_step_dr5_heatmap.pdf')
    out_png = os.path.join(OUT_DIR, 'fig_window_step_dr5_heatmap.png')
    fig.savefig(out_pdf); fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f'Wrote {out_pdf}')
    print(f'Wrote {out_png}')


if __name__ == '__main__':
    main()
