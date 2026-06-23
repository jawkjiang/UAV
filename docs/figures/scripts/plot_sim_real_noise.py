"""Plot sim vs real GPS noise statistics summary for response letter.

Reads docs/tables/sim_real_noise_compare.json and emits a single figure
that contrasts sigma_xy, sigma_v, excess kurtosis, and p99 step magnitude.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC = os.path.join(ROOT, "docs", "tables", "sim_real_noise_compare.json")
DST_DIR = os.path.join(ROOT, "docs", "figures", "response")
os.makedirs(DST_DIR, exist_ok=True)

with open(SRC, "r", encoding="utf-8") as f:
    data = json.load(f)

domains = ["sim_matlab", "ieee_live", "alfa"]
labels = ["MATLAB sim", "IEEE Live", "ALFA"]
colors = ["#3b6fb6", "#d9534f", "#5cb85c"]

sigma_xy = [data[d]["sigma_xy_per_axis_m"] for d in domains]
sigma_v = [data[d]["sigma_v_per_axis_mps"] for d in domains]
ek = [data[d]["excess_kurtosis_innov"] for d in domains]
step_p99 = [data[d]["pos_step_p99_m_mean"] for d in domains]

fig, axes = plt.subplots(1, 4, figsize=(11, 3.0))

def bar(ax, vals, title, ylabel, log=False, y_zero_line=False):
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    if log:
        ax.set_yscale("log")
    if y_zero_line:
        ax.axhline(0.0, color="black", linewidth=0.5, linestyle="--", alpha=0.6)
    ax.tick_params(axis="x", labelsize=8, rotation=15)
    ax.tick_params(axis="y", labelsize=8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2.0,
                b.get_height(),
                f"{v:.2f}" if abs(v) < 100 else f"{v:.1f}",
                ha="center", va="bottom", fontsize=8)

bar(axes[0], sigma_xy, r"$\sigma_{xy}$ per axis", "metres", log=True)
bar(axes[1], sigma_v, r"$\sigma_{v}$ per axis", "m/s", log=True)
bar(axes[2], ek, "Excess kurtosis", "(0 = Gaussian)", y_zero_line=True)
bar(axes[3], step_p99, "p99 step jump", "metres", log=True)

fig.suptitle("Sim vs real GPS innovation statistics (normal-flight segments)", fontsize=11)
plt.tight_layout(rect=(0, 0, 1, 0.94))

for ext in ("pdf", "png"):
    plt.savefig(os.path.join(DST_DIR, f"sim_real_noise.{ext}"),
                dpi=200, bbox_inches="tight")
print("Saved", os.path.join(DST_DIR, "sim_real_noise.pdf"))
