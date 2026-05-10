"""Augmentation isolation flow figure for response letter (R1.1, R3.4)."""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DST_DIR = os.path.join(ROOT, "docs", "figures", "response")
os.makedirs(DST_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(8.4, 3.8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis("off")


def box(x, y, w, h, text, color):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                       linewidth=1.2, edgecolor="black", facecolor=color)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def arrow(x0, y0, x1, y1, color="black"):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                        mutation_scale=12, linewidth=1.0, color=color)
    ax.add_patch(a)


box(0.2, 3.6, 2.0, 0.9, "3000 MATLAB\nflights\n(flight ids 0..2999)", "#dde3ef")
box(0.2, 1.4, 2.0, 0.9, "Stratified split\nby flight id", "#fff2cc")
box(2.8, 3.0, 2.0, 0.9, "Train\n1800 flights", "#cfe2cf")
box(2.8, 1.7, 2.0, 0.9, "Val\n600 flights", "#dadada")
box(2.8, 0.4, 2.0, 0.9, "Test\n600 flights", "#dadada")
box(5.6, 3.0, 2.4, 0.9, "TimeGAN\n(train ids only)", "#f7d7d7")
box(8.2, 3.0, 1.6, 0.9, "Augmented\ntrain", "#cfe2cf")

# guard
ax.text(8.0, 1.05,
        r"assert ids.augmented $\cap$ (ids.val $\cup$ ids.test) == $\emptyset$",
        ha="center", va="center", fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fbeed5", edgecolor="black"))

# arrows
arrow(2.2, 4.05, 2.8, 3.45)
arrow(2.2, 3.85, 2.2, 2.3)
arrow(2.2, 1.85, 2.8, 2.15)
arrow(2.2, 1.85, 2.8, 0.85)
arrow(4.8, 3.45, 5.6, 3.45)
arrow(8.0, 3.45, 8.2, 3.45)

ax.text(5.0, 4.7, "MATLAB pipeline -> split -> augment train only",
        ha="center", fontsize=11, fontweight="bold")

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(os.path.join(DST_DIR, f"timegan_isolation.{ext}"),
                dpi=200, bbox_inches="tight")
print("saved", os.path.join(DST_DIR, "timegan_isolation.pdf"))
