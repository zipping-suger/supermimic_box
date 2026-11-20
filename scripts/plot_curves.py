"""Validation success against iterations and training time for the release runs -> docs/release_curves.png"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
runs = [("release_scratch", "From scratch", "#2a78d6", -8), ("release_sonic", "SONIC + residual", "#eb6834", 8)]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK})
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, facecolor=SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)
for name, label, color, dy in runs:
    path = root / "results" / name / "report.json"
    if not path.is_file():
        continue
    report = json.loads(path.read_text())
    curve = report["curve"]
    success = [100 * c["success"] / c["clips"] for c in curve]
    for ax, x in ((axes[0], [c["iteration"] / 1000 for c in curve]), (axes[1], [c["minutes"] for c in curve])):
        ax.plot(x, success, color=color, linewidth=2, marker="o", markersize=5, label=label,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(f"{success[-1]:.1f} %", (x[-1], success[-1]), xytext=(6, dy), textcoords="offset points",
                    va="center", color=INK, fontsize=9)
axes[0].set_xlabel("Training iterations (thousands)")
axes[1].set_xlabel("Training time (minutes, RTX 4080, after compilation)")
axes[0].set_ylabel("Validation success (%, 145 held-out clips)")
axes[0].set_ylim(60, 100)
axes[0].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
axes[0].set_xlim(0, 22)
axes[1].set_xlim(left=0)
axes[1].set_xlim(right=axes[1].get_xlim()[1] * 1.08)
axes[0].legend(frameon=False, loc="lower right")
fig.tight_layout()
out = root / "docs" / "release_curves.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(out)
