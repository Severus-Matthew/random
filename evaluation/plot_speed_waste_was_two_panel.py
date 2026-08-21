#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("results/apexp_block/config_large_500/evaluation/final_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = [
    {"method": "EAGLE-3", "group": "Fixed", "depth": "k=8",  "speedup": 1.72, "waste_pct": 99.01},
    {"method": "EAGLE-3", "group": "Fixed", "depth": "k=16", "speedup": 1.14, "waste_pct": 99.50},

    {"method": "n-gram SD", "group": "Fixed", "depth": "k=8",  "speedup": 4.05, "waste_pct": 79.52},
    {"method": "n-gram SD", "group": "Fixed", "depth": "k=16", "speedup": 4.41, "waste_pct": 87.97},

    {"method": "Draft-SD", "group": "Fixed", "depth": "k=8",  "speedup": 0.67, "waste_pct": 50.49},
    {"method": "Draft-SD", "group": "Fixed", "depth": "k=16", "speedup": 0.45, "waste_pct": 68.21},

    {"method": "APEX-Speed", "group": "APEX", "depth": "adaptive", "speedup": 3.34, "waste_pct": 64.07},
    {"method": "APEX-Efficient", "group": "APEX", "depth": "adaptive", "speedup": 1.56, "waste_pct": 59.88},
    {"method": "APEX-Balanced", "group": "APEX", "depth": "adaptive", "speedup": 3.25, "waste_pct": 59.46},
]

df = pd.DataFrame(rows)
df["useful_pct"] = 100.0 - df["waste_pct"]
df["waste_frac"] = df["waste_pct"] / 100.0
df["WAS"] = df["speedup"] / df["waste_frac"]

df["label"] = df["method"]
df.loc[df["depth"] != "adaptive", "label"] = (
    df.loc[df["depth"] != "adaptive", "method"] + " " + df.loc[df["depth"] != "adaptive", "depth"]
)

df.to_csv(OUT_DIR / "speed_waste_was_points.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.15, 1.0]})

ax = axes[0]

fixed = df[df["group"] == "Fixed"]
apex = df[df["group"] == "APEX"]

ax.scatter(
    fixed["useful_pct"],
    fixed["speedup"],
    marker="s",
    s=80,
    alpha=0.75,
    label="Fixed baselines",
)

ax.scatter(
    apex["useful_pct"],
    apex["speedup"],
    marker="^",
    s=150,
    alpha=0.95,
    label="APEX",
)

# Highlight the APEX region.
xmin = max(0, apex["useful_pct"].min() - 3)
xmax = min(100, apex["useful_pct"].max() + 3)
ymin = max(0, apex["speedup"].min() - 0.25)
ymax = apex["speedup"].max() + 0.35
ax.fill_between([xmin, xmax], ymin, ymax, alpha=0.08)

# Draw arrows from closest aggressive fixed baselines toward APEX operating points.
def point(name):
    r = df[df["label"].eq(name)]
    if len(r) == 0:
        return None
    return float(r["useful_pct"].iloc[0]), float(r["speedup"].iloc[0])

arrow_pairs = [
    ("n-gram SD k=8", "APEX-Balanced"),
    ("n-gram SD k=16", "APEX-Speed"),
]

for a, b in arrow_pairs:
    pa = point(a)
    pb = point(b)
    if pa and pb:
        ax.annotate(
            "",
            xy=pb,
            xytext=pa,
            arrowprops=dict(arrowstyle="->", lw=1.6, alpha=0.65),
        )

# Annotate only important points to avoid clutter.
important = {
    "n-gram SD k=8": (-50, -14),
    "n-gram SD k=16": (-60, 8),
    "Draft-SD k=8": (-80, 10),
    "APEX-Speed": (8, 8),
    "APEX-Balanced": (8, -16),
    "APEX-Efficient": (8, -12),
}

for _, r in df.iterrows():
    if r["label"] not in important:
        continue
    dx, dy = important[r["label"]]
    ax.annotate(
        r["label"],
        (r["useful_pct"], r["speedup"]),
        textcoords="offset points",
        xytext=(dx, dy),
        fontsize=8.5,
    )

ax.set_xlabel("Useful draft tokens (%) ↑")
ax.set_ylabel("E2E speedup vs. AR ↑")
ax.set_title("(a) Speed--useful-token tradeoff")
ax.grid(True, linestyle=":", linewidth=0.8)
ax.legend(frameon=True, loc="lower left")

# Panel B: WAS ranking.
ax = axes[1]

bar_df = df.copy()
bar_df = bar_df.sort_values("WAS", ascending=True)

labels = bar_df["label"].tolist()
y = np.arange(len(bar_df))

ax.barh(y, bar_df["WAS"], alpha=0.85)

# Make APEX labels bold in annotation by adding text markers.
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8.5)

for i, (_, r) in enumerate(bar_df.iterrows()):
    ax.text(
        r["WAS"] + 0.06,
        i,
        f"{r['WAS']:.2f}",
        va="center",
        fontsize=8.5,
    )

# Add a visual divider showing strongest fixed baseline.
best_fixed = df[df["group"] == "Fixed"]["WAS"].max()
ax.axvline(best_fixed, linestyle="--", linewidth=1.2, alpha=0.75)
ax.text(
    best_fixed + 0.05,
    len(bar_df) - 0.7,
    "best fixed",
    fontsize=8,
    rotation=90,
    va="top",
)

ax.set_xlabel("Waste-Amortized Speedup (WAS) ↑")
ax.set_title("(b) Scalar efficiency score")
ax.grid(True, axis="x", linestyle=":", linewidth=0.8)

plt.tight_layout()

pdf_path = OUT_DIR / "speed_waste_was_two_panel.pdf"
png_path = OUT_DIR / "speed_waste_was_two_panel.png"

plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(png_path, dpi=300, bbox_inches="tight")

print("\n=== Points ===")
print(df[["label", "group", "speedup", "waste_pct", "useful_pct", "WAS"]].sort_values("WAS", ascending=False).to_string(index=False))

print(f"\nWrote:\n{pdf_path}\n{png_path}")
