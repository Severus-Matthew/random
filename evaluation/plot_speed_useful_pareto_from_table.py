#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path("results/apexp_block/config_large_500/evaluation/final_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Current paper-table numbers.
# x-axis: useful draft tokens = 100 - waste %
# y-axis: latency speedup vs AR
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

# WAS from your current definition:
# WAS = speedup / waste_fraction
df["was"] = df["speedup"] / (df["waste_pct"] / 100.0)

# Pareto frontier: maximize speedup and maximize useful_pct.
def is_pareto_efficient(frame):
    keep = []
    for i, r in frame.iterrows():
        dominated = False
        for j, q in frame.iterrows():
            if i == j:
                continue

            better_or_equal_speed = q["speedup"] >= r["speedup"]
            better_or_equal_useful = q["useful_pct"] >= r["useful_pct"]
            strictly_better = (q["speedup"] > r["speedup"]) or (q["useful_pct"] > r["useful_pct"])

            if better_or_equal_speed and better_or_equal_useful and strictly_better:
                dominated = True
                break

        keep.append(not dominated)

    return keep

df["pareto"] = is_pareto_efficient(df)

frontier = (
    df[df["pareto"]]
    .sort_values(["useful_pct", "speedup"], ascending=[True, True])
    .copy()
)

print("\n=== All points ===")
print(df.sort_values(["group", "method", "useful_pct"]).to_string(index=False))

print("\n=== Pareto frontier ===")
print(frontier[["method", "depth", "speedup", "waste_pct", "useful_pct", "was"]].to_string(index=False))

df.to_csv(OUT_DIR / "speed_useful_points.csv", index=False)
frontier.to_csv(OUT_DIR / "speed_useful_pareto_frontier.csv", index=False)

plt.figure(figsize=(7.3, 5.1))

markers = {
    "Fixed": "s",
    "APEX": "^",
}

for group, sub in df.groupby("group"):
    plt.scatter(
        sub["useful_pct"],
        sub["speedup"],
        marker=markers.get(group, "o"),
        s=110 if group == "APEX" else 80,
        label=group,
        alpha=0.9,
    )

# Pareto frontier line.
plt.plot(
    frontier["useful_pct"],
    frontier["speedup"],
    linewidth=2.0,
    linestyle="--",
    label="Pareto frontier",
)

# Annotate points.
for _, r in df.iterrows():
    label = r["method"]
    if r["depth"] != "adaptive":
        label += f" {r['depth']}"

    # small manual offsets to reduce overlap
    dx, dy = 5, 5
    if "APEX-Balanced" in label:
        dx, dy = 6, -12
    if "APEX-Speed" in label:
        dx, dy = 6, 8
    if "APEX-Efficient" in label:
        dx, dy = 6, -10
    if "n-gram SD k=16" in label:
        dx, dy = -70, 8
    if "n-gram SD k=8" in label:
        dx, dy = -65, -12

    plt.annotate(
        label,
        (r["useful_pct"], r["speedup"]),
        textcoords="offset points",
        xytext=(dx, dy),
        fontsize=8,
    )

plt.xlabel("Useful draft tokens (%) ↑")
plt.ylabel("E2E speedup vs. AR ↑")
plt.title("Speed--Useful-Speculation Tradeoff")
plt.grid(True, linestyle=":", linewidth=0.8)
plt.legend(frameon=True)
plt.tight_layout()

pdf_path = OUT_DIR / "speed_useful_pareto.pdf"
png_path = OUT_DIR / "speed_useful_pareto.png"

plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(png_path, dpi=300, bbox_inches="tight")

print(f"\nWrote:\n{pdf_path}\n{png_path}")
