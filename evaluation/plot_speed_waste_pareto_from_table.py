#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path("results/apexp_block/config_large_500/evaluation/final_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Current paper-table numbers.
# x-axis: wasted token %
# y-axis: latency speedup vs AR
rows = [
    {"method": "AR", "group": "AR", "depth": "--", "speedup": 1.00, "waste_pct": 0.00},

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

# Pareto frontier: maximize speedup, minimize waste.
# A point is Pareto-efficient if no other point has both:
#   speedup >= this speedup
#   waste <= this waste
# with at least one strict improvement.
def is_pareto_efficient(frame):
    keep = []
    for i, r in frame.iterrows():
        dominated = False
        for j, q in frame.iterrows():
            if i == j:
                continue
            better_or_equal_speed = q["speedup"] >= r["speedup"]
            lower_or_equal_waste = q["waste_pct"] <= r["waste_pct"]
            strictly_better = (q["speedup"] > r["speedup"]) or (q["waste_pct"] < r["waste_pct"])
            if better_or_equal_speed and lower_or_equal_waste and strictly_better:
                dominated = True
                break
        keep.append(not dominated)
    return keep

df["pareto"] = is_pareto_efficient(df)

frontier = (
    df[df["pareto"]]
    .sort_values(["waste_pct", "speedup"], ascending=[True, True])
    .copy()
)

print("\n=== All points ===")
print(df.sort_values(["group", "method", "waste_pct"]).to_string(index=False))

print("\n=== Pareto frontier ===")
print(frontier[["method", "depth", "speedup", "waste_pct"]].to_string(index=False))

# Save CSV for paper/debug.
df.to_csv(OUT_DIR / "speed_waste_points.csv", index=False)
frontier.to_csv(OUT_DIR / "speed_waste_pareto_frontier.csv", index=False)

plt.figure(figsize=(7.2, 5.0))

markers = {
    "AR": "o",
    "Fixed": "s",
    "APEX": "^",
}

for group, sub in df.groupby("group"):
    plt.scatter(
        sub["waste_pct"],
        sub["speedup"],
        marker=markers.get(group, "o"),
        s=90,
        label=group,
        alpha=0.9,
    )

# Pareto line.
plt.plot(
    frontier["waste_pct"],
    frontier["speedup"],
    linewidth=2.0,
    linestyle="--",
    label="Pareto frontier",
)

# Annotate selected points.
for _, r in df.iterrows():
    label = r["method"]
    if r["depth"] not in {"--", "adaptive"}:
        label += f" {r['depth']}"
    plt.annotate(
        label,
        (r["waste_pct"], r["speedup"]),
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=8,
    )

plt.xlabel("Wasted draft tokens (%) ↓")
plt.ylabel("E2E speedup vs. AR ↑")
plt.title("Speed--Waste Tradeoff")
plt.grid(True, linestyle=":", linewidth=0.8)
plt.legend(frameon=True)
plt.tight_layout()

pdf_path = OUT_DIR / "speed_waste_pareto.pdf"
png_path = OUT_DIR / "speed_waste_pareto.png"

plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(png_path, dpi=300, bbox_inches="tight")

print(f"\nWrote:\n{pdf_path}\n{png_path}")
