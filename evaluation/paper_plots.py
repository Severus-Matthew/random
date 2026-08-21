# #!/usr/bin/env python3
# """Create a single 3D plot of entropy, repetition density, and speedup.

# Each point is the median of one workload/source-dataset/method group from the
# A_baseline_sweep experiment at k=4. Hardware is excluded, and Conv-Gen and
# Conv-SFT are represented by one conversational point whose coordinates are the
# arithmetic means of the two grouped values. Points from the same method are
# connected in ascending mean-entropy order. The z=1 plane marks parity with
# standard autoregressive decoding.
# """

# from __future__ import annotations

# import argparse
# from pathlib import Path

# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from matplotlib.lines import Line2D


# EXPERIMENT = "A_baseline_sweep"
# K = 4

# METHODS = {
#     "ngram_sd": {"label": "N-gram", "color": "#0072B2", "marker": "o"},
#     "eagle3": {"label": "EAGLE-3", "color": "#D55E00", "marker": "s"},
#     "draft_sd": {"label": "Draft model", "color": "#009E73", "marker": "^"},
# }

# CONVERSATIONAL_WORKLOADS = {
#     "conversational_generation_gen",
#     "conversational_generation_sft",
# }


# def load_grouped_data(all_runs_path: Path, aggregate_path: Path) -> pd.DataFrame:
#     runs = pd.read_csv(all_runs_path)
#     aggregate = pd.read_csv(aggregate_path)

#     refs = aggregate.loc[
#         aggregate["experiment"].eq(EXPERIMENT)
#         & aggregate["tps_mean"].notna()
#         & aggregate["speedup_vs_ar"].gt(0),
#         ["workload", "source_dataset", "tps_mean", "speedup_vs_ar"],
#     ].copy()
#     refs["ar_reference_tps"] = refs["tps_mean"] / refs["speedup_vs_ar"]
#     refs = (
#         refs.groupby(["workload", "source_dataset"], as_index=False)[
#             "ar_reference_tps"
#         ]
#         .median()
#     )

#     data = runs.loc[
#         runs["experiment"].eq(EXPERIMENT)
#         & runs["method"].isin(METHODS)
#         & runs["k"].eq(K),
#         [
#             "workload",
#             "source_dataset",
#             "method",
#             "tokens_per_sec",
#             "mean_entropy",
#             "repetition_density",
#         ],
#     ].copy()
#     data = data.merge(
#         refs,
#         on=["workload", "source_dataset"],
#         how="left",
#         validate="many_to_one",
#     )
#     if data["ar_reference_tps"].isna().any():
#         raise ValueError("Some runs do not have a matching AR reference")

#     data["speedup_vs_ar"] = data["tokens_per_sec"] / data["ar_reference_tps"]
#     data = data.replace([np.inf, -np.inf], np.nan).dropna(
#         subset=["speedup_vs_ar", "mean_entropy", "repetition_density"]
#     )

#     grouped = (
#         data.groupby(["workload", "source_dataset", "method"], as_index=False)
#         .agg(
#             mean_entropy=("mean_entropy", "median"),
#             repetition_density=("repetition_density", "median"),
#             speedup_vs_ar=("speedup_vs_ar", "median"),
#             n=("speedup_vs_ar", "size"),
#         )
#     )

#     # Remove the hardware workload and replace Conv-Gen/Conv-SFT with one
#     # conversational point per method. The requested average is taken after
#     # each workload has first been summarized, so both conversational
#     # workloads receive equal weight.
#     grouped = grouped.loc[~grouped["workload"].eq("hardware_gen")].copy()
#     conversational = grouped.loc[
#         grouped["workload"].isin(CONVERSATIONAL_WORKLOADS)
#     ].copy()
#     counts = conversational.groupby("method").size()
#     if not counts.reindex(METHODS, fill_value=0).eq(2).all():
#         raise ValueError(
#             "Expected exactly one Conv-Gen and one Conv-SFT point per method"
#         )

#     conversational = (
#         conversational.groupby("method", as_index=False)
#         .agg(
#             mean_entropy=("mean_entropy", "mean"),
#             repetition_density=("repetition_density", "mean"),
#             speedup_vs_ar=("speedup_vs_ar", "mean"),
#             n=("n", "sum"),
#         )
#         .assign(
#             workload="conversational_generation",
#             source_dataset="Conv-Gen + Conv-SFT",
#         )
#     )
#     grouped = pd.concat(
#         [
#             grouped.loc[~grouped["workload"].isin(CONVERSATIONAL_WORKLOADS)],
#             conversational,
#         ],
#         ignore_index=True,
#     )

#     expected = 7 * len(METHODS)
#     if len(grouped) != expected:
#         raise ValueError(f"Expected {expected} grouped points, found {len(grouped)}")
#     return grouped


# def short_workload(row: pd.Series) -> str:
#     workload = row["workload"]
#     if workload == "code_gen":
#         return "HumanEval" if "humaneval" in row["source_dataset"].lower() else "MBPP"
#     return {
#         "conversational_generation": "Conversational",
#         "conversational_generation_gen": "Conv-Gen",
#         "conversational_generation_sft": "Conv-SFT",
#         "hardware_gen": "Hardware",
#         "long_chain_reasoning": "Long-chain",
#         "long_context_completion": "Long-context",
#         "long_horizon_swe": "SWE",
#         "mathematical_reasoning": "Math",
#     }.get(workload, workload)


# def configure_style() -> None:
#     mpl.rcParams.update(
#         {
#             "font.family": "serif",
#             "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
#             "mathtext.fontset": "stix",
#             "font.size": 8.5,
#             "axes.titlesize": 9,
#             "axes.labelsize": 8.5,
#             "xtick.labelsize": 7.5,
#             "ytick.labelsize": 7.5,
#             "legend.fontsize": 8,
#             "pdf.fonttype": 42,
#             "ps.fonttype": 42,
#             "savefig.bbox": None,
#             "savefig.pad_inches": 0.03,
#         }
#     )


# def make_figure(grouped: pd.DataFrame, output_stem: Path) -> None:
#     configure_style()
#     fig = plt.figure(figsize=(7.1, 4.5))
#     ax = fig.add_subplot(111, projection="3d")
#     ax.set_position([0.08, 0.04, 0.82, 0.88])

#     x_min, x_max = 0.05, 0.43
#     y_min, y_max = 0.24, 0.62
#     xx, yy = np.meshgrid(
#         np.linspace(x_min, x_max, 2),
#         np.linspace(y_min, y_max, 2),
#     )
#     ax.plot_surface(
#         xx,
#         yy,
#         np.ones_like(xx),
#         color="#777777",
#         alpha=0.10,
#         shade=False,
#         linewidth=0,
#         zorder=0,
#     )

#     for method, style in METHODS.items():
#         # The connecting path is ordered left-to-right by entropy. Without an
#         # explicit order, connecting points in their CSV order would imply an
#         # arbitrary trajectory.
#         subset = (
#             grouped.loc[grouped["method"].eq(method)]
#             .sort_values("mean_entropy")
#             .copy()
#         )
#         x = subset["mean_entropy"].to_numpy()
#         y = subset["repetition_density"].to_numpy()
#         z = subset["speedup_vs_ar"].to_numpy()

#         # Vertical stems make depth and the relation to the 1x plane readable.
#         for xi, yi, zi in zip(x, y, z):
#             ax.plot(
#                 [xi, xi],
#                 [yi, yi],
#                 [1.0, zi],
#                 color=style["color"],
#                 alpha=0.14,
#                 linewidth=0.65,
#                 zorder=1,
#             )

#         ax.plot(
#             x,
#             y,
#             z,
#             color=style["color"],
#             linewidth=1.45,
#             alpha=0.82,
#             zorder=2,
#         )

#         ax.scatter(
#             x,
#             y,
#             z,
#             s=42,
#             marker=style["marker"],
#             color=style["color"],
#             edgecolor="white",
#             linewidth=0.55,
#             depthshade=False,
#             label=style["label"],
#             zorder=3,
#         )

#     ax.set_title("Entropy, repetition, and speculative-decoding speedup", pad=8)
#     ax.set_xlabel("Mean token entropy", labelpad=8)
#     ax.set_ylabel("Repetition density", labelpad=8)
#     ax.set_zlabel("Median speedup vs. AR", labelpad=9)
#     ax.set_xlim(x_min, x_max)
#     ax.set_ylim(y_min, y_max)
#     ax.set_zlim(0.4, 2.8)
#     ax.set_xticks([0.1, 0.2, 0.3, 0.4])
#     ax.set_yticks([0.3, 0.4, 0.5, 0.6])
#     ax.set_zticks([0.5, 1.0, 1.5, 2.0, 2.5])
#     ax.view_init(elev=17, azim=-68)
#     ax.set_box_aspect((1.35, 1.0, 0.85))

#     # Quiet pane and grid styling keeps the data dominant.
#     for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
#         axis.pane.fill = False
#         axis.pane.set_edgecolor("#CCCCCC")
#         axis._axinfo["grid"]["color"] = (0.82, 0.82, 0.82, 0.65)
#         axis._axinfo["grid"]["linewidth"] = 0.55

#     handles = [
#         Line2D(
#             [0],
#             [0],
#             marker=style["marker"],
#             linestyle="none",
#             markerfacecolor=style["color"],
#             markeredgecolor="white",
#             markeredgewidth=0.5,
#             markersize=6.5,
#             label=style["label"],
#         )
#         for style in METHODS.values()
#     ]
#     handles.append(
#         Line2D([0], [0], color="#777777", linewidth=5, alpha=0.18, label="AR parity")
#     )
#     ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.96), frameon=False)

#     output_stem.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(output_stem.with_suffix(".pdf"))
#     fig.savefig(output_stem.with_suffix(".png"), dpi=300)
#     plt.close(fig)


# def main() -> None:
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--all-runs", type=Path, default=Path("/fsx/jmanvi/Internship_project/ASD/results/all_runs.csv"))
#     parser.add_argument(
#         "--aggregate",
#         type=Path,
#         default=Path("/fsx/jmanvi/Internship_project/ASD/results/aggregate_by_workload_method.csv"),
#     )
#     parser.add_argument(
#         "--output-stem",
#         type=Path,
#         default=Path("/fsx/jmanvi/Internship_project/ASD/evaluation/entropy_repetition_speedup_3d"),
#         help="Output path without extension; writes both PDF and PNG.",
#     )
#     args = parser.parse_args()
#     grouped = load_grouped_data(args.all_runs, args.aggregate)
#     make_figure(grouped, args.output_stem)


# if __name__ == "__main__":
#     main()


import random

# --- Configuration ---
file1 = "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/conversational_generation_test_gen.jsonl"  # Replace with your first JSONL file path
file2 = "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/conversational_generation_test_sft.jsonl"  # Replace with your second JSONL file path
output_file = "/fsx/jmanvi/Internship_project/ASD/data/benchmarks/By_split_phase_1/conversational_generation_test.jsonl"  # Output file name

# --- Read lines from both files ---
with open(file1, "r", encoding="utf-8") as f:
    lines1 = f.readlines()

with open(file2, "r", encoding="utf-8") as f:
    lines2 = f.readlines()

# --- Sample 250 random lines from each ---
sample1 = random.sample(lines1, 250)
sample2 = random.sample(lines2, 250)

# --- Combine and write to output ---
combined = sample1 + sample2
random.shuffle(combined)  # Optional: shuffle the combined lines

with open(output_file, "w", encoding="utf-8") as f:
    f.writelines(combined)

print(f"Done! Wrote 500 lines to '{output_file}'")

