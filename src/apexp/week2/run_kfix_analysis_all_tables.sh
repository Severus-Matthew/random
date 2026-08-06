#!/bin/bash
set -euo pipefail

cd /fsx/jmanvi/Internship_project/ASD

export LARGE_ROOT="results/apexp_block/config_large_500"
export BASELINE_ROOT="$LARGE_ROOT/week2/baseline_matrix_cap200_all_methods_all_k"
export KFIX_REPORT="$LARGE_ROOT/week2/final_hot_ablation_report_KFIX"

export HOT1="$LARGE_ROOT/week2/realtime_hot_e2e_cap50_w2_allk_1to16_KFIX_tracefix"
export HOT2="$LARGE_ROOT/week2/realtime_hot_e2e_cap50_w1_allk_1to16_KFIX_tracefix"
export HOT3="$LARGE_ROOT/week2/realtime_hot_e2e_cap50_w1_ks_1_2_3_4_5_6_8_12_16_KFIX_tracefix"
export HOT4="$LARGE_ROOT/week2/realtime_hot_e2e_cap50_w2_ks_1_2_3_4_5_6_8_12_16_KFIX_tracefix"
export HOT5="$LARGE_ROOT/week2/realtime_hot_e2e_cap50_w2_mink2_ks_2_3_4_5_6_8_12_16_KFIX_tracefix"

echo "=== Checking inputs ==="
echo "BASELINE_ROOT=$BASELINE_ROOT"
test -d "$BASELINE_ROOT"

for d in "$HOT1" "$HOT2" "$HOT3" "$HOT4" "$HOT5"; do
  echo
  echo "$d"
  test -d "$d"
  echo "results:" $(find "$d/results" -name "*.json" 2>/dev/null | wc -l)
done

rm -rf "$KFIX_REPORT"
mkdir -p "$KFIX_REPORT"

echo
echo "=== Running accepted_mask collector ==="
python3 src/apexp/week2/collect_hot_ablation_report_mask.py \
  --baseline-root "$BASELINE_ROOT" \
  --out-dir "$KFIX_REPORT" \
  --hot w2_allk_1to16_KFIX "$HOT1" \
  --hot w1_allk_1to16_KFIX "$HOT2" \
  --hot w1_sparse_KFIX "$HOT3" \
  --hot w2_sparse_KFIX "$HOT4" \
  --hot w2_mink2_sparse_KFIX "$HOT5" \
  2>&1 | tee "$KFIX_REPORT/collect_mask.log"

echo
echo "=== Creating clean normal overall table ==="
python3 - <<'PY'
import os
import pandas as pd
from pathlib import Path

root = Path(os.environ["KFIX_REPORT"])
df = pd.read_csv(root / "summary_overall_all_ablation_and_baselines_MASK.csv")

cols = [
    "system",
    "n_requests",
    "mean_tps_speedup_vs_ar",
    "latency_speedup_vs_ar",
    "wasted_token_pct",
    "accepted_per_block",
    "n_blocks_total",
    "mask_events_total",
    "missing_mask_events_total",
    "total_runtime_s",
    "avg_runtime_s",
    "mean_tps",
    "draft_tokens_total",
    "rejected_tokens_total",
]

out = df[cols].copy()
out = out.sort_values("mean_tps_speedup_vs_ar", ascending=False)

for c in [
    "mean_tps_speedup_vs_ar",
    "latency_speedup_vs_ar",
    "wasted_token_pct",
    "accepted_per_block",
    "total_runtime_s",
    "avg_runtime_s",
    "mean_tps",
]:
    out[c] = pd.to_numeric(out[c], errors="coerce").round(3)

for c in [
    "draft_tokens_total",
    "rejected_tokens_total",
    "n_blocks_total",
    "mask_events_total",
    "missing_mask_events_total",
]:
    out[c] = pd.to_numeric(out[c], errors="coerce").round(0).astype("Int64")

out.to_csv(root / "NORMAL_TABLE_all_baselines_all_apex_configs_KFIX.csv", index=False)
(root / "NORMAL_TABLE_all_baselines_all_apex_configs_KFIX.md").write_text(out.to_markdown(index=False))

print(out.to_string(index=False))
PY

echo
echo "=== Creating clean by-workload table ==="
python3 - <<'PY'
import os
import pandas as pd
from pathlib import Path

root = Path(os.environ["KFIX_REPORT"])
df = pd.read_csv(root / "summary_by_workload_all_ablation_and_baselines_MASK.csv")

cols = [
    "workload",
    "system",
    "n_requests",
    "mean_tps_speedup_vs_ar",
    "latency_speedup_vs_ar",
    "wasted_token_pct",
    "accepted_per_block",
    "n_blocks_total",
    "mask_events_total",
    "missing_mask_events_total",
    "total_runtime_s",
    "avg_runtime_s",
    "mean_tps",
    "draft_tokens_total",
    "rejected_tokens_total",
]

out = df[cols].copy()
out = out.sort_values(["workload", "mean_tps_speedup_vs_ar"], ascending=[True, False])

for c in [
    "mean_tps_speedup_vs_ar",
    "latency_speedup_vs_ar",
    "wasted_token_pct",
    "accepted_per_block",
    "total_runtime_s",
    "avg_runtime_s",
    "mean_tps",
]:
    out[c] = pd.to_numeric(out[c], errors="coerce").round(3)

for c in [
    "draft_tokens_total",
    "rejected_tokens_total",
    "n_blocks_total",
    "mask_events_total",
    "missing_mask_events_total",
]:
    out[c] = pd.to_numeric(out[c], errors="coerce").round(0).astype("Int64")

out.to_csv(root / "NORMAL_TABLE_by_workload_all_baselines_all_apex_configs_KFIX.csv", index=False)
(root / "NORMAL_TABLE_by_workload_all_baselines_all_apex_configs_KFIX.md").write_text(out.to_markdown(index=False))

print(out.to_string(index=False))
PY

echo
echo "=== Creating APEX-only ranking ==="
python3 - <<'PY'
import os
import pandas as pd
from pathlib import Path

root = Path(os.environ["KFIX_REPORT"])
df = pd.read_csv(root / "summary_overall_all_ablation_and_baselines_MASK.csv")

apex = df[df["system"].str.contains("KFIX", case=False, na=False)].copy()

cols = [
    "system",
    "n_requests",
    "mean_tps_speedup_vs_ar",
    "latency_speedup_vs_ar",
    "wasted_token_pct",
    "accepted_per_block",
    "n_blocks_total",
    "total_runtime_s",
    "avg_runtime_s",
    "mean_tps",
    "draft_tokens_total",
    "rejected_tokens_total",
]

apex = apex[cols].sort_values("mean_tps_speedup_vs_ar", ascending=False)

for c in [
    "mean_tps_speedup_vs_ar",
    "latency_speedup_vs_ar",
    "wasted_token_pct",
    "accepted_per_block",
    "total_runtime_s",
    "avg_runtime_s",
    "mean_tps",
]:
    apex[c] = pd.to_numeric(apex[c], errors="coerce").round(3)

for c in ["draft_tokens_total", "rejected_tokens_total", "n_blocks_total"]:
    apex[c] = pd.to_numeric(apex[c], errors="coerce").round(0).astype("Int64")

apex.to_csv(root / "APEX_ONLY_KFIX_ranked.csv", index=False)
(root / "APEX_ONLY_KFIX_ranked.md").write_text(apex.to_markdown(index=False))

print(apex.to_string(index=False))
PY

echo
echo "=== Creating active-k distribution ==="
python3 - <<'PY'
import os
import pandas as pd
from pathlib import Path

root = Path(os.environ["KFIX_REPORT"])
p = root / "active_k_all_ablations_raw_MASK.csv"
df = pd.read_csv(p)

g = (
    df.groupby(["run_family", "pool", "active_k"], dropna=False)
      .size()
      .reset_index(name="n_blocks")
      .sort_values(["run_family", "pool", "active_k"])
)

g.to_csv(root / "ACTIVE_K_DISTRIBUTION_KFIX.csv", index=False)
(root / "ACTIVE_K_DISTRIBUTION_KFIX.md").write_text(g.to_markdown(index=False))

print(g.to_string(index=False))
PY

echo
echo "=== Done ==="
echo "Report dir:"
echo "$KFIX_REPORT"
echo
echo "Main files:"
echo "$KFIX_REPORT/NORMAL_TABLE_all_baselines_all_apex_configs_KFIX.csv"
echo "$KFIX_REPORT/NORMAL_TABLE_all_baselines_all_apex_configs_KFIX.md"
echo "$KFIX_REPORT/NORMAL_TABLE_by_workload_all_baselines_all_apex_configs_KFIX.csv"
echo "$KFIX_REPORT/APEX_ONLY_KFIX_ranked.csv"
echo "$KFIX_REPORT/ACTIVE_K_DISTRIBUTION_KFIX.csv"
