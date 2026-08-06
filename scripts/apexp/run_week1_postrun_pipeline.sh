#!/usr/bin/env bash
set -euo pipefail

cd /fsx/jmanvi/Internship_project/ASD
export PYTHONPATH="$PWD:$PYTHONPATH"

ROOT="${1:-results/apexp_block/config_full}"

echo "ROOT=$ROOT"

echo
echo "=== 1. Build final datasets ==="
bash scripts/apexp/build_config_outputs.sh "$ROOT"

echo
echo "=== 2. Create Week-1 tables and plots ==="
python src/apexp/week1/make_week1_tables.py \
  --root "$ROOT"

echo
echo "=== 3. Train first deployable-lite controller ==="
python src/apexp/week1/train_week1_controller.py \
  --root "$ROOT"

echo
echo "=== 4. Train post-hoc state-feature controller ==="
python src/apexp/week1/train_week1_controller.py \
  --root "$ROOT" \
  --include-state-features \
  --out-dir "$ROOT/week1/controller_state_features"

echo
echo "=== 5. Final Week-1 summary ==="
python - "$ROOT" <<'PY'
from pathlib import Path
import pandas as pd
import json
import sys

root = Path(sys.argv[1])
week1 = root / "week1"

print("Root:", root)

print("\nDataset summary:")
print((week1 / "tables" / "week1_dataset_summary.json").read_text())

print("\nFailed jobs:")
failed = week1 / "tables" / "failed_jobs.csv"
if failed.exists():
    df = pd.read_csv(failed)
    print(df.to_string(index=False))
else:
    print("No failed_jobs.csv")

print("\nTop fixed candidates:")
p = week1 / "tables" / "table_best_global_fixed.csv"
if p.exists():
    df = pd.read_csv(p)
    print(df.head(15).to_string(index=False))

print("\nBest fixed by workload:")
p = week1 / "tables" / "table_best_fixed_by_workload.csv"
if p.exists():
    df = pd.read_csv(p)
    cols = [c for c in ["workload","candidate_id","method","k","speedup_vs_ar","mean_tps","mean_latency_s"] if c in df.columns]
    print(df[cols].to_string(index=False))

print("\nController summary:")
p = week1 / "controller" / "controller_eval_summary.csv"
if p.exists():
    df = pd.read_csv(p)
    print(df.to_string(index=False))

print("\nState-feature controller summary:")
p = week1 / "controller_state_features" / "controller_eval_summary.csv"
if p.exists():
    df = pd.read_csv(p)
    print(df.to_string(index=False))

print("\nImportant outputs:")
for p in [
    week1 / "tables" / "table_request_speedup_by_workload_method_k.csv",
    week1 / "tables" / "table_block_acceptance_by_workload_method_k.csv",
    week1 / "tables" / "table_best_fixed_by_workload.csv",
    week1 / "tables" / "table_oracle_choice_distribution.csv",
    week1 / "controller" / "controller_eval_summary.csv",
    week1 / "controller_state_features" / "controller_eval_summary.csv",
]:
    print(p, "exists=", p.exists())
PY

echo
echo "DONE Week-1 pipeline."
