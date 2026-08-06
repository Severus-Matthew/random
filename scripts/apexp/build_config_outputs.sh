#!/usr/bin/env bash
set -euo pipefail

cd /fsx/jmanvi/Internship_project/ASD
export PYTHONPATH="$PWD:$PYTHONPATH"

OUT_ROOT="${1:-results/apexp_block/config_full}"

echo "OUT_ROOT=$OUT_ROOT"

echo
echo "=== Failed jobs ==="
if [[ -f "$OUT_ROOT/job_status.tsv" ]]; then
  grep -v "SUCCESS\|SKIPPED" "$OUT_ROOT/job_status.tsv" || true
else
  echo "No job_status.tsv found"
fi

echo
echo "=== Spec trace dirs ==="
mapfile -t TRACE_DIRS < <(
  find "$OUT_ROOT/runs" -mindepth 1 -maxdepth 1 -type d | sort | while read -r d; do
    if [[ -s "$d/block_events.jsonl" ]]; then
      echo "$d"
    fi
  done
)

printf '%s\n' "${TRACE_DIRS[@]}"

if [[ "${#TRACE_DIRS[@]}" -eq 0 ]]; then
  echo "ERROR: no block_events.jsonl files found"
  exit 1
fi

echo
echo "=== Build block survival dataset ==="
python src/apexp/trace/build_block_dataset.py \
  --inputs "$OUT_ROOT/runs" \
  --out-dir "$OUT_ROOT/block_dataset"

echo
echo "=== Join block events with prompt/request features ==="
python src/apexp/trace/join_block_with_request_features.py \
  --trace-dirs "${TRACE_DIRS[@]}" \
  --out-dir "$OUT_ROOT/block_joined"

echo
echo "=== Collect request summaries, including AR ==="
python - "$OUT_ROOT" <<'PY'
from pathlib import Path
import pandas as pd
import sys

out_root = Path(sys.argv[1])
files = sorted(out_root.rglob("summary.csv"))

frames = []
for f in files:
    df = pd.read_csv(f)
    df["summary_file"] = str(f)
    frames.append(df)

if not frames:
    raise SystemExit("No summary.csv found")

full = pd.concat(frames, ignore_index=True)
out = out_root / "request_summary_all.csv"
full.to_csv(out, index=False)

print("Wrote:", out)
print("shape:", full.shape)
print(full.groupby(["workload", "method", "k"]).size())
PY

echo
echo "=== Final sanity checks ==="
python - "$OUT_ROOT" <<'PY'
from pathlib import Path
import pandas as pd
import json
import sys

out_root = Path(sys.argv[1])
joined_path = out_root / "block_joined" / "block_with_request_features.csv"
diag_path = out_root / "block_joined" / "join_diagnostics.json"
req_path = out_root / "request_summary_all.csv"

df = pd.read_csv(joined_path)
req = pd.read_csv(req_path)

print("joined shape:", df.shape)
print("request summary shape:", req.shape)

print()
print("blocks by workload/method/k:")
print(df.groupby(["workload", "method", "k_actual"]).size())

print()
print("requests by workload/method/k:")
print(req.groupby(["workload", "method", "k"]).size())

print()
print("missing block fields:")
for c in [
    "prompt_id",
    "prompt_hash",
    "prompt_token_len",
    "mean_entropy",
    "repetition_density",
    "n_output_tokens",
    "latency_s",
    "tokens_per_sec",
]:
    print(c, df[c].isna().sum() if c in df.columns else "MISSING_COLUMN")

print()
print("accepted_len histogram:")
print(df["accepted_len"].value_counts().sort_index())

print()
print("mean accepted_len by workload/method/k:")
print(df.groupby(["workload", "method", "k_actual"])["accepted_len"].mean())

print()
print("draft_sd request counts:")
print(req[req["method"] == "draft_sd"].groupby(["workload", "method", "k"]).size())

print()
print("join diagnostics first 5000 chars:")
print(diag_path.read_text()[:5000])
PY
