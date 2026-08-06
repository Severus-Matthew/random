#!/usr/bin/env python
"""
gen_jobs.py  —  generates flat job list from experiments_config.json
Called by run_full_study.sh as a clean subprocess.

Usage:
  python gen_jobs.py --config configs/experiments_config.json --out /tmp/jobs.jsonl
  python gen_jobs.py --config ... --out ... --filter E      # only E_* experiments
  python gen_jobs.py --config ... --out ... --dry-run       # count only
"""
import argparse, json, sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Experiment definitions
# All experiments are defined here in code so changes are version-controlled
# and self-documenting without needing a separate JSON file.
# ─────────────────────────────────────────────────────────────────────────────

def build_experiments(g: dict) -> list:
    """
    Returns the full experiment list.  g = global config dict.
    Each experiment is a dict with keys:
      name, description, workloads, configs, context_lengths, turn_depths
    """
    target     = g["target_model"]
    eagle3     = g["eagle3_model"]
    draft_25   = "Qwen/Qwen2.5-1.5B-Instruct"   # mismatched baseline
    draft_06   = "Qwen/Qwen3-0.6B"               # Phase 2 family sweep
    draft_17   = "Qwen/Qwen3-1.7B"               # Phase 2 family sweep
    draft_4B   = "Qwen/Qwen3-4B"                 # Phase 2 family sweep
    all_wl     = "all"

    return [

        # ── A: Baseline sweep ─────────────────────────────────────────────
        # Purpose: establish per-workload per-method TPS and acceptance rate
        # at fixed k=4, temperature=0.0.  This is the Phase 1 comparison table.
        {
            "name":        "A_baseline_sweep",
            "description": "Baseline: all methods at k=4 across all workloads",
            "workloads":   all_wl,
            "configs": [
                {"method":"ar",       "k":1, "temperature":0.0, "draft_model":""},
                {"method":"ngram_sd", "k":4, "temperature":0.0, "draft_model":""},
                {"method":"draft_sd", "k":4, "temperature":0.0, "draft_model":draft_06},
                {"method":"eagle3",   "k":4, "temperature":0.0, "draft_model":"",
                 "eagle3_model":eagle3},
            ],
            "context_lengths": [0],
            "turn_depths":     [1],
        },

        # ── B: k sweep ────────────────────────────────────────────────────
        # Purpose: find optimal k per method per workload.
        # Feeds oracle k analysis and adaptive controller design.
        {
            "name":        "B_k_sweep",
            "description": "k sensitivity: k in {1,2,4,8,16} for all speculative methods",
            "workloads":   all_wl,
            "configs": [
                *[{"method":"ngram_sd", "k":k, "temperature":0.0, "draft_model":""}
                  for k in [1,2,4,8,16]],
                *[{"method":"draft_sd", "k":k, "temperature":0.0, "draft_model":draft_06}
                  for k in [1,2,4,8,16]],
                *[{"method":"eagle3",   "k":k, "temperature":0.0, "draft_model":"",
                   "eagle3_model":eagle3}
                  for k in [1,2,4,8,16]],
            ],
            "context_lengths": [0],
            "turn_depths":     [1],
        },

        # ── C: Ngram config sweep ─────────────────────────────────────────
        # Purpose: find optimal lookup window (min,max) for ngram speculation.
        {
            "name":        "C_ngram_config_sweep",
            "description": "Ngram lookup window sensitivity",
            "workloads":   all_wl,
            "configs": [
                *[{"method":"ngram_sd","k":k,"temperature":0.0,"draft_model":"",
                   "ngram_lookup_min":lmin,"ngram_lookup_max":lmax}
                  for k in [4,8,16]
                  for lmin,lmax in [(1,4),(1,8),(2,8),(4,16)]],
            ],
            "context_lengths": [0],
            "turn_depths":     [1],
        },

        # ── D: Draft model family sweep (updated — now 3 models) ─────────
        # Purpose: quantify benefit of family-aligned draft models.
        # Qwen3-0.6B: cheapest draft, lowest quality.
        # Qwen3-1.7B: medium draft — likely sweet spot.
        # Qwen3-4B:   expensive draft — approaches verifier quality.
        {
            "name":        "D_draft_model_sweep",
            "description": "Qwen3 draft family: 0.6B / 1.7B / 4B vs mismatched Qwen2.5-1.5B",
            "workloads":   ["code_gen","long_chain_reasoning","mathematical_reasoning"],
            "configs": [
                # Mismatched baseline (already in A, re-run for direct comparison)
                *[{"method":"draft_sd","k":k,"temperature":0.0,"draft_model":draft_25}
                  for k in [4,8]],
                # Qwen3 0.6B
                *[{"method":"draft_sd","k":k,"temperature":0.0,"draft_model":draft_06}
                  for k in [4,8]],
                # Qwen3 1.7B
                *[{"method":"draft_sd","k":k,"temperature":0.0,"draft_model":draft_17}
                  for k in [4,8]],
                # Qwen3 4B
                *[{"method":"draft_sd","k":k,"temperature":0.0,"draft_model":draft_4B}
                  for k in [4,8]],
            ],
            "context_lengths": [0],
            "turn_depths":     [1],
        },

        # ── E: Temperature sweep (NOW INCLUDES draft_sd) ─────────────────
        # Purpose: how does stochastic sampling affect acceptance rate?
        # Critical for RL rollout generation (always uses T > 0).
        # draft_sd added to compare mismatch degradation under temperature.
        {
            "name":        "E_temperature_sweep",
            "description": "Temperature sensitivity: T in {0.3, 0.6, 1.0} — all 4 methods",
            "workloads":   all_wl,
            "configs": [
                *[{"method":"ar",       "k":1, "temperature":t, "draft_model":""}
                  for t in [0.3, 0.6, 1.0]],
                *[{"method":"ngram_sd", "k":4, "temperature":t, "draft_model":""}
                  for t in [0.3, 0.6, 1.0]],
                # draft_sd now included — compare mismatch degradation under temperature
                *[{"method":"draft_sd", "k":4, "temperature":t, "draft_model":draft_06}
                  for t in [0.3, 0.6, 1.0]],
                *[{"method":"eagle3",   "k":4, "temperature":t, "draft_model":"",
                   "eagle3_model":eagle3}
                  for t in [0.3, 0.6, 1.0]],
            ],
            "context_lengths": [0],
            "turn_depths":     [1],
        },

        # ── F: Context growth (NOW INCLUDES draft_sd) ────────────────────
        # Purpose: how does prompt length affect speculative efficiency?
        # draft_sd added: longer context does NOT help draft_sd (mismatch is
        # distributional not coverage-based) — important contrast to show.
        {
            "name":        "F_context_growth",
            "description": "Context length scaling: 64–1024 tokens — all 4 methods",
            "workloads":   all_wl,
            "configs": [
                {"method":"ar",       "k":1, "temperature":0.0, "draft_model":""},
                {"method":"ngram_sd", "k":4, "temperature":0.0, "draft_model":""},
                # draft_sd added — shows context does NOT help mismatched draft
                {"method":"draft_sd", "k":4, "temperature":0.0, "draft_model":draft_06},
                {"method":"eagle3",   "k":4, "temperature":0.0, "draft_model":"",
                 "eagle3_model":eagle3},
            ],
            "context_lengths": [64, 128, 256, 512, 1024],
            "turn_depths":     [1],
        },

        # ── G: Multi-turn (NOW INCLUDES draft_sd) ────────────────────────
        # Purpose: how does conversation history affect speculative efficiency?
        # draft_sd added: richer context has different effect on mismatched
        # draft vs aligned draft (hypothesis: mismatch unaffected by turns).
        {
            "name":        "G_multi_turn",
            "description": "Multi-turn depth: 1–8 turns — all 4 methods",
            "workloads":   ["conversational_generation","code_gen","long_horizon_swe"],
            "configs": [
                {"method":"ar",       "k":1, "temperature":0.0, "draft_model":""},
                {"method":"ngram_sd", "k":4, "temperature":0.0, "draft_model":""},
                # draft_sd added — test if turn accumulation helps or hurts mismatch
                {"method":"draft_sd", "k":4, "temperature":0.0, "draft_model":draft_06},
                {"method":"eagle3",   "k":4, "temperature":0.0, "draft_model":"",
                 "eagle3_model":eagle3},
            ],
            "context_lengths": [0],
            "turn_depths":     [1, 2, 4, 8],
        },

        # ── H: Oracle k study ─────────────────────────────────────────────
        # Purpose: establish the theoretical ceiling of the adaptive controller.
        # Run each prompt at ALL k values; post-processing selects oracle_k
        # per prompt from positional acceptance data and computes gap vs k=4.
        # The oracle_k_vs_k4_gain column in summary.csv captures this directly.
        {
            "name":        "H_oracle_k_study",
            "description": "Oracle k: run all k values per prompt to find retrospective optimum",
            "workloads":   all_wl,
            "configs": [
                # Eagle3: main method for oracle study
                *[{"method":"eagle3",   "k":k, "temperature":0.0, "draft_model":"",
                   "eagle3_model":eagle3}
                  for k in [1,2,4,8,16]],
                # Ngram-SD: oracle analysis for lookup-based method
                *[{"method":"ngram_sd", "k":k, "temperature":0.0, "draft_model":""}
                  for k in [1,2,4,8,16]],
            ],
            "context_lengths": [0],
            "turn_depths":     [1],
        },

        # ── I: Entropy-bucketed analysis ──────────────────────────────────
        # Purpose: stratify ALL results by entropy bucket (low/medium/high)
        # and measure speedup within each bucket.
        # This produces the lookup table: entropy_bucket → optimal (k, method).
        # The entropy_bucket column is now set automatically in run_benchmark.py.
        # This experiment runs a fresh sweep with bucket-aware analysis enabled.
        {
            "name":        "I_entropy_bucket_study",
            "description": "Entropy-stratified analysis: speedup within low/medium/high entropy regimes",
            "workloads":   all_wl,
            "configs": [
                # All methods at k=4 (bucket analysis done in post-processing)
                {"method":"ar",       "k":1, "temperature":0.0, "draft_model":""},
                {"method":"ngram_sd", "k":4, "temperature":0.0, "draft_model":""},
                {"method":"draft_sd", "k":4, "temperature":0.0, "draft_model":draft_06},
                {"method":"eagle3",   "k":4, "temperature":0.0, "draft_model":"",
                 "eagle3_model":eagle3},
                # Also sweep k for eagle3/ngram to get bucket-conditional optimal k
                *[{"method":"eagle3",   "k":k, "temperature":0.0, "draft_model":"",
                   "eagle3_model":eagle3}
                  for k in [8,16]],
                *[{"method":"ngram_sd", "k":k, "temperature":0.0, "draft_model":""}
                  for k in [8,16]],
            ],
            "context_lengths": [0],
            "turn_depths":     [1],
        },

        # ── J: Intra-sequence regime transition ───────────────────────────
        # Purpose: study how entropy evolves WITHIN a single generation.
        # Uses long-output workloads where regime transitions are most visible.
        # The entropy_w0/w1/w2 and entropy_trend columns capture this.
        # Key hypothesis: long-chain reasoning starts uncertain (problem setup)
        # then drops (repetitive computation) then rises (conclusion).
        {
            "name":        "J_intra_sequence_transition",
            "description": "Intra-sequence entropy transitions — long-output workloads",
            "workloads":   ["long_chain_reasoning","long_context_completion",
                            "long_horizon_swe","mathematical_reasoning"],
            "configs": [
                # High output length to maximise regime transitions
                {"method":"ar",       "k":1, "temperature":0.0, "draft_model":""},
                {"method":"ngram_sd", "k":4, "temperature":0.0, "draft_model":""},
                {"method":"ngram_sd", "k":16,"temperature":0.0, "draft_model":""},
                {"method":"eagle3",   "k":4, "temperature":0.0, "draft_model":"",
                 "eagle3_model":eagle3},
                {"method":"eagle3",   "k":16,"temperature":0.0, "draft_model":"",
                 "eagle3_model":eagle3},
            ],
            "context_lengths": [0],
            "turn_depths":     [1],
        },

        # ── K: Rejection deep-dive ─────────────────────────────────────────
        # Purpose: systematic study of WHEN and WHERE rejections occur.
        # Uses all new rejection metrics: rejection_rate, first_rejection_pos,
        # rejection_concentration, rejection_severity, acceptance_decay_slope.
        # Key questions:
        #   1. Do rejections cluster at specific token positions? (concentration)
        #   2. Does rejection rate correlate with local entropy at time of rejection?
        #   3. How does rejection severity vary across workloads?
        #   4. Does the acceptance decay slope predict throughput better than
        #      the mean acceptance rate alone?
        {
            "name":        "K_rejection_study",
            "description": "Rejection characterization: concentration, severity, decay",
            "workloads":   all_wl,
            "configs": [
                # Multiple k values to see how rejection profiles change with k
                *[{"method":"eagle3",   "k":k, "temperature":0.0, "draft_model":"",
                   "eagle3_model":eagle3}
                  for k in [2,4,8,16]],
                *[{"method":"ngram_sd", "k":k, "temperature":0.0, "draft_model":""}
                  for k in [2,4,8,16]],
                *[{"method":"draft_sd", "k":k, "temperature":0.0, "draft_model":draft_06}
                  for k in [2,4,8]],
                # Also at T=0.6 — does temperature change rejection patterns?
                *[{"method":"eagle3",   "k":4, "temperature":0.6, "draft_model":"",
                   "eagle3_model":eagle3}],
                *[{"method":"ngram_sd", "k":4, "temperature":0.6, "draft_model":""}],
            ],
            "context_lengths": [0],
            "turn_depths":     [1],
        },
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",   required=True,
                    help="Path to experiments_config.json (global settings)")
    ap.add_argument("--out",      required=True,
                    help="Output JSONL path for job list")
    ap.add_argument("--filter",   default="",
                    help="Only include experiments whose name starts with this prefix")
    ap.add_argument("--dry-run",  action="store_true",
                    help="Print job count and sample, do not write file")
    ap.add_argument("--list",     action="store_true",
                    help="List all experiment names and descriptions then exit")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    g         = cfg["global"]
    wf        = cfg["workload_files"]
    wmt       = cfg["workload_max_tokens"]
    all_wls   = list(wf.keys())
    data_dir  = g["data_dir"]
    results   = g["results_dir"]
    limit     = g["limit"]
    seed      = g["seed"]

    experiments = build_experiments(g)

    if args.list:
        print(f"{'Name':<35} {'Description'}")
        print("-"*80)
        for exp in experiments:
            print(f"{exp['name']:<35} {exp['description']}")
        return

    jobs = []
    for exp in experiments:
        name = exp["name"]
        if args.filter and not name.startswith(args.filter):
            continue

        workloads       = all_wls if exp["workloads"] == "all" else exp["workloads"]
        context_lengths = exp.get("context_lengths", [0])
        turn_depths     = exp.get("turn_depths",     [1])

        for workload in workloads:
            if workload not in wf:
                print(f"[WARN] workload '{workload}' not in workload_files, skipping",
                      file=sys.stderr)
                continue
            jsonl   = f"{data_dir}/{wf[workload]}"
            max_tok = wmt.get(workload, 512)

            for conf in exp["configs"]:
                method   = conf["method"]
                k        = conf["k"]
                temp     = conf["temperature"]
                nlmin    = conf.get("ngram_lookup_min", 1)
                nlmax    = conf.get("ngram_lookup_max", 4)
                draft    = conf.get("draft_model", "")
                eagle    = conf.get("eagle3_model",
                                    g.get("eagle3_model", "") if method=="eagle3" else "")

                for ctx in context_lengths:
                    for turns in turn_depths:
                        jobs.append({
                            "experiment":        name,
                            "workload":          workload,
                            "run_name":          workload,
                            "method":            method,
                            "k":                 k,
                            "temperature":       temp,
                            "ngram_lookup_min":  nlmin,
                            "ngram_lookup_max":  nlmax,
                            "draft_model":       draft,
                            "eagle3_model":      eagle,
                            "input_jsonl":       jsonl,
                            "max_tokens":        max_tok,
                            "max_prompt_tokens": ctx,
                            "num_turns":         turns,
                            "limit":             limit,
                            "seed":              seed,
                            "out_dir":           results,
                            "model":             g["target_model"],
                        })

    if args.dry_run:
        print(f"{len(jobs)} jobs would be generated "
              f"(filter='{args.filter or 'all'}')", file=sys.stderr)
        for j in jobs[:15]:
            print(f"  {j['experiment']:35s}  {j['workload']:30s}  "
                  f"{j['method']:10s}  k={j['k']:2d}  T={j['temperature']}  "
                  f"ctx={j['max_prompt_tokens']:4d}  turns={j['num_turns']}")
        if len(jobs) > 15:
            print(f"  ... and {len(jobs)-15} more")

        # Print per-experiment breakdown
        from collections import Counter
        counts = Counter(j["experiment"] for j in jobs)
        print(f"\nPer-experiment job counts:")
        for exp_name, cnt in sorted(counts.items()):
            print(f"  {exp_name:<35s}  {cnt:4d} jobs")
        return

    Path(args.out).write_text("\n".join(json.dumps(j) for j in jobs) + "\n")
    print(f"Generated {len(jobs)} jobs -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()