#!/usr/bin/env python
"""
run_staleness_study.py — orchestrate the staleness proxy study
==============================================================
Expands the factorial config (staleness_config.json) against the drift points
in drift_manifest.csv and drives your existing run_benchmark.py for each cell,
so all SD measurement / metric logging is inherited unchanged.

Each cell becomes a run_benchmark.py invocation with:
  --model  <drift checkpoint path>             (the proxy-drifted target)
  --experiment  S_stale__<study>__<drift_tag>  (so aggregate can recover drift)
  --method/--k/--temperature/--workload/...    (the swept axes)

Execution model:
  * Grouped by drift checkpoint. All cells for one checkpoint run first
    (parallel across GPUs); then, if --cleanup, that ~16 GB checkpoint is
    deleted before the next one. Keeps peak disk to one checkpoint.
  * Up to --gpus concurrent runs, each pinned to one CUDA_VISIBLE_DEVICES,
    tensor_parallel_size=1. spawn multiproc is forced (parent inits CUDA).
  * draft_sd cells get --enforce_eager + lower gpu_mem (the Phase-2 fix).

Modes:
  --plan_only   print the grid and per-study cell counts, then exit
  --dry_run     print every command, then exit
  (default)     execute

Example:
  python run_staleness_study.py --config staleness_proxy/staleness_config.json \
      --repo_root /fsx/jmanvi/Internship_project/ASD --gpus 8 --plan_only
  python run_staleness_study.py --config ... --repo_root ... --gpus 8 --cleanup
"""
import argparse
import csv
import json
import os
import shlex
import subprocess
import time
from pathlib import Path


def load_config(p):
    with open(p) as f:
        return json.load(f)


def load_manifest(p):
    p = Path(p)
    if not p.exists():
        raise SystemExit(f"drift manifest not found: {p}\n"
                         f"Run make_drifted_models.py first.")
    return list(csv.DictReader(open(p)))


def resolve_workloads(spec, cfg):
    if spec == "all_workloads":
        return cfg["all_workloads"]
    if spec == "representative_workloads":
        return cfg["representative_workloads"]
    if isinstance(spec, list):
        return spec
    raise SystemExit(f"bad workloads spec: {spec}")


def valid_checkpoint(path: str) -> bool:
    """Local dir -> must contain config.json + weights. Otherwise, treat a
    HF-style repo id ('namespace/name', not an absolute/relative local path)
    as valid and let vLLM resolve it."""
    p = Path(path)
    if p.is_absolute() or path.startswith(".") or p.exists():
        # it's meant to be a local path -> require a real checkpoint there
        return (p / "config.json").exists() and (
            (p / "model.safetensors").exists()
            or (p / "pytorch_model.bin").exists()
            or any(p.glob("*.safetensors")))
    # looks like a hub repo id (e.g. 'allenai/Llama-3.1-Tulu-3-8B-SFT')
    return path.count("/") == 1 and " " not in path

def preflight(cells):
    """Drop cells whose drift checkpoint is missing/incomplete; report them."""
    paths = sorted({c["drift_path"] for c in cells})
    bad = [p for p in paths if not valid_checkpoint(p)]
    if bad:
        print(f"\n[preflight] {len(bad)} drift checkpoint(s) missing or "
              f"incomplete (no config.json / weights) -- skipping their cells:")
        for p in bad:
            print(f"    MISSING: {p}")
        print("    -> re-run make_drifted_models.py (patched safetensors save) "
              "to create them.\n")
    bad_set = set(bad)
    return [c for c in cells if c["drift_path"] not in bad_set]


def build_cells(cfg, repo_root):
    """Return list of cells across all drift families x studies."""
    paths, dflt = cfg["paths"], cfg["defaults"]
    runner = str(Path(repo_root) / paths["runner"])
    data_dir = Path(repo_root) / paths["data_dir"]
    out_dir = str(Path(repo_root) / paths["out_dir"])
    families = cfg["drift_families"]
    wl_files = paths["workload_files"]            # explicit workload -> filename map

    # load each family's manifest once
    fam_manifest = {}
    for fam, fc in families.items():
        mpath = fc["manifest"]
        mpath = mpath if Path(mpath).is_absolute() else str(Path(repo_root) / mpath)
        if not Path(mpath).exists():
            print(f"  [warn] manifest missing for family '{fam}': {mpath} "
                  f"(run make_drifted_models.py --family {fam} first) — skipping")
            continue
        fam_manifest[fam] = load_manifest(mpath)

    cells = []
    for study, sc in cfg["studies"].items():
        if sc.get("enabled", True) is False:
            continue
        fam_list = (list(families) if sc.get("families", "ALL") == "ALL"
                    else sc["families"])
        workloads = resolve_workloads(sc["workloads"], cfg)
        for fam in fam_list:
            if fam not in fam_manifest:
                continue
            fc = families[fam]
            methods = [m for m in sc["methods"] if m in fc["methods_available"]]
            for d in fam_manifest[fam]:
                # normalize tag: strip any "<family>__" prefix so the experiment
                # name is uniform whether or not the manifest tag was prefixed
                base_tag = d["tag"]
                if base_tag.startswith(f"{fam}__"):
                    base_tag = base_tag[len(fam) + 2:]
                exp = f"S_stale__{study}__{fam}__{base_tag}"
                for method in methods:
                    for k in sc["k"]:
                        for temp in sc["temperature"]:
                            for wl in workloads:
                                if wl not in wl_files:
                                    raise SystemExit(
                                        f"workload '{wl}' missing from "
                                        f"paths.workload_files in the config")
                                jsonl = str(data_dir / wl_files[wl])
                                cmd = [
                                    "python", runner,
                                    "--workload", wl, "--run_name", wl,
                                    "--experiment", exp,
                                    "--input_jsonl", jsonl,
                                    "--out_dir", out_dir,
                                    "--model", d["path"],
                                    "--method", method,
                                    "--k", str(k),
                                    "--temperature", str(temp),
                                    "--limit", str(dflt["limit"]),
                                    "--max_tokens", str(dflt["max_tokens"]),
                                    "--seed", str(dflt["seed"]),
                                    "--tensor_parallel_size", str(dflt["tensor_parallel_size"]),
                                ]
                                if method == "ngram_sd":
                                    cmd += ["--ngram_lookup_min", str(dflt["ngram_lookup_min"]),
                                            "--ngram_lookup_max", str(dflt["ngram_lookup_max"])]
                                if method == "draft_sd":
                                    cmd += ["--draft_model", fc["draft_model"],
                                            "--enforce_eager",
                                            "--gpu_memory_utilization", "0.75"]
                                else:
                                    cmd += ["--gpu_memory_utilization",
                                            str(dflt["gpu_memory_utilization"])]
                                if method == "eagle3":
                                    cmd += ["--eagle3_model", fc["eagle3_model"]]
                                cells.append(dict(study=study, family=fam,
                                                  drift_tag=d["tag"], base_tag=base_tag,
                                                  mode=d.get("mode", ""),
                                                  param=d.get("param", ""),
                                                  rel_weight_dist=d.get("rel_weight_dist", ""),
                                                  drift_path=d["path"], method=method,
                                                  k=k, temp=temp, workload=wl,
                                                  experiment=exp, cmd=cmd))
    return cells


def write_index(cells, path):
    """Authoritative experiment -> (family, drift, distance) map for aggregation,
    so the aggregator never depends on manifest tag-prefix consistency."""
    seen, rows = set(), []
    for c in cells:
        if c["experiment"] in seen:
            continue
        seen.add(c["experiment"])
        rows.append({k: c[k] for k in ("experiment", "study", "family", "drift_tag",
                                       "base_tag", "mode", "param", "rel_weight_dist")})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  wrote staleness index -> {path} ({len(rows)} experiments)")


def already_done(out_dir, exp, wl, method):
    """Robust skip check: any summary.csv under out_dir/exp/wl/method/ (don't
    guess run_benchmark's exact leaf-dir naming)."""
    base = Path(out_dir) / exp / wl / method
    return base.exists() and any(base.glob("**/summary.csv"))


def run_group(cells, gpus, logdir, env, skip_existing, out_dir):
    """Run all cells of one drift group with up to `gpus` concurrent procs."""
    logdir.mkdir(parents=True, exist_ok=True)
    todo = []
    print(out_dir)
    for c in cells:
        if skip_existing and already_done(out_dir, c["experiment"], c["workload"], c["method"]):
            
            continue
        todo.append(c)

    running = {}   # gpu -> (proc, cell, logfile_handle)
    free = list(range(gpus))
    idx = 0
    results = []
    while idx < len(todo) or running:
        while free and idx < len(todo):
            gpu = free.pop(0)
            c = todo[idx]; idx += 1
            cenv = dict(env)
            cenv["CUDA_VISIBLE_DEVICES"] = str(gpu)
            cenv["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            lf = open(logdir / f"gpu{gpu}_{c['experiment']}_{c['workload']}_"
                              f"{c['method']}_k{c['k']}_T{c['temp']}.log", "w")
            print(f"  [GPU {gpu}] {c['experiment']} | {c['workload']} | "
                  f"{c['method']} k={c['k']} T={c['temp']}")
            p = subprocess.Popen(c["cmd"], stdout=lf, stderr=subprocess.STDOUT, env=cenv)
            running[gpu] = (p, c, lf)
        time.sleep(2)
        for gpu, (p, c, lf) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                lf.close()
                ok = (rc == 0)
                results.append((c, ok))
                if not ok:
                    print(f"  [GPU {gpu}] FAILED rc={rc}: {c['experiment']} "
                          f"{c['workload']} {c['method']} k{c['k']}")
                del running[gpu]
                free.append(gpu)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--repo_root", required=True,
                    help="Repo root that contains src/ and data/ (for absolute paths).")
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--logdir", default="staleness_logs")
    ap.add_argument("--cleanup", action="store_true",
                    help="Delete each drift checkpoint after its cells finish (bounds disk).")
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    ap.add_argument("--plan_only", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--index", default=None,
                    help="Where to write the experiment->drift index "
                         "(default: <repo_root>/results/staleness_index.csv).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cells = build_cells(cfg, args.repo_root)
    cells = preflight(cells)

    index_path = (args.index if args.index
                  else str(Path(args.repo_root) / "results" / "staleness_index.csv"))
    if cells:
        write_index(cells, index_path)

    # report
    from collections import Counter
    per_study = Counter((c["study"], c["family"]) for c in cells)
    print(f"Total cells: {len(cells)}  (gpus: {args.gpus})")
    for (s, fam), n in sorted(per_study.items()):
        print(f"  {s:28s} {fam:18s} {n:5d} cells")
    if args.plan_only:
        return

    if args.dry_run:
        for c in cells[:50]:
            print(" ".join(shlex.quote(x) for x in c["cmd"]))
        print(f"... ({len(cells)} total)")
        return

    out_dir = str(Path(args.repo_root) / cfg["paths"]["out_dir"])
    env = dict(os.environ)
    # group by drift checkpoint, run group, then optional cleanup
    groups = {}
    for c in cells:
        groups.setdefault(c["drift_path"], []).append(c)

    all_results = []
    for gi, (dpath, gcells) in enumerate(groups.items(), 1):
        print(f"\n=== drift group {gi}/{len(groups)}: {dpath} "
              f"({len(gcells)} cells) ===")
        res = run_group(gcells, args.gpus, Path(args.logdir), env,
                        args.skip_existing, out_dir)
        all_results += res
        if args.cleanup and Path(dpath).exists() and "drifted" in dpath:
            import shutil
            print(f"  [cleanup] rm -rf {dpath}")
            shutil.rmtree(dpath, ignore_errors=True)

    n_ok = sum(1 for _, ok in all_results if ok)
    print(f"\nDone. {n_ok}/{len(all_results)} cells succeeded.")
    if n_ok < len(all_results):
        print("Re-run failures with --skip_existing (default) to fill gaps.")


if __name__ == "__main__":
    main()
