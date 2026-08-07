#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def slugify(x: str) -> str:
    x = str(x).strip().replace("/", "_")
    x = re.sub(r"[^A-Za-z0-9_.=-]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_") or "unnamed"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True))


def iter_jsonl(path: str | Path):
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(errors="ignore").splitlines():
        # Some prior traces contain two JSON records joined by literal "\\n".
        for part in line.split("\\n"):
            part = part.strip()
            if not part:
                continue
            try:
                yield json.loads(part)
            except Exception:
                continue


def safe_read_csv(path: str | Path) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"[warn] failed reading {p}: {e}", file=sys.stderr)
        return None


def run_cmd(cmd: list[str], cwd: str | Path | None = None, check: bool = True) -> int:
    print("[cmd]", " ".join(cmd), flush=True)
    ret = subprocess.run(cmd, cwd=str(cwd) if cwd else None).returncode
    if check and ret != 0:
        raise RuntimeError(f"command failed with code {ret}: {' '.join(cmd)}")
    return ret


def infer_run_metadata(run_root: str | Path) -> dict:
    root = Path(run_root)
    name = root.name
    meta = {"run_root": str(root), "run_name": name}

    # Expected generated form:
    # <variant>__temp0p0__k4_8_16 or similar.
    parts = name.split("__")
    if parts:
        meta["model_variant"] = parts[0]
    for part in parts[1:]:
        if part.startswith("temp"):
            meta["temperature_label"] = part
            try:
                meta["temperature"] = float(part.replace("temp", "").replace("p", "."))
            except Exception:
                pass
        elif part.startswith("k"):
            meta["candidate_set"] = part

    # Fallback extraction from run.log/env not required.
    return meta


def add_waste_percent(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "wasted_token_rate" in df.columns:
        df["wasted_token_pct"] = 100.0 * pd.to_numeric(df["wasted_token_rate"], errors="coerce")
    return df


def summarize_request_results_from_json(result_dir: str | Path) -> pd.DataFrame:
    rows = []
    for p in Path(result_dir).rglob("*.json"):
        try:
            obj = json.loads(p.read_text())
        except Exception:
            continue
        if "tokens_per_sec" not in obj and "wall_s" not in obj:
            continue
        rows.append(obj)
    return pd.DataFrame(rows)


def save_plot_metric_by_variant(df: pd.DataFrame, metric: str, out_path: str | Path, x: str = "epoch", hue: str = "variant") -> None:
    import matplotlib.pyplot as plt

    if df.empty or metric not in df.columns or x not in df.columns:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5.5))
    for key, g in df.groupby(hue, dropna=False):
        g = g.sort_values(x)
        plt.plot(g[x], g[metric], marker="o", linewidth=1.8, label=str(key))
    plt.xlabel(x)
    plt.ylabel(metric)
    plt.title(metric)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_bar_plot(df: pd.DataFrame, x_col: str, y_col: str, out_path: str | Path, title: str | None = None, rotate: int = 25) -> None:
    import matplotlib.pyplot as plt

    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    d = df.copy()
    d[x_col] = d[x_col].astype(str)
    plt.figure(figsize=(max(8, 0.55 * len(d)), 5.5))
    plt.bar(d[x_col], pd.to_numeric(d[y_col], errors="coerce"))
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    if title:
        plt.title(title)
    plt.xticks(rotation=rotate, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_grouped_line_plot(df: pd.DataFrame, x_col: str, y_col: str, group_col: str, out_path: str | Path, title: str | None = None) -> None:
    import matplotlib.pyplot as plt

    if df.empty or not {x_col, y_col, group_col}.issubset(df.columns):
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5.5))
    for key, g in df.groupby(group_col, dropna=False):
        g = g.sort_values(x_col)
        plt.plot(g[x_col].astype(str), pd.to_numeric(g[y_col], errors="coerce"), marker="o", label=str(key))
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def read_summary_csv(run_root: str | Path, name: str) -> pd.DataFrame | None:
    return safe_read_csv(Path(run_root) / "summary" / name)


def concat_with_run_metadata(paths: Iterable[Path], filename: str) -> pd.DataFrame:
    rows = []
    for root in paths:
        df = read_summary_csv(root, filename)
        if df is None or df.empty:
            continue
        meta = infer_run_metadata(root)
        for k, v in meta.items():
            df[k] = v
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
