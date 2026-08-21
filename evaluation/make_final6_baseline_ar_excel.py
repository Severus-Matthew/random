#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_EVAL_ROOT = Path(
    "/fsx/jmanvi/Internship_project/ASD/results/apexp_block/config_large_500/evaluation/paper_eval_final_full6"
)

KEEP_WORKLOADS = {
    "code_gen",
    "conversational_generation",
    "long_chain_reasoning",
    "long_context_completion",
    "long_horizon_swe",
    "mathematical_reasoning",
}

KEEP_METHODS = {"ar", "ngram_sd", "draft_sd", "eagle3"}


def col_letter(n: int) -> str:
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def xml_escape(x: Any) -> str:
    s = "" if x is None else str(x)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def is_number(x: Any) -> bool:
    if x is None:
        return False
    if isinstance(x, bool):
        return False
    try:
        v = float(x)
        return math.isfinite(v)
    except Exception:
        return False


def df_to_sheet_xml(df: pd.DataFrame) -> str:
    rows = [list(df.columns)] + df.astype(object).where(pd.notna(df), "").values.tolist()

    xml = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
        "<sheetData>",
    ]

    for r_idx, row in enumerate(rows, start=1):
        xml.append(f'<row r="{r_idx}">')
        for c_idx, val in enumerate(row, start=1):
            ref = f"{col_letter(c_idx)}{r_idx}"
            if val == "":
                xml.append(f'<c r="{ref}"/>')
            elif r_idx > 1 and is_number(val):
                xml.append(f'<c r="{ref}"><v>{float(val):.12g}</v></c>')
            else:
                xml.append(
                    f'<c r="{ref}" t="inlineStr"><is><t>{xml_escape(val)}</t></is></c>'
                )
        xml.append("</row>")

    xml.extend(["</sheetData>", "</worksheet>"])
    return "\n".join(xml)


def write_xlsx(path: Path, sheets: dict[str, pd.DataFrame]):
    path.parent.mkdir(parents=True, exist_ok=True)

    sheet_names = []
    used = set()
    for name in sheets:
        clean = re.sub(r"[\[\]\:\*\?\/\\]", "_", name)[:31] or "Sheet"
        base = clean
        i = 1
        while clean in used:
            suffix = f"_{i}"
            clean = base[: 31 - len(suffix)] + suffix
            i += 1
        used.add(clean)
        sheet_names.append(clean)

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
"""
            + "\n".join(
                f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                for i in range(1, len(sheet_names) + 1)
            )
            + """
</Types>""",
        )

        z.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>""",
        )

        workbook_sheets = "\n".join(
            f'<sheet name="{xml_escape(name)}" sheetId="{i}" r:id="rId{i}"/>'
            for i, name in enumerate(sheet_names, start=1)
        )

        z.writestr(
            "xl/workbook.xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets>
{workbook_sheets}
</sheets>
</workbook>""",
        )

        rels = "\n".join(
            f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
            for i in range(1, len(sheet_names) + 1)
        )

        z.writestr(
            "xl/_rels/workbook.xml.rels",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
{rels}
</Relationships>""",
        )

        for i, (_, df) in enumerate(sheets.items(), start=1):
            z.writestr(f"xl/worksheets/sheet{i}.xml", df_to_sheet_xml(df))


def infer_workload(path: Path, df: pd.DataFrame) -> str | None:
    if "workload" in df.columns:
        vals = [str(x) for x in df["workload"].dropna().unique()]
        for v in vals:
            if v in KEEP_WORKLOADS:
                return v

    parts = list(path.parts)
    for p in parts:
        if p in KEEP_WORKLOADS:
            return p
    return None


def infer_method(path: Path, df: pd.DataFrame) -> str | None:
    if "method" in df.columns:
        vals = [str(x) for x in df["method"].dropna().unique()]
        for v in vals:
            if v in KEEP_METHODS:
                return v

    for p in path.parts:
        if p in KEEP_METHODS:
            return p
    return None


def infer_k(path: Path, method: str, df: pd.DataFrame):
    if method == "ar":
        return pd.NA

    if "k" in df.columns:
        vals = pd.to_numeric(df["k"], errors="coerce").dropna().unique()
        if len(vals) > 0:
            return int(vals[0])

    s = str(path)
    m = re.search(r"(?:^|[/_])k(\d+)(?:[_/]|$)", s)
    if m:
        return int(m.group(1))

    return pd.NA


def first_col(df: pd.DataFrame, names: list[str]):
    for n in names:
        if n in df.columns:
            return n
    return None


def normalize_one_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    workload = infer_workload(path, df)
    method = infer_method(path, df)

    if workload not in KEEP_WORKLOADS or method not in KEEP_METHODS:
        return pd.DataFrame()

    k = infer_k(path, method, df)

    tps_col = first_col(df, ["tokens_per_sec", "tps", "throughput_tps"])
    lat_col = first_col(df, ["latency_s", "latency", "elapsed_s", "wall_time_s"])
    out_col = first_col(df, ["n_output_tokens", "output_tokens", "num_output_tokens", "generated_tokens"])
    draft_col = first_col(df, ["draft_tokens", "num_draft_tokens", "spec_draft_tokens"])
    acc_col = first_col(df, ["accepted_tokens_total", "accepted_tokens", "num_accepted_tokens"])
    rej_col = first_col(df, ["rejected_tokens", "num_rejected_tokens", "wasted_draft_tokens"])

    n = len(df)
    out = pd.DataFrame()
    out["workload"] = [workload] * n
    out["method"] = [method] * n
    out["k"] = [k] * n
    out["baseline_config"] = [
        "ar" if method == "ar" else f"{method}_k{k}"
    ] * n
    out["source_summary_path"] = [str(path)] * n

    out["tokens_per_sec"] = pd.to_numeric(df[tps_col], errors="coerce") if tps_col else pd.NA
    out["latency_s"] = pd.to_numeric(df[lat_col], errors="coerce") if lat_col else pd.NA
    out["n_output_tokens"] = pd.to_numeric(df[out_col], errors="coerce") if out_col else pd.NA

    if method == "ar":
        out["draft_tokens"] = 0.0
        out["accepted_tokens"] = 0.0
        out["rejected_tokens"] = 0.0
    else:
        out["draft_tokens"] = pd.to_numeric(df[draft_col], errors="coerce") if draft_col else 0.0
        out["accepted_tokens"] = pd.to_numeric(df[acc_col], errors="coerce") if acc_col else 0.0

        if rej_col:
            out["rejected_tokens"] = pd.to_numeric(df[rej_col], errors="coerce")
        else:
            out["rejected_tokens"] = out["draft_tokens"] - out["accepted_tokens"]

        out["draft_tokens"] = out["draft_tokens"].fillna(0.0)
        out["accepted_tokens"] = out["accepted_tokens"].fillna(0.0)
        out["rejected_tokens"] = out["rejected_tokens"].fillna(
            out["draft_tokens"] - out["accepted_tokens"]
        )

    return out


def add_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    spec = df["method"].ne("ar")
    denom = df["draft_tokens"].replace(0, pd.NA)

    df["token_acceptance_pct"] = pd.NA
    df["wasted_token_pct"] = pd.NA

    df.loc[spec, "token_acceptance_pct"] = (
        100 * df.loc[spec, "accepted_tokens"] / denom.loc[spec]
    )
    df.loc[spec, "wasted_token_pct"] = (
        100 * df.loc[spec, "rejected_tokens"] / denom.loc[spec]
    )

    df.loc[~spec, "token_acceptance_pct"] = 100.0
    df.loc[~spec, "wasted_token_pct"] = 0.0

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default=str(DEFAULT_EVAL_ROOT))
    args = ap.parse_args()

    eval_root = Path(args.eval_root)
    runs_root = eval_root / "fixed_runs"
    out_dir = eval_root / "baseline_ar_excel_only"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not runs_root.exists():
        raise SystemExit(f"fixed_runs folder not found: {runs_root}")

    summary_paths = []
    for p in sorted(runs_root.rglob("summary.csv")):
        s = str(p).lower()
        if "backup" in s or "_backup" in s or "tmp" in s:
            continue
        summary_paths.append(p)

    rows = []
    skipped = []

    for p in summary_paths:
        try:
            d = normalize_one_summary(p)
            if len(d) == 0:
                skipped.append(str(p))
            else:
                rows.append(d)
        except Exception as e:
            skipped.append(f"{p} :: ERROR {e}")

    if not rows:
        raise SystemExit(f"No AR/baseline summary rows found under {runs_root}")

    req = pd.concat(rows, ignore_index=True)

    # Keep exactly finished AR + fixed baselines for the six-workload final6 set.
    req = req[
        req["workload"].isin(KEEP_WORKLOADS)
        & req["method"].isin(KEEP_METHODS)
    ].copy()

    for c in [
        "tokens_per_sec",
        "latency_s",
        "n_output_tokens",
        "draft_tokens",
        "accepted_tokens",
        "rejected_tokens",
    ]:
        req[c] = pd.to_numeric(req[c], errors="coerce")

    by_workload = req.groupby(
        ["workload", "method", "k", "baseline_config"], dropna=False
    ).agg(
        n_requests=("tokens_per_sec", "size"),
        mean_tps=("tokens_per_sec", "mean"),
        median_tps=("tokens_per_sec", "median"),
        mean_latency_s=("latency_s", "mean"),
        median_latency_s=("latency_s", "median"),
        total_output_tokens=("n_output_tokens", "sum"),
        mean_output_tokens=("n_output_tokens", "mean"),
        draft_tokens=("draft_tokens", "sum"),
        accepted_tokens=("accepted_tokens", "sum"),
        rejected_tokens=("rejected_tokens", "sum"),
    ).reset_index()

    by_workload = add_efficiency(by_workload)

    arw = by_workload[by_workload["method"].eq("ar")][
        ["workload", "mean_tps", "mean_latency_s"]
    ].rename(
        columns={
            "mean_tps": "ar_mean_tps_workload",
            "mean_latency_s": "ar_mean_latency_s_workload",
        }
    )

    by_workload = by_workload.merge(arw, on="workload", how="left")
    by_workload["throughput_speedup_vs_ar_workload"] = (
        by_workload["mean_tps"] / by_workload["ar_mean_tps_workload"]
    )
    by_workload["e2e_speedup_vs_ar_workload"] = (
        by_workload["ar_mean_latency_s_workload"] / by_workload["mean_latency_s"]
    )

    overall = req.groupby(
        ["method", "k", "baseline_config"], dropna=False
    ).agg(
        n_requests=("tokens_per_sec", "size"),
        n_workloads=("workload", "nunique"),
        mean_tps=("tokens_per_sec", "mean"),
        median_tps=("tokens_per_sec", "median"),
        mean_latency_s=("latency_s", "mean"),
        median_latency_s=("latency_s", "median"),
        total_output_tokens=("n_output_tokens", "sum"),
        mean_output_tokens=("n_output_tokens", "mean"),
        draft_tokens=("draft_tokens", "sum"),
        accepted_tokens=("accepted_tokens", "sum"),
        rejected_tokens=("rejected_tokens", "sum"),
    ).reset_index()

    overall = add_efficiency(overall)

    ar_rows = overall[overall["method"].eq("ar")]
    if len(ar_rows):
        ar = ar_rows.iloc[0]
        overall["ar_mean_tps_overall"] = ar["mean_tps"]
        overall["ar_mean_latency_s_overall"] = ar["mean_latency_s"]
        overall["throughput_speedup_vs_ar"] = overall["mean_tps"] / ar["mean_tps"]
        overall["e2e_speedup_vs_ar"] = ar["mean_latency_s"] / overall["mean_latency_s"]
    else:
        overall["ar_mean_tps_overall"] = pd.NA
        overall["ar_mean_latency_s_overall"] = pd.NA
        overall["throughput_speedup_vs_ar"] = pd.NA
        overall["e2e_speedup_vs_ar"] = pd.NA

    method_order = {"ar": 0, "ngram_sd": 1, "draft_sd": 2, "eagle3": 3}
    for d in [overall, by_workload, req]:
        d["_m"] = d["method"].map(method_order).fillna(99)
        d["_k"] = pd.to_numeric(d["k"], errors="coerce").fillna(-1)
        sort_cols = ["_m", "_k"]
        if "workload" in d.columns:
            sort_cols = ["workload", "_m", "_k"]
        d.sort_values(sort_cols, inplace=True)
        d.drop(columns=["_m", "_k"], inplace=True)

    inventory = pd.DataFrame(
        [
            {
                "eval_root": str(eval_root),
                "runs_root": str(runs_root),
                "summary_csv_found": len(summary_paths),
                "summary_csv_used": req["source_summary_path"].nunique(),
                "request_rows": len(req),
                "workloads_found": ",".join(sorted(req["workload"].dropna().unique())),
                "methods_found": ",".join(sorted(req["method"].dropna().unique())),
                "skipped_paths": len(skipped),
            }
        ]
    )

    missing_rows = []
    expected = [("ar", pd.NA)] + [
        (m, k) for m in ["ngram_sd", "draft_sd", "eagle3"] for k in [1, 2, 4, 8, 16]
    ]
    for workload in sorted(KEEP_WORKLOADS):
        for method, k in expected:
            if method == "ar":
                exists = len(req[(req["workload"].eq(workload)) & (req["method"].eq("ar"))]) > 0
                kk = ""
            else:
                exists = len(
                    req[
                        (req["workload"].eq(workload))
                        & (req["method"].eq(method))
                        & (pd.to_numeric(req["k"], errors="coerce").eq(k))
                    ]
                ) > 0
                kk = k
            if not exists:
                missing_rows.append({"workload": workload, "method": method, "k": kk})

    missing = pd.DataFrame(missing_rows)
    skipped_df = pd.DataFrame({"skipped_or_error_path": skipped})

    # CSV outputs
    req.to_csv(out_dir / "request_rows_baseline_ar_only.csv", index=False)
    overall.to_csv(out_dir / "overall_baseline_ar_only.csv", index=False)
    by_workload.to_csv(out_dir / "by_workload_baseline_ar_only.csv", index=False)
    inventory.to_csv(out_dir / "inventory.csv", index=False)
    missing.to_csv(out_dir / "missing_expected_runs.csv", index=False)
    skipped_df.to_csv(out_dir / "skipped_paths.csv", index=False)

    xlsx_path = out_dir / "final6_baseline_ar_only.xlsx"
    write_xlsx(
        xlsx_path,
        {
            "inventory": inventory,
            "overall": overall,
            "by_workload": by_workload,
            "missing_expected": missing,
            "request_rows": req,
            "skipped_paths": skipped_df,
        },
    )

    print("WROTE:", xlsx_path)
    print("CSV_DIR:", out_dir)
    print()
    print("=== INVENTORY ===")
    print(inventory.to_string(index=False))
    print()
    print("=== OVERALL ===")
    print(overall.to_string(index=False))
    if len(missing):
        print()
        print("[WARN] Missing expected AR/baseline runs:")
        print(missing.to_string(index=False))


if __name__ == "__main__":
    main()
