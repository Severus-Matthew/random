#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", required=True)
    args = ap.parse_args()

    path = Path(args.runner)
    text = path.read_text()

    if "APEXP_REQUEST_META_PATCH" in text:
        print(f"Already patched: {path}")
        return

    backup = path.with_suffix(path.suffix + ".apexp_request_meta_bak")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"Backup written: {backup}")
    else:
        print(f"Backup already exists: {backup}")

    # Add hashlib import.
    text = text.replace(
        "import argparse, json, time, math, os, random",
        "import argparse, json, time, math, os, random, hashlib",
        1,
    )

    # Capture vLLM request id immediately after generate().
    anchor = "    outputs = llm.generate([prompt], sp, use_tqdm=False)\n"
    insert = (
        "    outputs = llm.generate([prompt], sp, use_tqdm=False)\n"
        "    # APEXP_REQUEST_META_PATCH: vLLM request id used to join block events to prompt summaries.\n"
        "    vllm_request_id = getattr(outputs[0], \"request_id\", None)\n"
    )
    if anchor not in text:
        raise RuntimeError("Could not find llm.generate anchor in run_one().")
    text = text.replace(anchor, insert, 1)

    # Add vllm_request_id to trace return.
    anchor = "    return {\n        # provenance\n"
    insert = (
        "    return {\n"
        "        # provenance\n"
        "        \"vllm_request_id\": vllm_request_id,\n"
    )
    if anchor not in text:
        raise RuntimeError("Could not find return provenance anchor in run_one().")
    text = text.replace(anchor, insert, 1)

    # Add prompt metadata after optional truncation.
    anchor = (
        "            if args.max_prompt_tokens > 0:\n"
        "                base_prompt = truncate_prompt(base_prompt, tokenizer, args.max_prompt_tokens)\n"
    )
    insert = (
        "            if args.max_prompt_tokens > 0:\n"
        "                base_prompt = truncate_prompt(base_prompt, tokenizer, args.max_prompt_tokens)\n"
        "\n"
        "            # APEXP_REQUEST_META_PATCH: stable prompt/request metadata for block-level join.\n"
        "            prompt_id = str(row.get(\"id\", f\"row_{len(summary_rows)}\"))\n"
        "            request_ordinal = len(summary_rows)\n"
        "            prompt_hash = hashlib.sha1(base_prompt.encode(\"utf-8\", errors=\"ignore\")).hexdigest()\n"
        "            prompt_char_len = len(base_prompt)\n"
        "            prompt_token_len = len(tokenizer.encode(base_prompt, add_special_tokens=False))\n"
    )
    if anchor not in text:
        raise RuntimeError("Could not find prompt truncation anchor in main loop.")
    text = text.replace(anchor, insert, 1)

    # Replace id row and add metadata columns in summary rec.
    old = '                "id":             row.get("id", f"row_{len(summary_rows)}"),\n'
    new = (
        '                "id":             prompt_id,\n'
        '                "prompt_id":      prompt_id,\n'
        '                "request_ordinal": request_ordinal,\n'
        '                "prompt_hash":    prompt_hash,\n'
        '                "prompt_char_len": prompt_char_len,\n'
        '                "prompt_token_len": prompt_token_len,\n'
        '                "vllm_request_id": trace.get("vllm_request_id", ""),\n'
    )
    if old not in text:
        raise RuntimeError("Could not find rec['id'] line to replace.")
    text = text.replace(old, new, 1)

    path.write_text(text)
    print(f"Patched runner: {path}")
    print(f"Restore with: cp {backup} {path}")


if __name__ == "__main__":
    main()
