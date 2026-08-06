#!/usr/bin/env python3
"""
Patch vLLM 0.21.0 scheduler.py to log block-level speculative decoding events.

Patch point:
  vllm/v1/core/sched/scheduler.py

Anchor:
  num_accepted = len(generated_token_ids) - 1
  num_rejected = num_draft_tokens - num_accepted

The patch calls:
  apexp.trace.block_logger.get_block_logger().log_scheduler_locals(locals())

It is deliberately inserted at the scheduler level because this is CPU-side,
request-aware, and already has num_draft_tokens / num_accepted.
"""

from __future__ import annotations

import inspect
import os
import shutil
from pathlib import Path

import vllm


PATCH_MARKER_BEGIN = "# BEGIN APEXP_BLOCK_TRACE_PATCH"
PATCH_MARKER_END = "# END APEXP_BLOCK_TRACE_PATCH"

INSERT_AFTER = "num_rejected = num_draft_tokens - num_accepted"

PATCH = f"""
                {PATCH_MARKER_BEGIN}
                try:
                    from apexp.trace.block_logger import get_block_logger
                    get_block_logger().log_scheduler_locals(locals())
                except Exception:
                    # Never let tracing break vLLM unless the logger itself is
                    # configured strict and re-raises internally.
                    pass
                {PATCH_MARKER_END}
"""


def main() -> None:
    vllm_root = Path(os.path.dirname(inspect.getfile(vllm)))
    scheduler_path = vllm_root / "v1" / "core" / "sched" / "scheduler.py"

    if not scheduler_path.exists():
        raise FileNotFoundError(f"Could not find scheduler.py at {scheduler_path}")

    text = scheduler_path.read_text()

    if PATCH_MARKER_BEGIN in text:
        print(f"Already patched: {scheduler_path}")
        return

    if INSERT_AFTER not in text:
        raise RuntimeError(
            f"Anchor not found in {scheduler_path}: {INSERT_AFTER!r}. "
            "Print lines around num_accepted and patch manually."
        )

    backup_path = scheduler_path.with_suffix(".py.apexp_bak")
    if not backup_path.exists():
        shutil.copy2(scheduler_path, backup_path)
        print(f"Backup written: {backup_path}")
    else:
        print(f"Backup already exists: {backup_path}")

    new_text = text.replace(INSERT_AFTER, INSERT_AFTER + PATCH, 1)
    scheduler_path.write_text(new_text)

    print(f"Patched: {scheduler_path}")
    print()
    print("Verify with:")
    print(f"  grep -n \"APEXP_BLOCK_TRACE_PATCH\" {scheduler_path}")
    print()
    print("Restore with:")
    print(f"  cp {backup_path} {scheduler_path}")


if __name__ == "__main__":
    main()
