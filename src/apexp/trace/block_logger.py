#!/usr/bin/env python3
"""
APEX-P block-level speculative decoding logger.

This logger is designed to be called from vLLM's scheduler after speculative
verification has decided how many draft tokens were accepted.

It is intentionally conservative:
  - disabled unless APEXP_BLOCK_TRACE=1
  - writes JSONL append-only
  - validates accepted_len / first_rejection invariants
  - never decodes text in the hot path
  - never crashes vLLM unless APEXP_TRACE_STRICT=1
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _safe_list_int(x: Any, max_len: Optional[int] = None) -> Optional[List[int]]:
    if x is None:
        return None
    try:
        # torch tensor / numpy array
        if hasattr(x, "detach"):
            x = x.detach().cpu().tolist()
        elif hasattr(x, "tolist"):
            x = x.tolist()

        if isinstance(x, tuple):
            x = list(x)

        if not isinstance(x, list):
            return None

        out = []
        for v in x:
            if isinstance(v, list):
                # Avoid nested accidental huge structures in this helper.
                return None
            out.append(int(v))
        if max_len is not None:
            out = out[:max_len]
        return out
    except Exception:
        return None


def _json_default(x: Any) -> Any:
    try:
        if hasattr(x, "detach"):
            return x.detach().cpu().tolist()
        if hasattr(x, "tolist"):
            return x.tolist()
    except Exception:
        pass
    return str(x)


class ApexBlockLogger:
    def __init__(self) -> None:
        self.enabled = _env_bool("APEXP_BLOCK_TRACE", False)
        self.strict = _env_bool("APEXP_TRACE_STRICT", False)
        self.qualitative = _env_bool("APEXP_TRACE_QUALITATIVE", False)

        self.run_id = os.environ.get("APEXP_RUN_ID", "unknown_run")
        self.workload = os.environ.get("APEXP_WORKLOAD", "unknown_workload")
        self.method = os.environ.get("APEXP_METHOD", "unknown_method")
        self.k_requested = _safe_int(os.environ.get("APEXP_K"), None)
        self.temperature = os.environ.get("APEXP_TEMPERATURE", None)
        self.seed = os.environ.get("APEXP_SEED", None)

        out_dir = Path(os.environ.get("APEXP_TRACE_DIR", "results/apexp_block"))
        out_dir.mkdir(parents=True, exist_ok=True)

        self.block_path = out_dir / "block_events.jsonl"
        self.invalid_path = out_dir / "invalid_events.jsonl"
        self.meta_path = out_dir / "run_metadata.json"

        self._lock = threading.Lock()
        self._block_fh = None
        self._invalid_fh = None
        self._counter_by_req: Dict[str, int] = {}
        self._active_trace_dir = str(out_dir)

        if self.enabled:
            self._block_fh = open(self.block_path, "a", buffering=1)
            self._invalid_fh = open(self.invalid_path, "a", buffering=1)
            self._write_metadata()
            atexit.register(self.close)

    def _read_hot_control(self) -> Dict[str, Any]:
        path = os.environ.get("APEXP_REQUEST_CONTROL_FILE", "")
        if not path:
            return {}
        try:
            p = Path(path)
            if not p.exists():
                return {}
            obj = json.loads(p.read_text())
            if isinstance(obj, dict):
                return obj
        except Exception:
            if self.strict:
                raise
        return {}

    def _switch_trace_dir_from_control(self, control: Dict[str, Any]) -> None:
        if not control:
            return

        trace_dir = control.get("trace_dir")
        if not trace_dir:
            return

        trace_dir = str(trace_dir)
        if trace_dir == getattr(self, "_active_trace_dir", None):
            return

        new_dir = Path(trace_dir)
        new_dir.mkdir(parents=True, exist_ok=True)

        with self._lock:
            if self._block_fh is not None:
                self._block_fh.close()
                self._block_fh = None
            if self._invalid_fh is not None:
                self._invalid_fh.close()
                self._invalid_fh = None

            self._active_trace_dir = trace_dir
            self.block_path = new_dir / "block_events.jsonl"
            self.invalid_path = new_dir / "invalid_events.jsonl"
            self.meta_path = new_dir / "run_metadata.json"

            self.run_id = str(control.get("task_id") or control.get("control_id") or self.run_id)
            self.workload = str(control.get("workload") or self.workload)
            self.method = str(control.get("method") or self.method)
            self.k_requested = _safe_int(control.get("slow_k"), self.k_requested)
            self.temperature = control.get("temperature", self.temperature)
            self.seed = control.get("seed", self.seed)

            # Request ids can repeat across hot-worker requests, so reset block counter
            # whenever the output trace directory changes.
            self._counter_by_req = {}

            if self.enabled:
                self._block_fh = open(self.block_path, "a", buffering=1)
                self._invalid_fh = open(self.invalid_path, "a", buffering=1)
                self._write_metadata()

    def _write_metadata(self) -> None:
        metadata = {
            "run_id": self.run_id,
            "workload": self.workload,
            "method": self.method,
            "k_requested": self.k_requested,
            "temperature": self.temperature,
            "seed": self.seed,
            "trace_enabled": self.enabled,
            "strict": self.strict,
            "qualitative": self.qualitative,
            "created_time": time.time(),
            "env": {
                k: os.environ.get(k)
                for k in sorted(os.environ)
                if k.startswith("APEXP_")
            },
        }
        try:
            self.meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        except Exception:
            if self.strict:
                raise

    def close(self) -> None:
        with self._lock:
            if self._block_fh is not None:
                self._block_fh.close()
                self._block_fh = None
            if self._invalid_fh is not None:
                self._invalid_fh.close()
                self._invalid_fh = None

    def _next_block_index(self, req_id: str) -> int:
        cur = self._counter_by_req.get(req_id, 0)
        self._counter_by_req[req_id] = cur + 1
        return cur

    def _write_jsonl(self, fh, obj: Dict[str, Any]) -> None:
        if fh is None:
            return
        fh.write(json.dumps(obj, default=_json_default, sort_keys=True) + "\n")

    def _invalid(self, event: Dict[str, Any], reason: str, extra: Optional[Dict[str, Any]] = None) -> None:
        bad = {
            "reason": reason,
            "event": event,
            "extra": extra or {},
            "time": time.time(),
        }
        with self._lock:
            self._write_jsonl(self._invalid_fh, bad)
        if self.strict:
            raise RuntimeError(f"Invalid APEX block event: {reason}")

    def validate_event(self, event: Dict[str, Any]) -> bool:
        k_actual = event.get("k_actual")
        accepted_len = event.get("accepted_len")
        first_rejection = event.get("first_rejection")
        accepted_mask = event.get("accepted_mask")

        if k_actual is None or accepted_len is None or first_rejection is None:
            self._invalid(event, "missing required k_actual/accepted_len/first_rejection")
            return False

        if k_actual < 0:
            self._invalid(event, "negative k_actual")
            return False

        if accepted_len < 0:
            self._invalid(event, "negative accepted_len")
            return False

        if accepted_len > k_actual:
            self._invalid(event, "accepted_len greater than k_actual")
            return False

        if first_rejection < 0 or first_rejection > k_actual:
            self._invalid(event, "first_rejection outside [0, k_actual]")
            return False

        if accepted_len == k_actual and first_rejection != k_actual:
            self._invalid(event, "full acceptance must have first_rejection == k_actual")
            return False

        if accepted_len < k_actual and first_rejection != accepted_len:
            self._invalid(event, "partial acceptance must have first_rejection == accepted_len")
            return False

        if accepted_mask is not None:
            if len(accepted_mask) != k_actual:
                self._invalid(event, "accepted_mask length != k_actual")
                return False
            expected_mask = [True] * accepted_len + [False] * (k_actual - accepted_len)
            if accepted_mask != expected_mask:
                self._invalid(
                    event,
                    "accepted_mask inconsistent with accepted_len",
                    extra={"expected_mask": expected_mask},
                )
                return False

        return True

    def log_scheduler_locals(self, loc: Dict[str, Any]) -> None:
        """
        Log from vLLM scheduler locals.

        We use locals() deliberately so the patch is robust to small changes in
        local variable names. This function extracts what it can.
        """
        if not self.enabled:
            return

        try:
            control = self._read_hot_control()
            self._switch_trace_dir_from_control(control)

            req_id = str(
                loc.get("req_id")
                or loc.get("request_id")
                or loc.get("seq_id")
                or "unknown_req"
            )

            num_draft_tokens = _safe_int(loc.get("num_draft_tokens"), None)
            num_accepted = _safe_int(loc.get("num_accepted"), None)
            num_rejected = _safe_int(loc.get("num_rejected"), None)

            generated_token_ids = loc.get("generated_token_ids")
            generated_token_ids_list = _safe_list_int(generated_token_ids, max_len=512)

            if num_draft_tokens is None:
                self._invalid(
                    {"req_id": req_id},
                    "num_draft_tokens missing from scheduler locals",
                    extra={"available_keys": sorted(map(str, loc.keys()))},
                )
                return

            if num_accepted is None:
                if generated_token_ids_list is not None:
                    # vLLM convention: generated_token_ids = accepted draft tokens
                    # plus one correction/bonus token.
                    num_accepted = max(len(generated_token_ids_list) - 1, 0)
                else:
                    self._invalid(
                        {"req_id": req_id, "num_draft_tokens": num_draft_tokens},
                        "num_accepted missing and generated_token_ids unavailable",
                        extra={"available_keys": sorted(map(str, loc.keys()))},
                    )
                    return

            if num_rejected is None:
                num_rejected = num_draft_tokens - num_accepted

            k_actual = num_draft_tokens
            accepted_len = num_accepted
            first_rejection = accepted_len if accepted_len < k_actual else k_actual
            full_accept = accepted_len == k_actual
            accepted_mask = [True] * accepted_len + [False] * max(k_actual - accepted_len, 0)

            # Best-effort extraction of draft ids if scheduler locals contain them.
            draft_token_ids = None
            for key in [
                "draft_token_ids",
                "scheduled_spec_decode_tokens",
                "spec_decode_tokens",
                "draft_tokens",
            ]:
                draft_token_ids = _safe_list_int(loc.get(key), max_len=512)
                if draft_token_ids is not None:
                    break

            # Best effort: request object may carry scheduled draft tokens.
            if draft_token_ids is None:
                for obj_key in ["request", "req"]:
                    obj = loc.get(obj_key)
                    if obj is None:
                        continue
                    for attr in [
                        "draft_token_ids",
                        "scheduled_spec_decode_tokens",
                        "spec_decode_tokens",
                    ]:
                        if hasattr(obj, attr):
                            draft_token_ids = _safe_list_int(getattr(obj, attr), max_len=512)
                            if draft_token_ids is not None:
                                break
                    if draft_token_ids is not None:
                        break

            event = {
                "schema_version": 1,
                "event_type": "spec_decode_block",
                "time": time.time(),
                "run_id": self.run_id,
                "workload": self.workload,
                "method": self.method,
                "temperature": self.temperature,
                "seed": self.seed,

                "req_id": req_id,
                "block_index": self._next_block_index(req_id),

                "k_requested": self.k_requested,
                "k_actual": k_actual,
                "accepted_len": accepted_len,
                "first_rejection": first_rejection,
                "num_rejected": num_rejected,
                "full_accept": full_accept,
                "accepted_mask": accepted_mask,

                "generated_token_ids": generated_token_ids_list,
                "draft_token_ids": draft_token_ids,

                # Hot-worker request metadata from request_control.json.
                "control_id": control.get("control_id"),
                "task_id": control.get("task_id"),
                "pool": control.get("pool"),
                "mode": control.get("mode"),
                "global_index": control.get("global_index"),
                "source_file": control.get("source_file"),
                "source_index": control.get("source_index"),
                "prompt_id": control.get("task_id"),
                "prompt_hash": control.get("prompt_hash"),
                "slow_k": control.get("slow_k"),
                "router_source": control.get("router_source"),
                "router_score": control.get("router_score"),

                "state": {},
                "latency": {},
            }

            if self.validate_event(event):
                # BEGIN APEXP_ONLINE_FAST_OBSERVE_PATCH
                try:
                    if os.environ.get("APEXP_ONLINE_FAST", "0") == "1":
                        from apexp.runtime.online_fast_controller import observe_block_event
                        observe_block_event(event)
                except Exception:
                    if getattr(self, "strict", False):
                        raise
                # END APEXP_ONLINE_FAST_OBSERVE_PATCH

                with self._lock:
                    self._write_jsonl(self._block_fh, event)

        except Exception as e:
            err = {
                "reason": "logger_exception",
                "error": repr(e),
                "traceback": traceback.format_exc(),
                "time": time.time(),
            }
            with self._lock:
                self._write_jsonl(self._invalid_fh, err)
            if self.strict:
                raise


_GLOBAL_LOGGER: Optional[ApexBlockLogger] = None


def get_block_logger() -> ApexBlockLogger:
    global _GLOBAL_LOGGER
    if _GLOBAL_LOGGER is None:
        _GLOBAL_LOGGER = ApexBlockLogger()
    return _GLOBAL_LOGGER
