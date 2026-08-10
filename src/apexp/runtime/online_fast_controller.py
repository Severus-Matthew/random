from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict

import numpy as np

try:
    from src.apexp.runtime.learned_fast_survival_policy_fast import FastLearnedSurvivalPolicy
except Exception:
    FastLearnedSurvivalPolicy = None


@dataclass
class RequestState:
    req_id: str
    next_k: int
    current_k: int
    n_blocks: int = 0
    k_hist: Deque[int] = field(default_factory=lambda: deque(maxlen=32))
    accepted_hist: Deque[int] = field(default_factory=lambda: deque(maxlen=32))
    rejected_hist: Deque[int] = field(default_factory=lambda: deque(maxlen=32))
    full_accept_hist: Deque[int] = field(default_factory=lambda: deque(maxlen=32))


class OnlineFastController:
    def __init__(self):
        import inspect
        self._controller_file = inspect.getfile(OnlineFastController)
        self.enabled = os.environ.get("APEXP_ONLINE_FAST", "0") == "1"
        self.default_k = int(os.environ.get("APEXP_DEFAULT_K", "16"))
        self.initial_k = int(os.environ.get("APEXP_INITIAL_K", str(self.default_k)))
        self.min_k = int(os.environ.get("APEXP_MIN_K", "1"))
        self.max_k = int(os.environ.get("APEXP_MAX_K", str(self.default_k)))
        self.candidate_ks = self._candidate_ks()

        self.window = int(os.environ.get("APEXP_FAST_WINDOW", "4"))
        self.min_obs = int(os.environ.get("APEXP_FAST_MIN_OBS", str(self.window)))
        self.down_ratio = float(os.environ.get("APEXP_FAST_DOWN_RATIO", "0.45"))
        self.up_ratio = float(os.environ.get("APEXP_FAST_UP_RATIO", "0.85"))
        self.up_full_rate = float(os.environ.get("APEXP_FAST_UP_FULL_RATE", "0.75"))

        self.control_file = os.environ.get("APEXP_REQUEST_CONTROL_FILE", "")
        self.last_control_id = None

        self.states: Dict[str, RequestState] = {}
        self.global_next_k = self._clip(self.initial_k)

        self.learned_model_dir = os.environ.get("APEXP_LEARNED_FAST_MODEL_DIR", "")
        self.learned_policy = None
        self.learned_policy_error = ""

        if self.learned_model_dir and FastLearnedSurvivalPolicy is not None:
            try:
                learned_device = os.environ.get("APEXP_LEARNED_FAST_DEVICE", "cpu")
                self.learned_policy = FastLearnedSurvivalPolicy(self.learned_model_dir, device=learned_device)
            except Exception as e:
                self.learned_policy = None
                self.learned_policy_error = repr(e)

        self.trace_path = os.environ.get("APEXP_ONLINE_FAST_TRACE", "")
        self.trace_f = None
        if self.trace_path:
            Path(self.trace_path).parent.mkdir(parents=True, exist_ok=True)
            self.trace_f = open(self.trace_path, "a", buffering=1)

        if self.learned_model_dir:
            self._trace({
                "time": time.time(),
                "type": "learned_policy_init",
                "model_dir": self.learned_model_dir,
                "loaded": self.learned_policy is not None,
                "error": self.learned_policy_error,
                "controller_file": getattr(self, "_controller_file", ""),
            })

    def _candidate_ks(self):
        vals = []
        for x in os.environ.get("APEXP_FAST_CANDIDATE_KS", "1,2,4,8,16").split(","):
            x = x.strip()
            if x:
                vals.append(int(x))
        vals = sorted(set(k for k in vals if self.min_k <= k <= self.max_k))
        return vals or [self.default_k]

    def _clip(self, k: int, upper: int | None = None) -> int:
        upper = self.max_k if upper is None else min(self.max_k, int(upper))
        k = max(self.min_k, min(upper, int(k)))
        return min(self.candidate_ks, key=lambda x: abs(x - k))

    def _read_control(self) -> dict:
        if not self.control_file:
            return {}

        try:
            p = Path(self.control_file)
            if not p.exists():
                return {}
            c = json.loads(p.read_text())
        except Exception:
            return {}

        control_id = c.get("control_id") or c.get("task_id")
        if control_id and control_id != self.last_control_id:
            self.last_control_id = control_id
            self.default_k = int(c.get("default_k", self.default_k))
            self.initial_k = int(c.get("initial_k", c.get("slow_k", self.default_k)))
            self.max_k = int(c.get("max_k", self.max_k))
            self.min_k = int(c.get("min_k", self.min_k))
            if "candidate_ks" in c:
                self.candidate_ks = sorted(set(int(x) for x in c["candidate_ks"]))
            else:
                self.candidate_ks = self._candidate_ks()

            self.states.clear()
            self.global_next_k = self._clip(self.initial_k, self.default_k)

            self._trace({
                "time": time.time(),
                "type": "control_file_new_request",
                "control_id": control_id,
                "mode": c.get("mode", "dynamic"),
                "default_k": self.default_k,
                "initial_k": self.initial_k,
                "global_next_k": self.global_next_k,
            })

        return c

    def _state(self, req_id: str, default_k: int) -> RequestState:
        req_id = str(req_id)
        if req_id not in self.states:
            init = self._clip(self.global_next_k, default_k)
            self.states[req_id] = RequestState(req_id=req_id, next_k=init, current_k=init)
        return self.states[req_id]

    def choose_active_k(self, req_id: str = "unknown_req", default_k: int | None = None) -> int:
        if default_k is None:
            default_k = self.default_k

        if not self.enabled:
            return int(default_k)

        c = self._read_control()
        mode = str(c.get("mode", "dynamic"))
        request_default_k = int(c.get("default_k", default_k))
        request_default_k = min(request_default_k, int(default_k))

        if os.environ.get("APEXP_FORCE_ACTIVE_K"):
            k = int(os.environ["APEXP_FORCE_ACTIVE_K"])
        elif c.get("force_k") is not None and str(c.get("force_k")) != "":
            k = int(c["force_k"])
        elif mode == "fixed":
            k = int(c.get("slow_k", c.get("initial_k", request_default_k)))
        else:
            st = self._state(str(req_id), request_default_k)
            k = st.next_k

        k = self._clip(k, request_default_k)

        self._trace({
            "time": time.time(),
            "type": "choose_active_k_for_scheduler",
            "req_id": str(req_id),
            "mode": mode,
            "default_k": int(request_default_k),
            "active_k": int(k),
            "control_id": c.get("control_id", None),
            "learned_model_dir": self.learned_model_dir,
            "learned_policy_loaded": self.learned_policy is not None,
            "learned_policy_error": self.learned_policy_error,
        })

        return int(k)

    def observe_block_event(self, event: dict):
        if not self.enabled:
            return

        c = self._read_control()
        mode = str(c.get("mode", "dynamic"))

        # Fixed mode is slow-only. We still observe/log, but do not adapt.
        req_id = str(event.get("req_id", "unknown_req"))
        default_k = int(c.get("default_k", self.default_k))
        st = self._state(req_id, default_k)

        k_actual = int(event.get("k_actual", event.get("num_draft_tokens", st.current_k)) or st.current_k)
        accepted = int(event.get("accepted_len", 0) or 0)
        rejected = int(event.get("num_rejected", max(0, k_actual - accepted)) or 0)
        full = bool(event.get("full_accept", accepted >= k_actual))

        st.current_k = k_actual
        st.n_blocks += 1
        st.k_hist.append(k_actual)
        st.accepted_hist.append(accepted)
        st.rejected_hist.append(rejected)
        st.full_accept_hist.append(1 if full else 0)

        old_next = st.next_k
        policy_info = {}
        if mode == "dynamic":
            st.next_k, policy_info = self._policy(st, c)
            self.global_next_k = st.next_k
        else:
            st.next_k = self._clip(int(c.get("force_k", c.get("slow_k", default_k))), default_k)
            self.global_next_k = st.next_k

        stats = self._window_stats(st)

        self._trace({
            "time": time.time(),
            "type": "observe_block_after_verify_update_future_k",
            "req_id": req_id,
            "mode": mode,
            "control_id": c.get("control_id", None),
            "block": st.n_blocks,
            "k_actual": int(k_actual),
            "accepted_len": int(accepted),
            "num_rejected": int(rejected),
            "full_accept": full,
            "window_n": int(stats["n"]),
            "window_avg_accept_ratio": float(stats["avg_accept_ratio"]),
            "window_full_rate": float(stats["full_rate"]),
            "window_zero_rate": float(stats["zero_rate"]),
            "old_next_k": int(old_next),
            "next_k_for_future_unscheduled_block": int(st.next_k),
            "learned_policy_used": bool(policy_info.get("learned_policy_used", False)),
            "policy_runtime_ms": policy_info.get("policy_runtime_ms", None),
            "candidate_scores": policy_info.get("candidate_scores", None),
            "policy_error": policy_info.get("policy_error", ""),
        })

    def _window_stats(self, st: RequestState) -> dict:
        ks = list(st.k_hist)[-self.window:]
        acc = list(st.accepted_hist)[-self.window:]
        full = list(st.full_accept_hist)[-self.window:]
        n = min(len(ks), len(acc))
        if n == 0:
            return {"n": 0, "avg_accept_ratio": 1.0, "full_rate": 0.0, "zero_rate": 0.0}

        ks = ks[-n:]
        acc = acc[-n:]
        full = full[-n:]

        ratios = [float(a) / max(float(k), 1.0) for a, k in zip(acc, ks)]
        return {
            "n": n,
            "avg_accept_ratio": float(np.mean(ratios)),
            "full_rate": float(np.mean(full)) if full else 0.0,
            "zero_rate": float(np.mean([a == 0 for a in acc])),
        }

    def _policy(self, st: RequestState, control: dict | None = None):
        control = control or {}
        cur = int(st.current_k or self.initial_k)
        stats = self._window_stats(st)

        # Learned survival-utility policy. Fallback to heuristic if unavailable/failing.
        if self.learned_policy is not None and stats["n"] >= self.min_obs:
            try:
                res = self.learned_policy.choose_k(
                    st=st,
                    control=control,
                    candidate_ks=list(self.candidate_ks),
                    max_k=int(control.get("default_k", self.default_k)),
                    trace_scores=bool(self.learned_trace_scores),
                )
                k = self._clip(int(res["chosen_k"]), int(control.get("default_k", self.default_k)))
                return k, {
                    "learned_policy_used": True,
                    "policy_runtime_ms": res.get("policy_runtime_ms", None),
                    "candidate_scores": res.get("candidate_scores", None),
                    "policy_error": "",
                }
            except Exception as e:
                return cur, {
                    "learned_policy_used": False,
                    "policy_runtime_ms": None,
                    "candidate_scores": None,
                    "policy_error": repr(e),
                }

        if stats["n"] < self.min_obs:
            return cur, {"learned_policy_used": False, "policy_error": "min_obs_not_reached"}

        avg_ratio = stats["avg_accept_ratio"]
        full_rate = stats["full_rate"]
        zero_rate = stats["zero_rate"]

        if cur <= 1:
            if avg_ratio >= 0.90 and full_rate >= 0.75:
                return self._higher(cur), {"learned_policy_used": False, "policy_error": "heuristic_fallback"}
            return cur, {"learned_policy_used": False, "policy_error": "heuristic_fallback"}

        if cur == 2:
            if avg_ratio >= 0.85 and full_rate >= 0.50:
                return self._higher(cur), {"learned_policy_used": False, "policy_error": "heuristic_fallback"}
            if zero_rate >= 0.75 and avg_ratio <= 0.25:
                return self._lower(cur), {"learned_policy_used": False, "policy_error": "heuristic_fallback"}
            return cur, {"learned_policy_used": False, "policy_error": "heuristic_fallback"}

        if avg_ratio <= self.down_ratio or zero_rate >= 0.50:
            return self._lower(cur), {"learned_policy_used": False, "policy_error": "heuristic_fallback"}

        if avg_ratio >= self.up_ratio and full_rate >= self.up_full_rate:
            return self._higher(cur), {"learned_policy_used": False, "policy_error": "heuristic_fallback"}

        return cur, {"learned_policy_used": False, "policy_error": "heuristic_fallback"}

    def _lower(self, cur: int) -> int:
        lows = [k for k in self.candidate_ks if k < cur]
        return max(lows) if lows else min(self.candidate_ks)

    def _higher(self, cur: int) -> int:
        highs = [k for k in self.candidate_ks if k > cur]
        return min(highs) if highs else max(self.candidate_ks)

    def _trace(self, obj: dict):
        if self.trace_f is not None:
            self.trace_f.write(json.dumps(obj, sort_keys=True) + "\n")


_CONTROLLER = OnlineFastController()


def choose_active_k(req_id: str = "unknown_req", default_k: int | None = None) -> int:
    return _CONTROLLER.choose_active_k(req_id=req_id, default_k=default_k)


def observe_block_event(event: dict) -> None:
    _CONTROLLER.observe_block_event(event)
