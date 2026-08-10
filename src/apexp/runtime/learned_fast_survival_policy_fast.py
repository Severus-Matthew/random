from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.apexp.week2.neural_controller_lib import (
    MAX_K,
    PROMPT_EMB_COLS,
    FeaturePreprocessor,
)
from src.apexp.week2.prompt_embedder import PromptEmbedder
from src.apexp.week2.train_survival_utility_controller_v4 import SurvivalUtilityNet


def _safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def _safe_int(x, default=0) -> int:
    try:
        if x is None:
            return int(default)
        return int(float(x))
    except Exception:
        return int(default)


def _mean(xs, default=0.0) -> float:
    return float(np.mean(xs)) if xs else float(default)


def _std(xs, default=0.0) -> float:
    return float(np.std(xs)) if xs else float(default)


@torch.no_grad()
def _expected_accepted_len(hazard_logits: torch.Tensor, k_values: torch.Tensor) -> torch.Tensor:
    hazard = torch.sigmoid(hazard_logits)
    survival_token = torch.cumprod(1.0 - hazard, dim=1)
    bsz = hazard_logits.shape[0]
    device = hazard_logits.device
    pos = torch.arange(MAX_K, device=device).unsqueeze(0).expand(bsz, MAX_K)
    mask = pos < k_values.clamp(1, MAX_K).unsqueeze(1)
    return (survival_token * mask.float()).sum(dim=1)


class FastLearnedSurvivalPolicy:
    """
    True hot-path learned fast controller.

    No Pandas.
    No DataFrame.
    No per-block prompt embedding.
    No per-block sklearn-style preprocessing.
    """

    def __init__(self, model_dir: str | Path, device: str = "cpu"):
        self.model_dir = Path(model_dir)
        self.metadata = json.loads((self.model_dir / "metadata.json").read_text())

        self.pre: FeaturePreprocessor = FeaturePreprocessor.load(self.model_dir / "preprocessor.pkl")
        self.embedder: PromptEmbedder = PromptEmbedder.load(self.model_dir / "prompt_embedder.pkl")

        self.device = device
        self.model = SurvivalUtilityNet(
            num_dim=len(self.pre.num_cols),
            cat_cardinalities=self.pre.cat_cardinalities(),
            hidden=int(self.metadata.get("hidden", 512)),
            dropout=float(self.metadata.get("dropout", 0.10)),
        )

        state = torch.load(self.model_dir / "model.pt", map_location="cpu")
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self.cat_cols = list(self.pre.cat_cols)
        self.num_cols = list(self.pre.num_cols)
        self.vocab = self.pre.vocab
        self.num_stats = self.pre.num_stats

        self._num_mean = np.array(
            [float(self.num_stats[c]["mean"]) for c in self.num_cols],
            dtype=np.float32,
        )
        self._num_std = np.array(
            [max(float(self.num_stats[c]["std"]), 1e-6) for c in self.num_cols],
            dtype=np.float32,
        )

        self._cat_unknown = {
            c: int(self.vocab[c].get("<UNK>", 0))
            for c in self.cat_cols
        }

        self._request_cache: dict[str, dict[str, Any]] = {}

        self.score_utility_weight = float(self.metadata.get("online_score_utility_weight", 1.0))
        self.score_accept_weight = float(self.metadata.get("online_score_accept_weight", 0.10))
        self.score_tps_weight = float(self.metadata.get("online_score_tps_weight", 0.05))

    def clear_cache(self):
        self._request_cache.clear()

    def _cache_key(self, control: dict[str, Any]) -> str:
        return str(
            control.get("control_id")
            or control.get("task_id")
            or control.get("prompt_hash")
            or control.get("req_id")
            or "global"
        )

    def _cat_id(self, col: str, val: Any) -> int:
        mp = self.vocab.get(col, {})
        return int(mp.get(str(val) if val is not None else "NA", self._cat_unknown.get(col, 0)))

    def _build_request_cache(self, control: dict[str, Any]) -> dict[str, Any]:
        key = self._cache_key(control)
        cached = self._request_cache.get(key)
        if cached is not None:
            return cached

        prompt_text = str(control.get("prompt_text", "") or "")
        try:
            emb = self.embedder.transform([prompt_text])[0].astype(np.float32)
        except Exception:
            emb = np.zeros(len(PROMPT_EMB_COLS), dtype=np.float32)

        prompt_emb = {
            c: float(emb[i]) if i < len(emb) else 0.0
            for i, c in enumerate(PROMPT_EMB_COLS)
        }

        base_cat_vals = {
            "method": str(control.get("method", "ngram_sd") or "ngram_sd"),
            "draft_model": str(control.get("draft_model", "NA") or "NA"),
            "eagle3_model": str(control.get("eagle3_model", "NA") or "NA"),
            "ngram_lookup_min": str(control.get("ngram_lookup_min", "1") or "1"),
            "ngram_lookup_max": str(control.get("ngram_lookup_max", "4") or "4"),
        }

        cat_ids = np.array(
            [self._cat_id(c, base_cat_vals.get(c, "NA")) for c in self.cat_cols],
            dtype=np.int64,
        )

        base_num = {
            "temperature": _safe_float(control.get("temperature", 0.0), 0.0),
            "prompt_token_len": _safe_float(control.get("prompt_token_len", 0.0), 0.0),
            "prefix_entropy_mean": _safe_float(control.get("prefix_entropy_mean", 0.0), 0.0),
            "prefix_entropy_std": _safe_float(control.get("prefix_entropy_std", 0.0), 0.0),
        }

        for w in [16, 32, 64, 128, 256]:
            base_num[f"prefix_entropy_mean_w{w}"] = _safe_float(
                control.get(f"prefix_entropy_mean_w{w}", base_num["prefix_entropy_mean"]),
                base_num["prefix_entropy_mean"],
            )
            base_num[f"prefix_entropy_std_w{w}"] = _safe_float(
                control.get(f"prefix_entropy_std_w{w}", 0.0),
                0.0,
            )
            base_num[f"prefix_repetition_w{w}"] = _safe_float(
                control.get(f"prefix_repetition_w{w}", 0.0),
                0.0,
            )

        base_num.update(prompt_emb)

        cached = {
            "key": key,
            "cat_ids": cat_ids,
            "base_num": base_num,
            "prompt_text_len": len(prompt_text),
        }
        self._request_cache[key] = cached
        return cached

    def _history_features(self, st: Any) -> dict[str, float]:
        accepted = [float(x) for x in list(getattr(st, "accepted_hist", []))]
        rejected = [float(x) for x in list(getattr(st, "rejected_hist", []))]
        full = [float(x) for x in list(getattr(st, "full_accept_hist", []))]
        k_hist = [float(x) for x in list(getattr(st, "k_hist", []))]

        first_rej = []
        for a, k in zip(accepted, k_hist):
            first_rej.append(k if a >= k else a)

        out = {
            "hist_n_blocks": float(len(accepted)),
            "hist_mean_accepted_len": _mean(accepted),
            "hist_std_accepted_len": _std(accepted),
            "hist_mean_first_rejection": _mean(first_rej),
            "hist_full_accept_rate": _mean(full),
            "hist_reject_rate": 1.0 - _mean(full) if full else 0.0,
            "hist_mean_num_rejected": _mean(rejected),
            "prefix_pos": float(len(accepted)),
            "prefix_frac": 0.0,
        }

        for w in [1, 2, 4, 8, 16]:
            out[f"hist_mean_accepted_len_last{w}"] = _mean(accepted[-w:])
            out[f"hist_full_accept_rate_last{w}"] = _mean(full[-w:])
            out[f"hist_mean_first_rejection_last{w}"] = _mean(first_rej[-w:])
            out[f"hist_mean_num_rejected_last{w}"] = _mean(rejected[-w:])

        return out

    def _make_tensors(
        self,
        st: Any,
        control: dict[str, Any],
        candidate_ks: list[int],
        max_k: int,
    ):
        cached = self._build_request_cache(control)
        hist = self._history_features(st)

        valid_ks = [int(k) for k in candidate_ks if 1 <= int(k) <= int(max_k)]
        if not valid_ks:
            valid_ks = [int(max_k)]

        n = len(valid_ks)

        x_cat = np.tile(cached["cat_ids"][None, :], (n, 1)).astype(np.int64)
        x_num = np.zeros((n, len(self.num_cols)), dtype=np.float32)

        base_num = cached["base_num"]

        for j, col in enumerate(self.num_cols):
            if col == "k_requested":
                vals = np.array(valid_ks, dtype=np.float32)
            elif col in hist:
                vals = np.full(n, float(hist[col]), dtype=np.float32)
            elif col in base_num:
                vals = np.full(n, float(base_num[col]), dtype=np.float32)
            else:
                vals = np.zeros(n, dtype=np.float32)

            x_num[:, j] = vals

        x_num = (x_num - self._num_mean[None, :]) / self._num_std[None, :]

        xb_num = torch.as_tensor(x_num, dtype=torch.float32, device=self.device)
        xb_cat = torch.as_tensor(x_cat, dtype=torch.long, device=self.device)
        k_tensor = torch.as_tensor(valid_ks, dtype=torch.long, device=self.device)

        return valid_ks, xb_num, xb_cat, k_tensor

    @torch.no_grad()
    def choose_k(
        self,
        st: Any,
        control: dict[str, Any],
        candidate_ks: list[int],
        max_k: int,
        trace_scores: bool = False,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()

        valid_ks, xb_num, xb_cat, k_tensor = self._make_tensors(
            st=st,
            control=control,
            candidate_ks=candidate_ks,
            max_k=max_k,
        )

        hazard_logits, pred_log_cost, pred_log_tps, pred_utility = self.model(xb_num, xb_cat)
        eacc = _expected_accepted_len(hazard_logits, k_tensor)

        score = (
            self.score_utility_weight * pred_utility
            + self.score_accept_weight * torch.log(eacc.clamp_min(0.0) + 0.1)
            + self.score_tps_weight * pred_log_tps
        )

        best_i = int(torch.argmax(score).detach().cpu().item())
        chosen_k = int(valid_ks[best_i])

        out = {
            "chosen_k": chosen_k,
            "policy_runtime_ms": float((time.perf_counter() - t0) * 1000.0),
            "model_dir": str(self.model_dir),
        }

        if trace_scores:
            score_cpu = score.detach().cpu().numpy()
            util_cpu = pred_utility.detach().cpu().numpy()
            eacc_cpu = eacc.detach().cpu().numpy()
            tps_cpu = pred_log_tps.detach().cpu().numpy()
            cost_cpu = pred_log_cost.detach().cpu().numpy()

            out["candidate_scores"] = [
                {
                    "k": int(k),
                    "score": float(score_cpu[i]),
                    "pred_utility": float(util_cpu[i]),
                    "pred_expected_accepted_len": float(eacc_cpu[i]),
                    "pred_log_tps": float(tps_cpu[i]),
                    "pred_log_cost": float(cost_cpu[i]),
                }
                for i, k in enumerate(valid_ks)
            ]
        else:
            out["candidate_scores"] = None

        return out
