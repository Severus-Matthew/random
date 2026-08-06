from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.apexp.week2.neural_controller_lib import (
    MAX_K,
    DEPLOYABLE_CAT_COLS,
    DEPLOYABLE_NUM_COLS,
    FeaturePreprocessor,
    expected_accepted_len,
    add_prompt_embedding_columns,
)
from src.apexp.week2.prompt_embedder import PromptEmbedder
from src.apexp.week2.train_survival_utility_controller_v4 import SurvivalUtilityNet


def _safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x, default=0):
    try:
        if x is None:
            return default
        return int(float(x))
    except Exception:
        return default


class LearnedFastSurvivalPolicy:
    def __init__(self, model_dir: str | Path, device: str | None = None):
        self.model_dir = Path(model_dir)
        self.metadata = json.loads((self.model_dir / "metadata.json").read_text())

        self.pre: FeaturePreprocessor = FeaturePreprocessor.load(self.model_dir / "preprocessor.pkl")
        self.embedder: PromptEmbedder = PromptEmbedder.load(self.model_dir / "prompt_embedder.pkl")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

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

        self.score_utility_weight = float(self.metadata.get("online_score_utility_weight", 1.0))
        self.score_accept_weight = float(self.metadata.get("online_score_accept_weight", 0.10))
        self.score_tps_weight = float(self.metadata.get("online_score_tps_weight", 0.05))

    def _history_features(self, st: Any) -> dict[str, float]:
        accepted = list(getattr(st, "accepted_hist", []))
        rejected = list(getattr(st, "rejected_hist", []))
        full = list(getattr(st, "full_accept_hist", []))
        k_hist = list(getattr(st, "k_hist", []))

        # Approximate first rejection from accepted length.
        first_rej = []
        for a, k in zip(accepted, k_hist):
            if a >= k:
                first_rej.append(k)
            else:
                first_rej.append(a)

        out = {
            "hist_n_blocks": float(len(accepted)),
            "hist_mean_accepted_len": float(np.mean(accepted)) if accepted else 0.0,
            "hist_std_accepted_len": float(np.std(accepted)) if accepted else 0.0,
            "hist_mean_first_rejection": float(np.mean(first_rej)) if first_rej else 0.0,
            "hist_full_accept_rate": float(np.mean(full)) if full else 0.0,
            "hist_reject_rate": 1.0 - float(np.mean(full)) if full else 0.0,
            "hist_mean_num_rejected": float(np.mean(rejected)) if rejected else 0.0,
        }

        for w in [1, 2, 4, 8, 16]:
            pa = accepted[-w:]
            pf = full[-w:]
            pr = first_rej[-w:]
            nr = rejected[-w:]

            out[f"hist_mean_accepted_len_last{w}"] = float(np.mean(pa)) if pa else 0.0
            out[f"hist_full_accept_rate_last{w}"] = float(np.mean(pf)) if pf else 0.0
            out[f"hist_mean_first_rejection_last{w}"] = float(np.mean(pr)) if pr else 0.0
            out[f"hist_mean_num_rejected_last{w}"] = float(np.mean(nr)) if nr else 0.0

        return out

    def _base_row(self, st: Any, control: dict[str, Any]) -> dict[str, Any]:
        prompt_text = str(control.get("prompt_text", "") or "")

        n_blocks = int(getattr(st, "n_blocks", 0))
        current_k = int(getattr(st, "current_k", control.get("slow_k", control.get("default_k", 16))) or 16)

        row = {
            "method": str(control.get("method", "ngram_sd")),
            "draft_model": str(control.get("draft_model", "NA") or "NA"),
            "eagle3_model": str(control.get("eagle3_model", "NA") or "NA"),
            "ngram_lookup_min": str(control.get("ngram_lookup_min", "1") or "1"),
            "ngram_lookup_max": str(control.get("ngram_lookup_max", "4") or "4"),
            "temperature": _safe_float(control.get("temperature", 0.0), 0.0),
            "prompt_token_len": _safe_float(control.get("prompt_token_len", 0.0), 0.0),
            "prefix_pos": float(n_blocks),
            "prefix_frac": 0.0,
            "prefix_entropy_mean": _safe_float(control.get("prefix_entropy_mean", 0.0), 0.0),
            "prefix_entropy_std": _safe_float(control.get("prefix_entropy_std", 0.0), 0.0),
            "prompt_text": prompt_text,
        }

        # If online entropy/repetition is not available, default to 0.
        for w in [16, 32, 64, 128, 256]:
            row[f"prefix_entropy_mean_w{w}"] = _safe_float(control.get(f"prefix_entropy_mean_w{w}", row["prefix_entropy_mean"]), row["prefix_entropy_mean"])
            row[f"prefix_entropy_std_w{w}"] = _safe_float(control.get(f"prefix_entropy_std_w{w}", 0.0), 0.0)
            row[f"prefix_repetition_w{w}"] = _safe_float(control.get(f"prefix_repetition_w{w}", 0.0), 0.0)

        row.update(self._history_features(st))
        return row

    @torch.no_grad()
    def choose_k(
        self,
        st: Any,
        control: dict[str, Any],
        candidate_ks: list[int],
        max_k: int,
    ) -> dict[str, Any]:
        t0 = time.time()

        base = self._base_row(st, control)
        rows = []

        for k in candidate_ks:
            if int(k) > int(max_k):
                continue
            r = dict(base)
            r["k_requested"] = int(k)
            rows.append(r)

        if not rows:
            return {
                "chosen_k": int(max_k),
                "policy_runtime_ms": 0.0,
                "candidate_scores": [],
                "error": "no_candidate_rows",
            }

        df = pd.DataFrame(rows)

        # Ensure all feature columns exist.
        for c in DEPLOYABLE_CAT_COLS:
            if c not in df.columns:
                df[c] = "NA"
        for c in DEPLOYABLE_NUM_COLS:
            if c not in df.columns:
                df[c] = 0.0

        df = add_prompt_embedding_columns(df, self.embedder)
        x_num, x_cat = self.pre.transform(df)

        xb_num = torch.tensor(x_num, dtype=torch.float32, device=self.device)
        xb_cat = torch.tensor(x_cat, dtype=torch.long, device=self.device)

        hz, pred_log_cost, pred_log_tps, pred_utility = self.model(xb_num, xb_cat)

        k_tensor = torch.tensor(df["k_requested"].astype(int).to_numpy(), dtype=torch.long, device=self.device)
        eacc = expected_accepted_len(hz, k_tensor)

        score = (
            self.score_utility_weight * pred_utility
            + self.score_accept_weight * torch.log(eacc.clamp_min(0.0) + 0.1)
            + self.score_tps_weight * pred_log_tps
        )

        best_i = int(torch.argmax(score).detach().cpu().item())
        chosen_k = int(df.iloc[best_i]["k_requested"])

        cand = []
        for i, r in df.iterrows():
            cand.append({
                "k": int(r["k_requested"]),
                "score": float(score[i].detach().cpu().item()),
                "pred_utility": float(pred_utility[i].detach().cpu().item()),
                "pred_expected_accepted_len": float(eacc[i].detach().cpu().item()),
                "pred_log_tps": float(pred_log_tps[i].detach().cpu().item()),
                "pred_log_cost": float(pred_log_cost[i].detach().cpu().item()),
            })

        return {
            "chosen_k": chosen_k,
            "policy_runtime_ms": float((time.time() - t0) * 1000.0),
            "candidate_scores": cand,
            "model_dir": str(self.model_dir),
        }
