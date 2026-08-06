from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SlowRouterAction:
    method: str
    k: int
    temperature: float
    score: float
    source: str


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _text_stats(prompt: str) -> dict[str, float]:
    n = max(len(prompt), 1)
    lines = prompt.splitlines() or [prompt]
    n_lines = max(len(lines), 1)

    alpha = sum(ch.isalpha() for ch in prompt)
    digit = sum(ch.isdigit() for ch in prompt)
    space = sum(ch.isspace() for ch in prompt)
    newline = prompt.count("\n")
    symbol = n - alpha - digit - space

    code_chars = sum(prompt.count(x) for x in ["{", "}", ";", "def ", "class ", "import ", "#include", "```"])
    math_chars = sum(prompt.count(x) for x in ["=", "+", "-", "*", "/", "\\", "$", "^", "∑", "√"])

    return {
        "prompt_char_len": float(len(prompt)),
        "prompt_num_lines": float(n_lines),
        "prompt_avg_line_len": float(np.mean([len(x) for x in lines])) if lines else float(len(prompt)),
        "prompt_digit_frac": float(digit / n),
        "prompt_alpha_frac": float(alpha / n),
        "prompt_space_frac": float(space / n),
        "prompt_symbol_frac": float(symbol / n),
        "prompt_code_frac": float(code_chars / n),
        "prompt_math_frac": float(math_chars / n),
        "prompt_newline_frac": float(newline / n),
    }


def _safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def _safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


class ArbitraryPromptSlowRouter:
    """
    Slow-router inference for new prompts.

    It prefers the saved trained router. It also keeps an exact scored-table map
    for prompts that are already in test_candidates_scored.csv, which is useful
    for sanity checks against previous results.
    """

    def __init__(self, slow_router_dir: str | Path):
        self.dir = Path(slow_router_dir)
        router_path = self.dir / "router.pkl"
        if not router_path.exists():
            router_path = self.dir / "request_prompt_router.pkl"
        if not router_path.exists():
            raise FileNotFoundError(
                f"Could not find router.pkl or request_prompt_router.pkl in {self.dir}"
            )
        self.router = _load_pickle(router_path)
        self.prompt_embedder = _load_pickle(self.dir / "prompt_embedder.pkl")

        self.metadata = {}
        meta_path = self.dir / "metadata.json"
        if meta_path.exists():
            self.metadata = json.loads(meta_path.read_text())

        self.train_candidates = pd.read_csv(
            self.dir / "train_candidates_scored.csv",
            keep_default_na=False,
        ) if (self.dir / "train_candidates_scored.csv").exists() else pd.DataFrame()

        self.test_candidates = pd.read_csv(
            self.dir / "test_candidates_scored.csv",
            keep_default_na=False,
        ) if (self.dir / "test_candidates_scored.csv").exists() else pd.DataFrame()

        self.exact_prompt_map = self._build_exact_prompt_map()

        self.feature_columns = self._infer_feature_columns()
        self.candidate_actions = self._infer_candidate_actions()

        print(f"[slow-router] feature columns: {len(self.feature_columns)}")
        print(f"[slow-router] candidate actions: {len(self.candidate_actions)}")
        print(f"[slow-router] exact prompt actions: {len(self.exact_prompt_map)}")

    def _score_column(self, df: pd.DataFrame) -> str:
        for c in ["pred_tps", "pred_actual_tps", "predicted_tps", "router_score", "score", "pred"]:
            if c in df.columns:
                return c
        for c in ["actual_tps", "tokens_per_sec", "target_tokens_per_sec"]:
            if c in df.columns:
                return c
        raise RuntimeError(f"No score column found. Columns={list(df.columns)}")

    def _k_col(self, df: pd.DataFrame) -> str:
        return "k_requested" if "k_requested" in df.columns else "k"

    def _build_exact_prompt_map(self) -> dict[str, SlowRouterAction]:
        out = {}
        df = self.test_candidates
        if len(df) == 0:
            return out

        text_col = None
        for c in ["prompt_text", "prompt", "input_text"]:
            if c in df.columns:
                text_col = c
                break
        if text_col is None:
            return out

        sc = self._score_column(df)
        k_col = self._k_col(df)

        idx = df.groupby(text_col)[sc].idxmax()
        best = df.loc[idx]

        for _, r in best.iterrows():
            out[str(r[text_col])] = SlowRouterAction(
                method=str(r["method"]),
                k=_safe_int(r[k_col], 16),
                temperature=_safe_float(r.get("temperature", 0.0)),
                score=_safe_float(r[sc]),
                source="exact_test_candidates_scored",
            )
        return out

    def _infer_feature_columns(self) -> list[str]:
        # Best case: sklearn pipeline/model remembers feature names.
        if hasattr(self.router, "feature_names_in_"):
            return list(self.router.feature_names_in_)

        # Common metadata keys.
        for k in ["feature_columns", "features", "model_features", "input_features"]:
            if k in self.metadata and isinstance(self.metadata[k], list):
                return list(self.metadata[k])

        # Fallback: infer from train scored table.
        if len(self.train_candidates):
            bad = {
                "actual_tps", "tokens_per_sec", "target_tokens_per_sec",
                "pred_tps", "pred_actual_tps", "predicted_tps",
                "router_score", "score", "pred",
                "prompt", "prompt_text", "input_text",
                "prompt_hash", "prompt_id",
                "workload",
            }
            cols = [c for c in self.train_candidates.columns if c not in bad]
            return cols

        raise RuntimeError("Could not infer feature columns for slow router.")

    def _infer_candidate_actions(self) -> pd.DataFrame:
        dfs = []
        if len(self.train_candidates):
            dfs.append(self.train_candidates)
        if len(self.test_candidates):
            dfs.append(self.test_candidates)
        if not dfs:
            # Conservative default action set.
            return pd.DataFrame([
                {"method": "ngram_sd", "k_requested": 16, "temperature": 0.0,
                 "ngram_lookup_min": "1", "ngram_lookup_max": "4",
                 "draft_model": "NA", "eagle3_model": "NA",
                 "max_prompt_tokens": 0, "num_turns": 1, "prompt_token_len": 0},
                {"method": "eagle3", "k_requested": 4, "temperature": 0.0,
                 "ngram_lookup_min": "NA", "ngram_lookup_max": "NA",
                 "draft_model": "NA", "eagle3_model": "RedHatAI/Qwen3-8B-speculator.eagle3",
                 "max_prompt_tokens": 0, "num_turns": 1, "prompt_token_len": 0},
            ])

        df = pd.concat(dfs, ignore_index=True)
        k_col = self._k_col(df)

        cols = [
            "method", k_col, "temperature",
            "ngram_lookup_min", "ngram_lookup_max",
            "draft_model", "eagle3_model",
            "max_prompt_tokens", "num_turns",
        ]
        cols = [c for c in cols if c in df.columns]

        cand = df[cols].drop_duplicates().copy()
        if k_col != "k_requested":
            cand = cand.rename(columns={k_col: "k_requested"})

        cand["k_requested"] = cand["k_requested"].map(lambda x: _safe_int(x, 16))
        if "temperature" not in cand.columns:
            cand["temperature"] = 0.0

        return cand.reset_index(drop=True)

    def _embed_prompt(self, prompt: str) -> np.ndarray:
        # Support likely APIs from the saved PromptEmbedder.
        if hasattr(self.prompt_embedder, "transform"):
            emb = self.prompt_embedder.transform([prompt])
        elif hasattr(self.prompt_embedder, "encode"):
            emb = self.prompt_embedder.encode([prompt])
        else:
            raise RuntimeError("prompt_embedder has neither transform nor encode")

        emb = np.asarray(emb)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        return emb[0]

    def _build_candidates(self, prompt: str, workload: str | None = None) -> pd.DataFrame:
        cand = self.candidate_actions.copy()
        cand = cand.replace("", "NA")

        stats = _text_stats(prompt)
        for k, v in stats.items():
            cand[k] = v

        emb = self._embed_prompt(prompt)
        for i, val in enumerate(emb):
            cand[f"prompt_emb_{i}"] = float(val)

        # Approximate prompt_token_len if no tokenizer-specific column is available.
        approx_tokens = max(1, int(len(prompt.split()) * 1.3))
        cand["prompt_token_len"] = approx_tokens

        if workload is not None:
            cand["workload"] = workload

        # Normalize common categorical fields.
        for c in ["draft_model", "eagle3_model", "ngram_lookup_min", "ngram_lookup_max"]:
            if c not in cand.columns:
                cand[c] = "NA"
            cand[c] = cand[c].astype(str).replace({"": "NA", "nan": "NA", "None": "NA"})

        for c in ["max_prompt_tokens", "num_turns"]:
            if c not in cand.columns:
                cand[c] = 0 if c == "max_prompt_tokens" else 1
            cand[c] = cand[c].map(lambda x: _safe_int(x, 0 if c == "max_prompt_tokens" else 1))

        return cand

    def _align_features(self, cand: pd.DataFrame) -> pd.DataFrame:
        X = cand.copy()

        # Add missing feature cols.
        for c in self.feature_columns:
            if c not in X.columns:
                X[c] = 0.0

        X = X[self.feature_columns].copy()

        # Convert obvious numeric strings.
        for c in X.columns:
            if X[c].dtype == object:
                # If the saved model/pipeline handles categoricals, leave them.
                # If it is a raw tree model, prediction may fail and we try get_dummies below.
                pass

        return X

    def choose(self, prompt: str, workload: str | None = None, exact_ok: bool = True) -> SlowRouterAction:
        if exact_ok and prompt in self.exact_prompt_map:
            return self.exact_prompt_map[prompt]

        cand = self._build_candidates(prompt, workload)
        X = self._align_features(cand)

        try:
            pred = self.router.predict(X)
        except Exception:
            # Fallback for raw sklearn model trained on one-hot columns.
            X2 = pd.get_dummies(X, dummy_na=False)
            for c in self.feature_columns:
                if c not in X2.columns:
                    X2[c] = 0.0
            common_cols = [c for c in self.feature_columns if c in X2.columns]
            pred = self.router.predict(X2[common_cols])

        cand = cand.copy()
        cand["pred_tps"] = np.asarray(pred, dtype=float)

        i = int(cand["pred_tps"].idxmax())
        r = cand.loc[i]

        return SlowRouterAction(
            method=str(r["method"]),
            k=_safe_int(r["k_requested"], 16),
            temperature=_safe_float(r.get("temperature", 0.0)),
            score=float(r["pred_tps"]),
            source="router_pkl_arbitrary_prompt",
        )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--slow-router-dir", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--workload", default="")
    args = ap.parse_args()

    r = ArbitraryPromptSlowRouter(args.slow_router_dir)
    a = r.choose(args.prompt, args.workload or None, exact_ok=False)
    print(json.dumps(a.__dict__, indent=2, sort_keys=True))
