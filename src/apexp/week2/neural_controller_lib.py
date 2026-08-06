#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.apexp.week2.prompt_embedder import PromptEmbedder


MAX_K = 16
PROMPT_EMB_DIM = 64
PROMPT_EMB_COLS = [f"prompt_emb_{i}" for i in range(PROMPT_EMB_DIM)]


# Deployable controller inputs.
# IMPORTANT: no workload label here.
DEPLOYABLE_CAT_COLS = [
    "method",
    "draft_model",
    "eagle3_model",
    "ngram_lookup_min",
    "ngram_lookup_max",
]


DEPLOYABLE_NUM_COLS = [
    "k_requested",
    "temperature",
    "prompt_token_len",
    "prefix_pos",
    "prefix_frac",

    "prefix_entropy_mean",
    "prefix_entropy_std",
    "prefix_entropy_mean_w16",
    "prefix_entropy_mean_w32",
    "prefix_entropy_mean_w64",
    "prefix_entropy_mean_w128",
    "prefix_entropy_mean_w256",
    "prefix_entropy_std_w16",
    "prefix_entropy_std_w32",
    "prefix_entropy_std_w64",
    "prefix_entropy_std_w128",

    "prefix_repetition_w16",
    "prefix_repetition_w32",
    "prefix_repetition_w64",
    "prefix_repetition_w128",
    "prefix_repetition_w256",

    "hist_n_blocks",
    "hist_mean_accepted_len",
    "hist_std_accepted_len",
    "hist_mean_first_rejection",
    "hist_full_accept_rate",
    "hist_reject_rate",
    "hist_mean_num_rejected",

    "hist_mean_accepted_len_last1",
    "hist_mean_accepted_len_last2",
    "hist_mean_accepted_len_last4",
    "hist_mean_accepted_len_last8",
    "hist_mean_accepted_len_last16",

    "hist_full_accept_rate_last1",
    "hist_full_accept_rate_last2",
    "hist_full_accept_rate_last4",
    "hist_full_accept_rate_last8",
    "hist_full_accept_rate_last16",

    "hist_mean_first_rejection_last4",
    "hist_mean_num_rejected_last4",
] + PROMPT_EMB_COLS


def safe_num(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def prompt_group_key(df: pd.DataFrame) -> pd.Series:
    if "prompt_hash" in df.columns:
        return df["prompt_hash"].astype(str)
    if "prompt_id" in df.columns:
        return df["prompt_id"].astype(str)
    if "id" in df.columns:
        return df["id"].astype(str)
    return df.index.astype(str)


def ensure_feature_columns(df: pd.DataFrame, cat_cols, num_cols):
    for c in cat_cols:
        if c not in df.columns:
            df[c] = "NA"
    for c in num_cols:
        if c not in df.columns:
            df[c] = 0.0
    return df


def add_prompt_embedding_columns(df: pd.DataFrame, embedder: PromptEmbedder | None):
    df = df.copy()

    if embedder is None:
        for c in PROMPT_EMB_COLS:
            if c not in df.columns:
                df[c] = 0.0
        return df

    if "prompt_text" not in df.columns:
        raise ValueError("prompt_text is required to compute prompt embeddings.")

    Z = embedder.transform(df["prompt_text"].fillna("").astype(str).tolist())
    for i, c in enumerate(PROMPT_EMB_COLS):
        df[c] = Z[:, i]
    return df


class FeaturePreprocessor:
    def __init__(self, cat_cols=None, num_cols=None):
        self.cat_cols = cat_cols or DEPLOYABLE_CAT_COLS
        self.num_cols = num_cols or DEPLOYABLE_NUM_COLS
        self.vocab = {}
        self.num_stats = {}

    def fit(self, df: pd.DataFrame):
        df = ensure_feature_columns(df.copy(), self.cat_cols, self.num_cols)

        self.vocab = {}
        for c in self.cat_cols:
            vals = df[c].fillna("NA").astype(str).unique().tolist()
            vals = ["<UNK>"] + sorted(vals)
            self.vocab[c] = {v: i for i, v in enumerate(vals)}

        self.num_stats = {}
        for c in self.num_cols:
            x = safe_num(df[c]).fillna(0.0).to_numpy(dtype=np.float32)
            mu = float(np.mean(x))
            sd = float(np.std(x))
            if not np.isfinite(sd) or sd < 1e-6:
                sd = 1.0
            self.num_stats[c] = {"mean": mu, "std": sd}

        return self

    def transform(self, df: pd.DataFrame):
        df = ensure_feature_columns(df.copy(), self.cat_cols, self.num_cols)

        cat_arrs = []
        for c in self.cat_cols:
            mp = self.vocab[c]
            arr = df[c].fillna("NA").astype(str).map(lambda x: mp.get(x, 0)).to_numpy()
            cat_arrs.append(arr)

        x_cat = np.stack(cat_arrs, axis=1).astype(np.int64)

        num_arrs = []
        for c in self.num_cols:
            x = safe_num(df[c]).fillna(0.0).to_numpy(dtype=np.float32)
            mu = self.num_stats[c]["mean"]
            sd = self.num_stats[c]["std"]
            num_arrs.append((x - mu) / sd)

        x_num = np.stack(num_arrs, axis=1).astype(np.float32)
        return x_num, x_cat

    def cat_cardinalities(self):
        return [len(self.vocab[c]) for c in self.cat_cols]

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path):
        with open(path, "rb") as f:
            return pickle.load(f)


class BlockDataset(torch.utils.data.Dataset):
    def __init__(self, x_num, x_cat, y_accept, y_k, y_log_cost, y_log_tps):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.y_accept = torch.tensor(y_accept, dtype=torch.long)
        self.y_k = torch.tensor(y_k, dtype=torch.long)
        self.y_log_cost = torch.tensor(y_log_cost, dtype=torch.float32)
        self.y_log_tps = torch.tensor(y_log_tps, dtype=torch.float32)

    def __len__(self):
        return len(self.y_accept)

    def __getitem__(self, idx):
        return (
            self.x_num[idx],
            self.x_cat[idx],
            self.y_accept[idx],
            self.y_k[idx],
            self.y_log_cost[idx],
            self.y_log_tps[idx],
        )


class SurvivalCostNet(nn.Module):
    def __init__(self, num_dim, cat_cardinalities, emb_dim=16, hidden=256, dropout=0.1):
        super().__init__()

        self.embs = nn.ModuleList()
        emb_total = 0
        for card in cat_cardinalities:
            d = min(emb_dim, max(4, int(math.sqrt(card)) + 1))
            self.embs.append(nn.Embedding(card, d))
            emb_total += d

        self.trunk = nn.Sequential(
            nn.Linear(num_dim + emb_total, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.hazard = nn.Linear(hidden, MAX_K)
        self.log_cost = nn.Linear(hidden, 1)
        self.log_tps = nn.Linear(hidden, 1)

    def forward(self, x_num, x_cat):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        x = torch.cat([x_num] + embs, dim=1)
        h = self.trunk(x)

        hazard_logits = self.hazard(h)
        pred_log_cost = self.log_cost(h).squeeze(-1)
        pred_log_tps = self.log_tps(h).squeeze(-1)

        return hazard_logits, pred_log_cost, pred_log_tps


def survival_loss(hazard_logits, y_accept, y_k):
    """
    Discrete censored hazard loss.

    If accepted_len = r < k:
      positions 0..r-1 survived, position r rejected.

    If accepted_len >= k:
      positions 0..k-1 survived, no rejection observed.
    """
    bce = nn.BCEWithLogitsLoss(reduction="none")

    B = hazard_logits.shape[0]
    device = hazard_logits.device

    pos = torch.arange(MAX_K, device=device).unsqueeze(0).expand(B, MAX_K)
    k = y_k.clamp(1, MAX_K).unsqueeze(1)
    acc = y_accept.clamp(0, MAX_K).unsqueeze(1)

    active = pos < k

    labels = torch.zeros_like(hazard_logits)
    has_rejection = (y_accept < y_k).float().unsqueeze(1)
    reject_idx = y_accept.clamp(0, MAX_K - 1).unsqueeze(1)
    labels = labels.scatter(1, reject_idx, has_rejection)

    include = active & (pos <= acc)

    raw = bce(hazard_logits, labels)
    return (raw * include.float()).sum() / include.float().sum().clamp_min(1.0)


def expected_accepted_len(hazard_logits, k_values):
    hazard = torch.sigmoid(hazard_logits)
    survival_token = torch.cumprod(1.0 - hazard, dim=1)

    B = hazard_logits.shape[0]
    device = hazard_logits.device
    pos = torch.arange(MAX_K, device=device).unsqueeze(0).expand(B, MAX_K)

    mask = pos < k_values.clamp(1, MAX_K).unsqueeze(1)
    return (survival_token * mask.float()).sum(dim=1)


def load_model(model_dir: str | Path, device=None):
    model_dir = Path(model_dir)
    pre = FeaturePreprocessor.load(model_dir / "preprocessor.pkl")
    metadata = json.loads((model_dir / "metadata.json").read_text())

    model = SurvivalCostNet(
        num_dim=len(pre.num_cols),
        cat_cardinalities=pre.cat_cardinalities(),
        hidden=metadata.get("hidden", 256),
        dropout=metadata.get("dropout", 0.1),
    )

    state = torch.load(model_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)

    embedder_path = model_dir / "prompt_embedder.pkl"
    embedder = PromptEmbedder.load(embedder_path) if embedder_path.exists() else None

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    return model, pre, metadata, embedder, device


@torch.no_grad()
def score_dataframe(model, pre, df, device, embedder=None, variant="full", batch_size=65536):
    df = add_prompt_embedding_columns(df, embedder)
    x_num, x_cat = pre.transform(df)

    scores = []
    pred_eaccs = []
    pred_log_costs = []
    pred_log_tpss = []

    for i in range(0, len(df), batch_size):
        xb_num = torch.tensor(x_num[i:i+batch_size], dtype=torch.float32, device=device)
        xb_cat = torch.tensor(x_cat[i:i+batch_size], dtype=torch.long, device=device)

        hz, log_cost, log_tps = model(xb_num, xb_cat)

        if "k_requested" in df.columns:
            kval = safe_num(df["k_requested"].iloc[i:i+batch_size]).fillna(1).clip(1, MAX_K).to_numpy()
        else:
            kval = np.ones(len(xb_num), dtype=np.int64)

        k_tensor = torch.tensor(kval, dtype=torch.long, device=device)
        eacc = expected_accepted_len(hz, k_tensor)

        if variant == "direct":
            score = torch.expm1(log_tps).clamp_min(0.0)
        elif variant == "no_cost":
            score = eacc
        else:
            score = eacc * torch.exp(-log_cost)

        scores.append(score.detach().cpu().numpy())
        pred_eaccs.append(eacc.detach().cpu().numpy())
        pred_log_costs.append(log_cost.detach().cpu().numpy())
        pred_log_tpss.append(log_tps.detach().cpu().numpy())

    return pd.DataFrame({
        "pred_score": np.concatenate(scores),
        "pred_expected_accepted_len": np.concatenate(pred_eaccs),
        "pred_log_cost": np.concatenate(pred_log_costs),
        "pred_log_tps": np.concatenate(pred_log_tpss),
    })
