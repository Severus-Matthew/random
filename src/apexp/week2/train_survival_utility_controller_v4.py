#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.apexp.week2.prompt_embedder import PromptEmbedder
from src.apexp.week2.neural_controller_lib import (
    MAX_K,
    DEPLOYABLE_CAT_COLS,
    DEPLOYABLE_NUM_COLS,
    PROMPT_EMB_COLS,
    FeaturePreprocessor,
    survival_loss,
    expected_accepted_len,
    safe_num,
    prompt_group_key,
    add_prompt_embedding_columns,
)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def merge_prompt_text(df: pd.DataFrame, prompt_text_map_path: str | Path) -> pd.DataFrame:
    mp = pd.read_csv(prompt_text_map_path)
    if "prompt_hash" not in mp.columns or "prompt_text" not in mp.columns:
        raise ValueError("prompt_text_map must contain prompt_hash and prompt_text")

    df = df.copy()
    df["prompt_hash"] = df["prompt_hash"].astype(str)
    mp["prompt_hash"] = mp["prompt_hash"].astype(str)

    before = len(df)
    df = df.merge(mp[["prompt_hash", "prompt_text"]], on="prompt_hash", how="left")
    assert len(df) == before

    match_rate = float(df["prompt_text"].notna().mean())
    print(f"prompt_text row match rate: {match_rate:.4f}")

    if match_rate < 0.90:
        raise RuntimeError(f"prompt_text row match rate too low: {match_rate:.4f}")

    df["prompt_text"] = df["prompt_text"].fillna("")
    return df


def add_targets_and_utility(
    df: pd.DataFrame,
    alpha_accept: float,
    alpha_waste: float,
    alpha_global: float,
    alpha_method: float,
) -> pd.DataFrame:
    df = df.copy()

    required = [
        "target_accepted_len",
        "target_num_rejected",
        "target_k_actual",
        "target_tokens_per_sec",
        "k_requested",
        "method",
        "prompt_hash",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["target_accepted_len"] = (
        safe_num(df["target_accepted_len"])
        .fillna(0)
        .clip(0, MAX_K)
        .astype(int)
    )

    df["target_k_actual"] = (
        safe_num(df["target_k_actual"])
        .fillna(df["k_requested"])
        .clip(1, MAX_K)
        .astype(int)
    )

    df["target_num_rejected"] = (
        safe_num(df["target_num_rejected"])
        .fillna(0)
        .clip(0, MAX_K)
        .astype(float)
    )

    df["target_tokens_per_sec"] = (
        safe_num(df["target_tokens_per_sec"])
        .fillna(0.0)
        .clip(lower=1e-6)
    )

    if "target_latency_s" in df.columns:
        latency = safe_num(df["target_latency_s"]).fillna(0.0)
        has_latency = latency > 0
        df["target_cost_s"] = np.where(
            has_latency,
            latency.clip(lower=1e-9),
            1.0 / df["target_tokens_per_sec"].clip(lower=1e-6),
        )
        cost_source = "target_latency_s_or_inverse_tps"
    else:
        df["target_cost_s"] = 1.0 / df["target_tokens_per_sec"].clip(lower=1e-6)
        cost_source = "inverse_tps"

    df["target_log_cost"] = np.log(df["target_cost_s"].clip(lower=1e-9))
    df["target_log_tps"] = np.log1p(df["target_tokens_per_sec"].clip(lower=1e-6))

    df["target_waste_rate"] = (
        df["target_num_rejected"] / df["target_k_actual"].clip(lower=1)
    ).clip(0, 1)

    # Best fixed expert for this prompt across all methods/k.
    best_global = (
        df.groupby(["prompt_hash"], dropna=False)["target_tokens_per_sec"]
        .max()
        .rename("best_expert_tps")
        .reset_index()
    )
    df = df.merge(best_global, on="prompt_hash", how="left")

    # Best fixed depth within the same method for this prompt.
    best_method = (
        df.groupby(["prompt_hash", "method"], dropna=False)["target_tokens_per_sec"]
        .max()
        .rename("best_method_tps")
        .reset_index()
    )
    df = df.merge(best_method, on=["prompt_hash", "method"], how="left")

    df["best_expert_tps"] = safe_num(df["best_expert_tps"]).fillna(0.0)
    df["best_method_tps"] = safe_num(df["best_method_tps"]).fillna(0.0)

    df["global_best_regret"] = 0.0
    m = df["best_expert_tps"] > 0
    df.loc[m, "global_best_regret"] = (
        (df.loc[m, "best_expert_tps"] - df.loc[m, "target_tokens_per_sec"]).clip(lower=0)
        / df.loc[m, "best_expert_tps"]
    )

    df["method_best_regret"] = 0.0
    m = df["best_method_tps"] > 0
    df.loc[m, "method_best_regret"] = (
        (df.loc[m, "best_method_tps"] - df.loc[m, "target_tokens_per_sec"]).clip(lower=0)
        / df.loc[m, "best_method_tps"]
    )

    eps = 0.1
    df["target_utility"] = (
        np.log(df["target_tokens_per_sec"].clip(lower=1e-6))
        + alpha_accept * np.log(df["target_accepted_len"].clip(lower=0) + eps)
        - alpha_waste * df["target_waste_rate"]
        - alpha_global * df["global_best_regret"]
        - alpha_method * df["method_best_regret"]
    )

    # Ranking group: same prompt/method/block state, different candidate k.
    # This teaches the model which k to select.
    if "block_index" not in df.columns:
        df["block_index"] = 0

    df["rank_group_key"] = (
        df["prompt_hash"].astype(str)
        + "||" + df["method"].astype(str)
        + "||" + df["block_index"].astype(str)
    )

    # Analysis metrics only.
    df["target_speed_efficiency"] = df["target_tokens_per_sec"] * (1.0 - df["target_waste_rate"])
    df["target_accepted_throughput_efficiency"] = (
        df["target_tokens_per_sec"]
        * df["target_accepted_len"].clip(lower=0)
        * (1.0 - df["target_waste_rate"])
    )

    return df, cost_source


class UtilityBlockDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        x_num,
        x_cat,
        y_accept,
        y_k,
        y_log_cost,
        y_log_tps,
        y_utility,
        rank_group_id,
    ):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.y_accept = torch.tensor(y_accept, dtype=torch.long)
        self.y_k = torch.tensor(y_k, dtype=torch.long)
        self.y_log_cost = torch.tensor(y_log_cost, dtype=torch.float32)
        self.y_log_tps = torch.tensor(y_log_tps, dtype=torch.float32)
        self.y_utility = torch.tensor(y_utility, dtype=torch.float32)
        self.rank_group_id = torch.tensor(rank_group_id, dtype=torch.long)

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
            self.y_utility[idx],
            self.rank_group_id[idx],
        )


class SurvivalUtilityNet(nn.Module):
    def __init__(self, num_dim, cat_cardinalities, emb_dim=16, hidden=512, dropout=0.1):
        super().__init__()

        self.embs = nn.ModuleList()
        emb_total = 0
        for card in cat_cardinalities:
            d = min(emb_dim, max(4, int(math.sqrt(card)) + 1))
            self.embs.append(nn.Embedding(card, d))
            emb_total += d

        self.trunk = nn.Sequential(
            nn.Linear(num_dim + emb_total, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        h2 = hidden // 2
        self.hazard = nn.Linear(h2, MAX_K)
        self.log_cost = nn.Linear(h2, 1)
        self.log_tps = nn.Linear(h2, 1)
        self.utility = nn.Linear(h2, 1)

    def forward(self, x_num, x_cat):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        x = torch.cat([x_num] + embs, dim=1)
        h = self.trunk(x)

        hazard_logits = self.hazard(h)
        pred_log_cost = self.log_cost(h).squeeze(-1)
        pred_log_tps = self.log_tps(h).squeeze(-1)
        pred_utility = self.utility(h).squeeze(-1)

        return hazard_logits, pred_log_cost, pred_log_tps, pred_utility


def rank_loss_in_batch(pred_utility, target_utility, rank_group_id, max_groups=2048):
    """
    Within each rank group, encourage the highest-predicted k to match
    the highest true-utility k.

    This is intentionally in-batch. With shuffled large batches, many groups
    will have multiple candidate k rows.
    """
    unique = torch.unique(rank_group_id)
    if unique.numel() > max_groups:
        unique = unique[:max_groups]

    losses = []
    for gid in unique:
        idx = torch.nonzero(rank_group_id == gid, as_tuple=False).squeeze(-1)
        if idx.numel() < 2:
            continue

        logits = pred_utility[idx].unsqueeze(0)
        best_local = torch.argmax(target_utility[idx]).view(1)
        losses.append(nn.functional.cross_entropy(logits, best_local))

    if not losses:
        return pred_utility.new_tensor(0.0)

    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate_prediction(model, loader, device):
    model.eval()
    rows = []

    for batch in loader:
        (
            x_num,
            x_cat,
            y_accept,
            y_k,
            y_log_cost,
            y_log_tps,
            y_utility,
            rank_group_id,
        ) = batch

        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y_accept = y_accept.to(device)
        y_k = y_k.to(device)
        y_log_cost = y_log_cost.to(device)
        y_log_tps = y_log_tps.to(device)
        y_utility = y_utility.to(device)
        rank_group_id = rank_group_id.to(device)

        hz, pred_log_cost, pred_log_tps, pred_utility = model(x_num, x_cat)
        pred_eacc = expected_accepted_len(hz, y_k)

        rows.append(pd.DataFrame({
            "target_accepted_len": y_accept.cpu().numpy(),
            "target_k": y_k.cpu().numpy(),
            "target_log_cost": y_log_cost.cpu().numpy(),
            "target_log_tps": y_log_tps.cpu().numpy(),
            "target_utility": y_utility.cpu().numpy(),
            "pred_expected_accepted_len": pred_eacc.cpu().numpy(),
            "pred_log_cost": pred_log_cost.cpu().numpy(),
            "pred_log_tps": pred_log_tps.cpu().numpy(),
            "pred_utility": pred_utility.cpu().numpy(),
            "rank_group_id": rank_group_id.cpu().numpy(),
        }))

    out = pd.concat(rows, ignore_index=True)

    metrics = {
        "mae_accepted_len": float(np.mean(np.abs(out["target_accepted_len"] - out["pred_expected_accepted_len"]))),
        "mae_log_cost": float(np.mean(np.abs(out["target_log_cost"] - out["pred_log_cost"]))),
        "mae_log_tps": float(np.mean(np.abs(out["target_log_tps"] - out["pred_log_tps"]))),
        "mae_utility": float(np.mean(np.abs(out["target_utility"] - out["pred_utility"]))),
        "mean_target_accepted_len": float(out["target_accepted_len"].mean()),
        "mean_pred_expected_accepted_len": float(out["pred_expected_accepted_len"].mean()),
        "mean_target_utility": float(out["target_utility"].mean()),
        "mean_pred_utility": float(out["pred_utility"].mean()),
    }

    # Ranking sanity: for groups present in eval rows, compare selected vs oracle utility.
    grp = out.groupby("rank_group_id", sort=False)
    selected_utils = []
    oracle_utils = []
    exact = []

    for _, g in grp:
        if len(g) < 2:
            continue
        pred_i = int(g["pred_utility"].values.argmax())
        oracle_i = int(g["target_utility"].values.argmax())
        selected_utils.append(float(g["target_utility"].values[pred_i]))
        oracle_utils.append(float(g["target_utility"].values[oracle_i]))
        exact.append(int(pred_i == oracle_i))

    if oracle_utils:
        metrics["policy_selected_utility_mean"] = float(np.mean(selected_utils))
        metrics["policy_oracle_utility_mean"] = float(np.mean(oracle_utils))
        metrics["policy_oracle_fraction"] = float(np.mean(selected_utils) / max(1e-9, np.mean(oracle_utils)))
        metrics["policy_exact_k_match_proxy"] = float(np.mean(exact))
        metrics["n_eval_rank_groups"] = int(len(oracle_utils))
    else:
        metrics["policy_selected_utility_mean"] = None
        metrics["policy_oracle_utility_mean"] = None
        metrics["policy_oracle_fraction"] = None
        metrics["policy_exact_k_match_proxy"] = None
        metrics["n_eval_rank_groups"] = 0

    return metrics, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--prompt-text-map", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--max-rows", type=int, default=0, help="0 means use full dataset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--prompt-emb-dim", type=int, default=64)

    ap.add_argument("--alpha-accept", type=float, default=0.30)
    ap.add_argument("--alpha-waste", type=float, default=0.35)
    ap.add_argument("--alpha-global", type=float, default=0.15)
    ap.add_argument("--alpha-method", type=float, default=0.20)

    ap.add_argument("--lambda-survival", type=float, default=1.00)
    ap.add_argument("--lambda-cost", type=float, default=0.20)
    ap.add_argument("--lambda-tps", type=float, default=0.30)
    ap.add_argument("--lambda-utility", type=float, default=1.00)
    ap.add_argument("--lambda-rank", type=float, default=1.00)
    ap.add_argument("--grouped-rank-batches", action="store_true",
                    help="Sort training rows so candidate k rows from the same rank group appear in the same batch.")

    args = ap.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading:", args.data)
    df = pd.read_csv(args.data, low_memory=False)
    print("raw rows:", len(df))

    df = df[df["method"].isin(["ngram_sd", "draft_sd", "eagle3"])].copy()
    print("spec rows:", len(df))

    df = merge_prompt_text(df, args.prompt_text_map)

    df, cost_source = add_targets_and_utility(
        df,
        alpha_accept=args.alpha_accept,
        alpha_waste=args.alpha_waste,
        alpha_global=args.alpha_global,
        alpha_method=args.alpha_method,
    )

    df["group_key"] = prompt_group_key(df)
    rank_keys = sorted(df["rank_group_key"].astype(str).unique())
    rank_map = {k: i for i, k in enumerate(rank_keys)}
    df["rank_group_id"] = df["rank_group_key"].astype(str).map(rank_map).astype(int)

    groups = np.array(sorted(df["group_key"].unique()))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(groups)

    n_train = int(0.8 * len(groups))
    train_groups = set(groups[:n_train])
    test_groups = set(groups[n_train:])

    train = df[df["group_key"].isin(train_groups)].copy()
    test = df[df["group_key"].isin(test_groups)].copy()

    if args.max_rows and args.max_rows > 0 and len(train) > args.max_rows:
        train = train.sample(args.max_rows, random_state=args.seed).copy()

    train_shuffle = True
    if args.grouped_rank_batches:
        # Shuffle rank groups, but keep rows within each group adjacent.
        # This makes the in-batch ranking loss actually see multiple candidate-k
        # rows for the same prompt/method/block state.
        unique_rank_groups = train["rank_group_id"].drop_duplicates().to_numpy()
        rng_rank = np.random.default_rng(args.seed)
        rng_rank.shuffle(unique_rank_groups)
        rank_order = {int(g): i for i, g in enumerate(unique_rank_groups)}
        train["_rank_order"] = train["rank_group_id"].map(rank_order).astype(int)
        train = (
            train.sort_values(["_rank_order", "rank_group_id", "k_requested"])
            .drop(columns=["_rank_order"])
            .reset_index(drop=True)
        )
        train_shuffle = False
        print("using grouped-rank batches: train rows sorted by shuffled rank_group_id")

    print("train rows:", len(train), "test rows:", len(test))
    print("train prompts:", len(train_groups), "test prompts:", len(test_groups))
    print("train methods:", train["method"].value_counts().to_dict())
    print("train k:", train["k_requested"].value_counts().sort_index().to_dict())

    # Fit prompt embedder only on training prompt texts.
    train_prompt_texts = (
        train[["prompt_hash", "prompt_text"]]
        .drop_duplicates("prompt_hash")["prompt_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    embedder = PromptEmbedder(dim=args.prompt_emb_dim, seed=args.seed)
    embedder.fit(train_prompt_texts)
    embedder.save(out_dir / "prompt_embedder.pkl")

    train = add_prompt_embedding_columns(train, embedder)
    test = add_prompt_embedding_columns(test, embedder)

    cat_cols = DEPLOYABLE_CAT_COLS
    num_cols = DEPLOYABLE_NUM_COLS

    assert "workload" not in cat_cols
    assert "workload" not in num_cols

    pre = FeaturePreprocessor(cat_cols=cat_cols, num_cols=num_cols)
    pre.fit(train)

    xnum_train, xcat_train = pre.transform(train)
    xnum_test, xcat_test = pre.transform(test)

    train_ds = UtilityBlockDataset(
        xnum_train,
        xcat_train,
        train["target_accepted_len"].to_numpy(np.int64),
        train["target_k_actual"].to_numpy(np.int64),
        train["target_log_cost"].to_numpy(np.float32),
        train["target_log_tps"].to_numpy(np.float32),
        train["target_utility"].to_numpy(np.float32),
        train["rank_group_id"].to_numpy(np.int64),
    )

    test_ds = UtilityBlockDataset(
        xnum_test,
        xcat_test,
        test["target_accepted_len"].to_numpy(np.int64),
        test["target_k_actual"].to_numpy(np.int64),
        test["target_log_cost"].to_numpy(np.float32),
        test["target_log_tps"].to_numpy(np.float32),
        test["target_utility"].to_numpy(np.float32),
        test["rank_group_id"].to_numpy(np.int64),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    model = SurvivalUtilityNet(
        num_dim=len(num_cols),
        cat_cardinalities=pre.cat_cardinalities(),
        hidden=args.hidden,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_score = None
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for batch in train_loader:
            (
                x_num,
                x_cat,
                y_accept,
                y_k,
                y_log_cost,
                y_log_tps,
                y_utility,
                rank_group_id,
            ) = batch

            x_num = x_num.to(device, non_blocking=True)
            x_cat = x_cat.to(device, non_blocking=True)
            y_accept = y_accept.to(device, non_blocking=True)
            y_k = y_k.to(device, non_blocking=True)
            y_log_cost = y_log_cost.to(device, non_blocking=True)
            y_log_tps = y_log_tps.to(device, non_blocking=True)
            y_utility = y_utility.to(device, non_blocking=True)
            rank_group_id = rank_group_id.to(device, non_blocking=True)

            hz, pred_log_cost, pred_log_tps, pred_utility = model(x_num, x_cat)

            loss_surv = survival_loss(hz, y_accept, y_k)
            loss_cost = nn.functional.smooth_l1_loss(pred_log_cost, y_log_cost)
            loss_tps = nn.functional.smooth_l1_loss(pred_log_tps, y_log_tps)
            loss_util = nn.functional.smooth_l1_loss(pred_utility, y_utility)
            loss_rank = rank_loss_in_batch(pred_utility, y_utility, rank_group_id)

            loss = (
                args.lambda_survival * loss_surv
                + args.lambda_cost * loss_cost
                + args.lambda_tps * loss_tps
                + args.lambda_utility * loss_util
                + args.lambda_rank * loss_rank
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            losses.append(float(loss.item()))

        metrics, pred = evaluate_prediction(model, test_loader, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = float(np.mean(losses))
        metrics["loss_weights"] = {
            "lambda_survival": args.lambda_survival,
            "lambda_cost": args.lambda_cost,
            "lambda_tps": args.lambda_tps,
            "lambda_utility": args.lambda_utility,
            "lambda_rank": args.lambda_rank,
        }

        print(json.dumps(metrics, indent=2))
        history.append(metrics)

        # Selection-oriented best score.
        if metrics["policy_oracle_fraction"] is not None:
            score = -metrics["policy_oracle_fraction"] + 0.05 * metrics["mae_accepted_len"] + 0.05 * metrics["mae_utility"]
        else:
            score = metrics["mae_accepted_len"] + metrics["mae_utility"]

        if best_score is None or score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            pred.to_csv(out_dir / "test_predictions_best.csv", index=False)

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "model.pt")
    pre.save(out_dir / "preprocessor.pkl")

    metadata = {
        "architecture": "SurvivalUtilityNet",
        "deployable_no_workload": True,
        "uses_workload_as_input": False,
        "uses_prompt_embedding": True,
        "prompt_embedding": embedder.metadata(),
        "prompt_text_map": str(args.prompt_text_map),
        "cost_target_source": cost_source,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_rows": args.max_rows,
        "seed": args.seed,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "lr": args.lr,
        "alpha_accept": args.alpha_accept,
        "alpha_waste": args.alpha_waste,
        "alpha_global": args.alpha_global,
        "alpha_method": args.alpha_method,
        "lambda_survival": args.lambda_survival,
        "lambda_cost": args.lambda_cost,
        "lambda_tps": args.lambda_tps,
        "lambda_utility": args.lambda_utility,
        "lambda_rank": args.lambda_rank,
        "grouped_rank_batches": bool(args.grouped_rank_batches),
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "n_train_prompts": int(len(train_groups)),
        "n_test_prompts": int(len(test_groups)),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "best_score": float(best_score),
        "model_outputs": [
            "hazard_logits",
            "pred_log_cost",
            "pred_log_tps",
            "pred_utility",
        ],
        "policy_score": "pred_utility plus survival/TPS diagnostics",
        "anti_leakage": "Current block acceptance/rejection/tps are target-only. Inputs use causal prefix and previous-block history.",
    }

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    with open(out_dir / "split_groups.json", "w") as f:
        json.dump({
            "train_prompt_hashes": sorted(list(train_groups)),
            "test_prompt_hashes": sorted(list(test_groups)),
        }, f)

    print("Wrote model to:", out_dir)


if __name__ == "__main__":
    main()
