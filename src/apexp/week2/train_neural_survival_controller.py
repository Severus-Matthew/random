#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    BlockDataset,
    SurvivalCostNet,
    survival_loss,
    expected_accepted_len,
    safe_num,
    prompt_group_key,
    add_prompt_embedding_columns,
)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def merge_prompt_text(df, prompt_text_map_path):
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


def prepare_targets(df):
    df = df.copy()

    if "target_accepted_len" not in df.columns:
        raise ValueError("Missing target_accepted_len in online block dataset.")

    df["target_accepted_len"] = (
        safe_num(df["target_accepted_len"])
        .fillna(0)
        .clip(0, MAX_K)
        .astype(int)
    )

    if "target_k_actual" in df.columns:
        k = safe_num(df["target_k_actual"])
    elif "k_actual" in df.columns:
        k = safe_num(df["k_actual"])
    else:
        k = safe_num(df["k_requested"])

    df["target_k_actual"] = k.fillna(df["k_requested"]).clip(1, MAX_K).astype(int)

    cost_source = None

    for c in ["target_latency_s", "latency_s", "block_latency_s", "block_latency_ms"]:
        if c in df.columns:
            raw = safe_num(df[c])
            if c.endswith("_ms"):
                raw = raw / 1000.0
            if raw.notna().sum() > 0 and (raw > 0).mean() > 0.5:
                df["target_cost_s"] = raw.clip(lower=1e-9)
                cost_source = c
                break

    if cost_source is None:
        if "target_tokens_per_sec" in df.columns:
            tps = safe_num(df["target_tokens_per_sec"])
            cost_source = "1/target_tokens_per_sec"
        elif "tokens_per_sec" in df.columns:
            tps = safe_num(df["tokens_per_sec"])
            cost_source = "1/tokens_per_sec"
        else:
            raise ValueError("No usable cost or throughput target found.")

        tps = tps.fillna(0.0).clip(lower=1e-6)
        df["target_cost_s"] = 1.0 / tps

    if "target_tokens_per_sec" in df.columns:
        tps = safe_num(df["target_tokens_per_sec"]).fillna(0.0).clip(lower=1e-6)
    elif "tokens_per_sec" in df.columns:
        tps = safe_num(df["tokens_per_sec"]).fillna(0.0).clip(lower=1e-6)
    else:
        tps = 1.0 / df["target_cost_s"].clip(lower=1e-9)

    df["target_tps_clean"] = tps
    df["target_log_tps"] = np.log1p(tps)
    df["target_log_cost"] = np.log(df["target_cost_s"].clip(lower=1e-9))

    return df, cost_source


def evaluate_prediction(model, loader, device):
    model.eval()
    rows = []

    with torch.no_grad():
        for x_num, x_cat, y_accept, y_k, y_log_cost, y_log_tps in loader:
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            y_accept = y_accept.to(device)
            y_k = y_k.to(device)
            y_log_cost = y_log_cost.to(device)
            y_log_tps = y_log_tps.to(device)

            hz, pred_log_cost, pred_log_tps = model(x_num, x_cat)
            pred_eacc = expected_accepted_len(hz, y_k)

            rows.append(pd.DataFrame({
                "target_accepted_len": y_accept.cpu().numpy(),
                "target_k": y_k.cpu().numpy(),
                "target_log_cost": y_log_cost.cpu().numpy(),
                "target_log_tps": y_log_tps.cpu().numpy(),
                "pred_expected_accepted_len": pred_eacc.cpu().numpy(),
                "pred_log_cost": pred_log_cost.cpu().numpy(),
                "pred_log_tps": pred_log_tps.cpu().numpy(),
            }))

    out = pd.concat(rows, ignore_index=True)

    metrics = {
        "mae_accepted_len": float(np.mean(np.abs(out["target_accepted_len"] - out["pred_expected_accepted_len"]))),
        "mae_log_cost": float(np.mean(np.abs(out["target_log_cost"] - out["pred_log_cost"]))),
        "mae_log_tps": float(np.mean(np.abs(out["target_log_tps"] - out["pred_log_tps"]))),
        "mean_target_accepted_len": float(out["target_accepted_len"].mean()),
        "mean_pred_expected_accepted_len": float(out["pred_expected_accepted_len"].mean()),
        "mean_target_log_cost": float(out["target_log_cost"].mean()),
        "mean_pred_log_cost": float(out["pred_log_cost"].mean()),
    }

    return metrics, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--prompt-text-map", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--variant", choices=["full", "no_cost", "direct"], default="full")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--max-rows", type=int, default=1500000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lambda-cost", type=float, default=0.25)
    ap.add_argument("--prompt-emb-dim", type=int, default=64)
    args = ap.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading:", args.data)
    df = pd.read_csv(args.data, low_memory=False)

    df = df[df["method"].isin(["ngram_sd", "draft_sd", "eagle3"])].copy()
    df = merge_prompt_text(df, args.prompt_text_map)
    df, cost_source = prepare_targets(df)

    if "trace_found" in df.columns:
        print("trace_found counts:")
        print(df["trace_found"].value_counts(dropna=False))

    df["group_key"] = prompt_group_key(df)

    groups = np.array(sorted(df["group_key"].unique()))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(groups)

    n_train = int(0.8 * len(groups))
    train_groups = set(groups[:n_train])
    test_groups = set(groups[n_train:])

    train = df[df["group_key"].isin(train_groups)].copy()
    test = df[df["group_key"].isin(test_groups)].copy()

    if args.max_rows and len(train) > args.max_rows:
        train = train.sample(args.max_rows, random_state=args.seed).copy()

    print("train rows:", len(train), "test rows:", len(test))
    print("train prompts:", len(train_groups), "test prompts:", len(test_groups))

    # Fit prompt embedder only on training prompts.
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

    # Safety assertion: workload must not be used by the deployable controller.
    assert "workload" not in cat_cols
    assert "workload" not in num_cols

    pre = FeaturePreprocessor(cat_cols=cat_cols, num_cols=num_cols)
    pre.fit(train)

    xnum_train, xcat_train = pre.transform(train)
    xnum_test, xcat_test = pre.transform(test)

    train_ds = BlockDataset(
        xnum_train,
        xcat_train,
        train["target_accepted_len"].to_numpy(np.int64),
        train["target_k_actual"].to_numpy(np.int64),
        train["target_log_cost"].to_numpy(np.float32),
        train["target_log_tps"].to_numpy(np.float32),
    )

    test_ds = BlockDataset(
        xnum_test,
        xcat_test,
        test["target_accepted_len"].to_numpy(np.int64),
        test["target_k_actual"].to_numpy(np.int64),
        test["target_log_cost"].to_numpy(np.float32),
        test["target_log_tps"].to_numpy(np.float32),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    model = SurvivalCostNet(
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

        for x_num, x_cat, y_accept, y_k, y_log_cost, y_log_tps in train_loader:
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            y_accept = y_accept.to(device)
            y_k = y_k.to(device)
            y_log_cost = y_log_cost.to(device)
            y_log_tps = y_log_tps.to(device)

            hz, pred_log_cost, pred_log_tps = model(x_num, x_cat)

            if args.variant == "direct":
                loss = nn.functional.mse_loss(pred_log_tps, y_log_tps)
            else:
                loss_surv = survival_loss(hz, y_accept, y_k)
                if args.variant == "no_cost":
                    loss = loss_surv
                else:
                    loss_cost = nn.functional.l1_loss(pred_log_cost, y_log_cost)
                    loss = loss_surv + args.lambda_cost * loss_cost

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            losses.append(float(loss.item()))

        metrics, pred = evaluate_prediction(model, test_loader, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = float(np.mean(losses))
        print(json.dumps(metrics, indent=2))
        history.append(metrics)

        if args.variant == "direct":
            score = metrics["mae_log_tps"]
        elif args.variant == "no_cost":
            score = metrics["mae_accepted_len"]
        else:
            score = metrics["mae_accepted_len"] + metrics["mae_log_cost"]

        if best_score is None or score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            pred.to_csv(out_dir / "test_predictions_best.csv", index=False)

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "model.pt")
    pre.save(out_dir / "preprocessor.pkl")

    metadata = {
        "variant": args.variant,
        "deployable_no_workload": True,
        "uses_workload_as_input": False,
        "uses_prompt_embedding": True,
        "prompt_embedding": embedder.metadata(),
        "prompt_text_map": args.prompt_text_map,
        "cost_target_source": cost_source,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_rows": args.max_rows,
        "seed": args.seed,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "lr": args.lr,
        "lambda_cost": args.lambda_cost,
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "n_train_prompts": int(len(train_groups)),
        "n_test_prompts": int(len(test_groups)),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "trace_found_rate": float(df["trace_found"].mean()) if "trace_found" in df.columns else None,
        "best_score": float(best_score),
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
