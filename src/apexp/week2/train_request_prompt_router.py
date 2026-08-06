from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.apexp.week2.prompt_embedder import PromptEmbedder

def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

CAT_COLS = [
    "method",
    "draft_model",
    "eagle3_model",
    "ngram_lookup_min",
    "ngram_lookup_max",
]

BASE_NUM_COLS = [
    "k_requested",
    "temperature",
    "prompt_token_len",
    "max_prompt_tokens",
    "num_turns",
]
ACTION_COLS = [
    "method",
    "k_requested",
    "draft_model",
    "eagle3_model",
    "ngram_lookup_min",
    "ngram_lookup_max",
    "temperature",
    "max_prompt_tokens",
    "num_turns",
]

ACTION_PRIOR_COLS = [
    "action_prior_mean_tps",
    "action_prior_std_tps",
    "action_prior_n_prompts",
    "action_prior_rank",
]
PROMPT_STAT_COLS = [
    "prompt_char_len_local",
    "prompt_num_lines",
    "prompt_avg_line_len",
    "prompt_digit_frac",
    "prompt_alpha_frac",
    "prompt_space_frac",
    "prompt_symbol_frac",
    "prompt_code_symbol_frac",
    "prompt_math_symbol_frac",
    "prompt_newline_frac",

    "prompt_word_count",
    "prompt_unique_word_frac",
    "prompt_avg_word_len",
    "prompt_backtick_count",
    "prompt_brace_count",
    "prompt_paren_count",
    "prompt_indent_frac",
    "prompt_code_keyword_count",
    "prompt_math_keyword_count",
]

def add_action_priors(train, test):
    cols = [c for c in ACTION_COLS if c in train.columns]

    perf = (
        train.groupby(cols, dropna=False)
        .agg(
            action_prior_mean_tps=("actual_tps", "mean"),
            action_prior_std_tps=("actual_tps", "std"),
            action_prior_n_prompts=("prompt_hash", "nunique"),
        )
        .reset_index()
    )

    perf["action_prior_std_tps"] = perf["action_prior_std_tps"].fillna(0.0)
    perf["action_prior_rank"] = perf["action_prior_mean_tps"].rank(
        ascending=False,
        method="dense",
    )

    train = train.merge(perf, on=cols, how="left")
    test = test.merge(perf, on=cols, how="left")
    

    for c in ACTION_PRIOR_COLS:
        train[c] = safe_num(train[c]).fillna(0.0)
        test[c] = safe_num(test[c]).fillna(0.0)

    return train, test, perf

def safe_num(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def merge_prompt_text(df, prompt_text_map_path):
    mp = pd.read_csv(prompt_text_map_path)
    df = df.copy()
    df["prompt_hash"] = df["prompt_hash"].astype(str)
    mp["prompt_hash"] = mp["prompt_hash"].astype(str)

    before = len(df)
    df = df.merge(mp[["prompt_hash", "prompt_text"]], on="prompt_hash", how="left")
    assert len(df) == before

    rate = float(df["prompt_text"].notna().mean())
    print(f"prompt_text row match rate: {rate:.4f}")
    if rate < 0.90:
        raise RuntimeError(f"prompt_text match too low: {rate:.4f}")

    df["prompt_text"] = df["prompt_text"].fillna("")
    return df


def add_prompt_stats(df):
    df = df.copy()
    texts = df["prompt_text"].fillna("").astype(str)

    def frac(text, pred):
        if len(text) == 0:
            return 0.0
        return sum(1 for ch in text if pred(ch)) / len(text)

    code_chars = set("{}[]();:._#<>/\\=*+-")
    math_chars = set("+-=*/^√∑∫≤≥≠≈πθλμσ")
    code_keywords = {
        "def", "class", "return", "import", "for", "while", "if", "else",
        "elif", "try", "except", "public", "private", "module", "assign",
        "always", "wire", "reg", "input", "output"
    }

    math_keywords = {
        "prove", "solve", "equation", "integral", "derivative", "matrix",
        "probability", "expectation", "variance", "theorem", "lemma",
        "constraint", "optimization"
    }

    def words(x):
        return re.findall(r"[A-Za-z_][A-Za-z_0-9]*", x.lower())

    import re

    ws = texts.map(words)
    df["prompt_word_count"] = ws.map(len)
    df["prompt_unique_word_frac"] = ws.map(
        lambda w: len(set(w)) / max(1, len(w))
    )
    df["prompt_avg_word_len"] = ws.map(
        lambda w: sum(len(z) for z in w) / max(1, len(w))
    )
    df["prompt_backtick_count"] = texts.str.count("`")
    df["prompt_brace_count"] = texts.str.count(r"\{|\}")
    df["prompt_paren_count"] = texts.str.count(r"\(|\)")
    df["prompt_indent_frac"] = texts.map(
        lambda x: sum(1 for line in x.splitlines() if line.startswith((" ", "\t"))) / max(1, len(x.splitlines()))
    )
    df["prompt_code_keyword_count"] = ws.map(
        lambda w: sum(1 for z in w if z in code_keywords)
    )
    df["prompt_math_keyword_count"] = ws.map(
        lambda w: sum(1 for z in w if z in math_keywords)
    )


    df["prompt_char_len_local"] = texts.str.len()
    df["prompt_num_lines"] = texts.map(lambda x: max(1, x.count("\n") + 1))
    df["prompt_avg_line_len"] = df["prompt_char_len_local"] / df["prompt_num_lines"].clip(lower=1)
    df["prompt_digit_frac"] = texts.map(lambda x: frac(x, lambda c: c.isdigit()))
    df["prompt_alpha_frac"] = texts.map(lambda x: frac(x, lambda c: c.isalpha()))
    df["prompt_space_frac"] = texts.map(lambda x: frac(x, lambda c: c.isspace()))
    df["prompt_symbol_frac"] = texts.map(lambda x: frac(x, lambda c: not c.isalnum() and not c.isspace()))
    df["prompt_code_symbol_frac"] = texts.map(lambda x: frac(x, lambda c: c in code_chars))
    df["prompt_math_symbol_frac"] = texts.map(lambda x: frac(x, lambda c: c in math_chars))
    df["prompt_newline_frac"] = texts.map(lambda x: frac(x, lambda c: c == "\n"))

    return df



def build_candidate_table(df):
    df = df.copy()

    if "target_tokens_per_sec" in df.columns:
        df["actual_tps"] = safe_num(df["target_tokens_per_sec"])
    elif "tokens_per_sec" in df.columns:
        df["actual_tps"] = safe_num(df["tokens_per_sec"])
    else:
        raise RuntimeError("No target_tokens_per_sec/tokens_per_sec column found.")

    df["actual_tps"] = df["actual_tps"].fillna(0.0)
    df["k_requested"] = safe_num(df["k_requested"]).fillna(1).astype(int)

    for c in CAT_COLS:
        if c not in df.columns:
            df[c] = "NA"
        df[c] = df[c].fillna("NA").astype(str)

    for c in BASE_NUM_COLS:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = safe_num(df[c]).fillna(0.0)

    group_cols = [
        "prompt_hash",
        "prompt_id",
        "prompt_text",
        "workload",
        "method",
        "k_requested",
        "draft_model",
        "eagle3_model",
        "ngram_lookup_min",
        "ngram_lookup_max",
        "temperature",
        "prompt_token_len",
        "max_prompt_tokens",
        "num_turns",
    ]
    group_cols = [c for c in group_cols if c in df.columns]

    cand = (
        df.groupby(group_cols, dropna=False, as_index=False)["actual_tps"]
        .mean()
    )

    cand = add_prompt_stats(cand)
    return cand

ACTION_COLS = [
    "method",
    "k_requested",
    "draft_model",
    "eagle3_model",
    "ngram_lookup_min",
    "ngram_lookup_max",
    "temperature",
    "max_prompt_tokens",
    "num_turns",
]


def one_per_prompt(df, score_col="actual_tps"):
    if df.empty:
        return df.copy()
    idx = df.groupby("prompt_hash")[score_col].idxmax()
    return df.loc[idx].copy()


def choose_by_score(cand, score_col):
    return one_per_prompt(cand, score_col=score_col)


def choose_oracle(cand):
    return one_per_prompt(cand, score_col="actual_tps")


def choose_fixed(cand, action):
    q = cand.copy()
    for c, v in action.items():
        if c in q.columns:
            q = q[q[c].astype(str) == str(v)]
    return one_per_prompt(q, score_col="actual_tps")


def best_global_fixed_action(train):
    cols = [c for c in ACTION_COLS if c in train.columns]

    perf = (
        train.groupby(cols, dropna=False)
        .agg(mean_tps=("actual_tps", "mean"), n_prompts=("prompt_hash", "nunique"))
        .reset_index()
    )

    # Require broad support so we don't pick an action available for only a few prompts.
    total_prompts = train["prompt_hash"].nunique()
    perf = perf[perf["n_prompts"] >= 0.90 * total_prompts].copy()

    if perf.empty:
        raise RuntimeError("No global fixed action has >=90% train prompt coverage.")

    best = perf.sort_values("mean_tps", ascending=False).iloc[0]
    return {c: best[c] for c in cols}, perf





def summarize(name, selected, oracle, ar_tps):
    if selected.empty:
        return {
            "policy": name,
            "n_requests": 0,
            "mean_tps": np.nan,
            "speedup_vs_ar": np.nan,
            "oracle_fraction": np.nan,
        }

    m = selected[["prompt_hash", "actual_tps"]].merge(
        oracle[["prompt_hash", "actual_tps"]].rename(columns={"actual_tps": "oracle_tps"}),
        on="prompt_hash",
        how="inner",
    )

    mean_tps = float(m["actual_tps"].mean())
    oracle_tps = float(m["oracle_tps"].mean())

    return {
        "policy": name,
        "n_requests": int(len(m)),
        "mean_tps": mean_tps,
        "speedup_vs_ar": mean_tps / ar_tps,
        "oracle_fraction": mean_tps / oracle_tps if oracle_tps > 0 else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--prompt-text-map", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ar-tps", type=float, default=63.726752)
    ap.add_argument("--prompt-emb-dim", type=int, default=64)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data, low_memory=False)
    df = df[df["method"].isin(["ngram_sd", "draft_sd", "eagle3"])].copy()
    df = merge_prompt_text(df, args.prompt_text_map)

    cand = build_candidate_table(df)
    cand["prompt_hash"] = cand["prompt_hash"].astype(str)

    groups = np.array(sorted(cand["prompt_hash"].unique()))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(groups)

    n_train = int(0.8 * len(groups))
    train_groups = set(groups[:n_train])
    test_groups = set(groups[n_train:])

    train = cand[cand["prompt_hash"].isin(train_groups)].copy()
    test = cand[cand["prompt_hash"].isin(test_groups)].copy()
    train, test, action_prior_table = add_action_priors(train, test)
    action_prior_table.to_csv(out_dir / "action_prior_train_table.csv", index=False)

    print("candidate train:", train.shape)
    print("candidate test:", test.shape)
    print("train prompts:", len(train_groups), "test prompts:", len(test_groups))

    embedder = PromptEmbedder(dim=args.prompt_emb_dim, seed=args.seed)
    train_prompt_texts = (
        train[["prompt_hash", "prompt_text"]]
        .drop_duplicates("prompt_hash")["prompt_text"]
        .astype(str)
        .tolist()
    )
    embedder.fit(train_prompt_texts)
    embedder.save(out_dir / "prompt_embedder.pkl")

    for split_name, part in [("train", train), ("test", test)]:
        Z = embedder.transform(part["prompt_text"].astype(str).tolist())
        for i in range(args.prompt_emb_dim):
            part[f"prompt_emb_{i}"] = Z[:, i]
        if split_name == "train":
            train = part
        else:
            test = part

    emb_cols = [f"prompt_emb_{i}" for i in range(args.prompt_emb_dim)]
    # num_cols = BASE_NUM_COLS + PROMPT_STAT_COLS + emb_cols
    num_cols = BASE_NUM_COLS + PROMPT_STAT_COLS + ACTION_PRIOR_COLS + emb_cols
    cat_cols = CAT_COLS

    for c in cat_cols:
        train[c] = train[c].fillna("NA").astype(str)
        test[c] = test[c].fillna("NA").astype(str)

    for c in num_cols:
        train[c] = safe_num(train[c]).fillna(0.0)
        test[c] = safe_num(test[c]).fillna(0.0)

    X_train = train[cat_cols + num_cols]
    y_train = train["actual_tps"].to_numpy()

    X_test = test[cat_cols + num_cols]
    y_test = test["actual_tps"].to_numpy()

    pre = ColumnTransformer(
        transformers=[
            ("cat", make_onehot(), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
    )
    model = ExtraTreesRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        random_state=args.seed,
        n_jobs=-1,
    )

    pipe = Pipeline([
        ("pre", pre),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)

    test["pred_tps"] = pipe.predict(X_test)
    train["pred_tps"] = pipe.predict(X_train)

    mae = float(mean_absolute_error(y_test, test["pred_tps"]))
    r2 = float(r2_score(y_test, test["pred_tps"]))

    oracle = choose_oracle(test)
    learned = choose_by_score(test, "pred_tps")

    best_action, fixed_train_table = best_global_fixed_action(train)
    best_global_fixed = choose_fixed(test, best_action)

    fixed_train_table.to_csv(out_dir / "fixed_action_train_table.csv", index=False)

    # policies = [
    #     ("oracle", oracle),
    #     ("request_prompt_router_no_workload", learned),
    #     ("best_global_fixed_train_chosen", best_global_fixed),
    #     ("fixed_ngram_k16", choose_fixed(test, {"method": "ngram_sd", "k_requested": 16})),
    #     ("fixed_eagle3_k4", choose_fixed(test, {"method": "eagle3", "k_requested": 4})),
    #     ("fixed_draft_k4", choose_fixed(test, {"method": "draft_sd", "k_requested": 4})),
    # ]
    policies = [
        ("oracle", oracle),
        ("request_prompt_router_no_workload", learned),
        ("best_global_fixed_train_chosen", best_global_fixed),
        ("fixed_ngram_k16", choose_fixed(test, {"method": "ngram_sd", "k_requested": 16})),
        ("fixed_ngram_k8", choose_fixed(test, {"method": "ngram_sd", "k_requested": 8})),
        ("fixed_eagle3_k8", choose_fixed(test, {"method": "eagle3", "k_requested": 8})),
        ("fixed_eagle3_k4", choose_fixed(test, {"method": "eagle3", "k_requested": 4})),
        ("fixed_draft_k4", choose_fixed(test, {"method": "draft_sd", "k_requested": 4})),
    ]

    summary = pd.DataFrame([
        summarize(name, sel, oracle, args.ar_tps)
        for name, sel in policies
    ]).sort_values("mean_tps", ascending=False)

    action_dist = (
        learned.groupby(["method", "k_requested"])
        .size()
        .reset_index(name="count")
    )
    action_dist["fraction"] = action_dist["count"] / action_dist["count"].sum()
    action_dist = action_dist.sort_values("fraction", ascending=False)

    metadata = {
        "model": "ExtraTreesRegressor request-level prompt router",
        "uses_workload_as_input": False,
        "uses_prompt_embedding": True,
        "prompt_embedding": embedder.metadata(),
        "train_candidates": int(len(train)),
        "test_candidates": int(len(test)),
        "train_prompts": int(len(train_groups)),
        "test_prompts": int(len(test_groups)),
        "mae_tps": mae,
        "r2_tps": r2,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "target": "actual_tps/request-level",
        "best_global_fixed_action_train_chosen": {
            k: str(v) for k, v in best_action.items()
        },
        "split": {
            "train_prompt_hashes": sorted(map(str, train_groups)),
            "test_prompt_hashes": sorted(map(str, test_groups)),
        },

    }

    with open(out_dir / "request_prompt_router.pkl", "wb") as f:
        pickle.dump(pipe, f)

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    train.to_csv(out_dir / "train_candidates_scored.csv", index=False)
    test.to_csv(out_dir / "test_candidates_scored.csv", index=False)
    summary.to_csv(out_dir / "policy_summary.csv", index=False)
    action_dist.to_csv(out_dir / "action_distribution.csv", index=False)

    print("\n=== REQUEST ROUTER METRICS ===")
    print(json.dumps(metadata, indent=2))

    print("\n=== POLICY SUMMARY ===")
    print(summary.to_string(index=False))

    print("\n=== ACTION DISTRIBUTION ===")
    print(action_dist.to_string(index=False))

    print("\nWrote:", out_dir)


if __name__ == "__main__":
    main()
