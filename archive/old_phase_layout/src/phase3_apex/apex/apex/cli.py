"""Command-line utilities for the APEX core package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apex.controllers import CostModel, SurvivalDepthController
from apex.hazard import BetaHazardEstimator
from apex.metrics import summarize_traces
from apex.replay import replay_depth_controller
from apex.rl.bandit import LinUCBBandit, make_trace_context
from apex.rl.rewards import RewardConfig, block_reward
from apex.training.drafter_trainer import build_weighted_labels
from apex.traces import read_jsonl


def _cmd_fit_hazard(args: argparse.Namespace) -> int:
    traces = read_jsonl(args.traces)
    estimator = BetaHazardEstimator(args.max_depth, alpha=args.alpha, beta=args.beta)
    estimator.update_many(traces)
    estimate = estimator.estimate()

    payload = {
        "accept_probs": estimate.accept_probs,
        "hazards": estimate.hazards,
        "survival": estimate.survival,
        "exposures": estimate.exposures,
        "accepts": estimate.accepts,
        "expected_len": estimate.expected_len,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_choose_depth(args: argparse.Namespace) -> int:
    accept_probs = [float(value) for value in args.accept_probs.split(",") if value.strip()]
    controller = SurvivalDepthController(
        [int(value) for value in args.depths.split(",") if value.strip()],
        cost_model=CostModel(
            verifier_ms=args.verifier_ms,
            draft_ms_per_token=args.draft_ms_per_token,
            controller_ms=args.controller_ms,
            fixed_overhead_ms=args.fixed_overhead_ms,
        ),
        min_depth=args.min_depth,
    )
    decision = controller.choose(accept_probs)
    print(
        json.dumps(
            {
                "depth": decision.depth,
                "expected_len": decision.expected_len,
                "utility": decision.utility,
                "accept_probs": decision.accept_probs,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    traces = read_jsonl(args.traces)
    summary = summarize_traces(traces, max_depth=args.max_depth)
    payload = summary.to_dict()
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    traces = read_jsonl(args.traces)
    decisions = replay_depth_controller(
        traces,
        max_depth=args.max_depth,
        candidate_depths=tuple(int(value) for value in args.depths.split(",") if value.strip()),
        alpha=args.alpha,
        beta=args.beta,
        cost_model=CostModel(
            verifier_ms=args.verifier_ms,
            draft_ms_per_token=args.draft_ms_per_token,
            controller_ms=args.controller_ms,
            fixed_overhead_ms=args.fixed_overhead_ms,
        ),
    )
    payload = [decision.to_dict() for decision in decisions]
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_prepare_drafter_data(args: argparse.Namespace) -> int:
    rows = build_weighted_labels(args.examples, rejection_bonus=args.rejection_bonus)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
    return 0


def _cmd_train_bandit(args: argparse.Namespace) -> int:
    traces = read_jsonl(args.traces)
    depths = tuple(int(value) for value in args.depths.split(",") if value.strip())
    if not traces:
        raise ValueError("cannot train bandit on empty traces")
    first_context = make_trace_context(traces[0], max_depth=args.max_depth)
    bandit = LinUCBBandit(depths, context_dim=len(first_context), alpha=args.alpha, ridge=args.ridge)
    reward_config = RewardConfig(
        waste_penalty=args.waste_penalty,
        tail_latency_penalty=args.tail_latency_penalty,
        target_latency_ms=args.target_latency_ms,
        switch_penalty=args.switch_penalty,
    )
    previous_depth: int | None = None
    rewards = []
    for trace in traces:
        context = make_trace_context(trace, max_depth=args.max_depth)
        reward = block_reward(trace, config=reward_config, previous_depth=previous_depth)
        bandit.update(context, trace.draft_len, reward)
        rewards.append(reward)
        previous_depth = trace.draft_len

    payload = {
        "num_traces": len(traces),
        "mean_reward": sum(rewards) / len(rewards),
        "bandit": bandit.state_dict(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="apex", description="APEX speculative-decoding utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser("fit-hazard", help="fit empirical p_j/h_j from block traces")
    fit.add_argument("--traces", required=True, help="JSONL file of BlockTrace records")
    fit.add_argument("--max-depth", type=int, default=16)
    fit.add_argument("--alpha", type=float, default=1.0)
    fit.add_argument("--beta", type=float, default=1.0)
    fit.add_argument("--output", help="optional JSON output path")
    fit.set_defaults(func=_cmd_fit_hazard)

    choose = subparsers.add_parser("choose-depth", help="choose depth from acceptance probabilities")
    choose.add_argument("--accept-probs", required=True, help="comma-separated p_0..p_{k-1}")
    choose.add_argument("--depths", default="1,2,4,8,16")
    choose.add_argument("--min-depth", type=int, default=1)
    choose.add_argument("--verifier-ms", type=float, default=1.0)
    choose.add_argument("--draft-ms-per-token", type=float, default=0.0)
    choose.add_argument("--controller-ms", type=float, default=0.0)
    choose.add_argument("--fixed-overhead-ms", type=float, default=0.0)
    choose.set_defaults(func=_cmd_choose_depth)

    summarize = subparsers.add_parser("summarize", help="summarize speculative block traces")
    summarize.add_argument("--traces", required=True)
    summarize.add_argument("--max-depth", type=int)
    summarize.add_argument("--output")
    summarize.set_defaults(func=_cmd_summarize)

    replay = subparsers.add_parser("replay", help="offline replay of online depth choices")
    replay.add_argument("--traces", required=True)
    replay.add_argument("--max-depth", type=int, default=16)
    replay.add_argument("--depths", default="1,2,4,8,16")
    replay.add_argument("--alpha", type=float, default=1.0)
    replay.add_argument("--beta", type=float, default=1.0)
    replay.add_argument("--verifier-ms", type=float, default=1.0)
    replay.add_argument("--draft-ms-per-token", type=float, default=0.0)
    replay.add_argument("--controller-ms", type=float, default=0.0)
    replay.add_argument("--fixed-overhead-ms", type=float, default=0.0)
    replay.add_argument("--output")
    replay.set_defaults(func=_cmd_replay)

    prepare = subparsers.add_parser("prepare-drafter-data", help="write JSONL rows with acceptance-utility weights")
    prepare.add_argument("--examples", required=True, help="DraftTrainingExample JSONL input")
    prepare.add_argument("--output", required=True, help="weighted JSONL output")
    prepare.add_argument("--rejection-bonus", type=float, default=1.0)
    prepare.set_defaults(func=_cmd_prepare_drafter_data)

    bandit = subparsers.add_parser("train-bandit", help="train a LinUCB depth controller from traces")
    bandit.add_argument("--traces", required=True)
    bandit.add_argument("--output", required=True)
    bandit.add_argument("--max-depth", type=int, default=16)
    bandit.add_argument("--depths", default="1,2,4,8,16")
    bandit.add_argument("--alpha", type=float, default=1.0)
    bandit.add_argument("--ridge", type=float, default=1.0)
    bandit.add_argument("--waste-penalty", type=float, default=0.05)
    bandit.add_argument("--tail-latency-penalty", type=float, default=0.0)
    bandit.add_argument("--target-latency-ms", type=float)
    bandit.add_argument("--switch-penalty", type=float, default=0.0)
    bandit.set_defaults(func=_cmd_train_bandit)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
