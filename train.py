"""Train the A2C ABR agent (README §7).

    python train.py --run-name a2c_lin --iters 30000
    python train.py --run-name a2c_datasaver --lambda-range 0 2      # preference-conditioned (U2)

Outputs in ``runs/<run-name>/``: config.json, log.csv (one row per validation), ckpt_<iter>.pt
(kept for the learning timelapse), best.pt (highest validation QoE).
Validation uses a held-out slice of the *training* traces; the test traces are never touched.
"""

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from abrsim import ABREnv, load_dataset
from agents import A2CAgent, A2CConfig, RLPolicy, RolloutCollector
from evaluate import run_policy


def split_validation(traces, frac, seed):
    order = np.random.default_rng(seed).permutation(len(traces))
    n_val = max(1, round(len(traces) * frac))
    return [traces[i] for i in order[n_val:]], [traces[i] for i in order[:n_val]]


def validate(agent, traces, qoe, data_lambda, seed):
    env = ABREnv(traces, qoe=qoe, data_lambda=data_lambda, seed=seed)
    _, per_trace = run_policy(RLPolicy(agent, data_lambda=data_lambda), env, len(traces))
    return {k: float(np.mean([p[k] for p in per_trace])) for k in ("qoe", "bitrate_mbps", "rebuffer_s", "switches")}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", default="a2c_lin")
    p.add_argument("--train-set", default="train", help="dataset spec, e.g. train, hsdpa/train, fcc/train")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--qoe", default="lin", choices=["lin", "log", "hd"])
    p.add_argument("--data-lambda", type=float, default=0.0, help="fixed data-cost weight")
    p.add_argument("--lambda-range", type=float, nargs=2, help="sample lambda per video and condition on it")
    p.add_argument("--iters", type=int, default=30_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="runs")
    for field, default in asdict(A2CConfig()).items():
        if field == "extra_dim":
            continue
        kind = (lambda v: v.lower() in ("1", "true", "yes")) if isinstance(default, bool) else type(default)
        p.add_argument(f"--{field.replace('_', '-')}", type=kind, default=default)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    run_dir = Path(args.out) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_traces, val_traces = split_validation(load_dataset(args.train_set), args.val_frac, args.seed)
    cfg_fields = {k: getattr(args, k) for k in asdict(A2CConfig()) if k != "extra_dim"}
    cfg = A2CConfig(**cfg_fields, extra_dim=1 if args.lambda_range else 0)
    agent = A2CAgent(cfg, args.device)
    collector = RolloutCollector(train_traces, args.n_envs, args.qoe, args.data_lambda,
                                 tuple(args.lambda_range) if args.lambda_range else None, args.seed)
    (run_dir / "config.json").write_text(json.dumps(
        vars(args) | {"n_train_traces": len(train_traces), "val_traces": [t.name for t in val_traces]}, indent=2))

    log_path = run_dir / "log.csv"
    best_qoe, log_file, writer = -np.inf, open(log_path, "w", newline=""), None
    start = time.time()
    for it in range(1, args.iters + 1):
        rollout = collector.collect(agent)
        stats = agent.update(rollout)

        if it % args.ckpt_every == 0 or it == 1:
            agent.save(run_dir / f"ckpt_{it:06d}.pt")
        if it % args.eval_every == 0 or it == args.iters:
            val = validate(agent, val_traces, args.qoe, args.data_lambda, args.seed)
            row = {"iter": it, "time_s": round(time.time() - start, 1),
                   "train_reward": float(rollout["rewards"][1:].mean()),
                   **{f"val_{k}": v for k, v in val.items()}, **stats}
            if writer is None:
                writer = csv.DictWriter(log_file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            log_file.flush()
            if val["qoe"] > best_qoe:
                best_qoe = val["qoe"]
                agent.save(run_dir / "best.pt", val_qoe=best_qoe)
            print(f"iter {it:6d} | train r {row['train_reward']:7.3f} | val QoE {val['qoe']:7.3f} "
                  f"(best {best_qoe:.3f}) | H {stats['entropy']:.2f} beta {stats['entropy_weight']:.2f} "
                  f"| {row['time_s']:.0f}s", flush=True)

    log_file.close()
    agent.save(run_dir / "final.pt")
    return run_dir


if __name__ == "__main__":
    main()
