"""Run ABR policies on a set of traces and report QoE metrics (README §9).

    python evaluate.py                                # synthetic traces, all baselines
    python evaluate.py --traces data/traces/test --qoe log

Writes per-chunk logs to ``<out>/<policy>.csv`` and a summary to ``<out>/summary.csv``.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from abrsim import ABREnv, load_traces, synthetic_trace
from abrsim import config as C
from baselines import all_baselines

LOG_FIELDS = ["trace", "chunk", "bitrate_kbps", "buffer_s", "rebuffer_s", "delay_s", "throughput_mbps",
              "chunk_size", "reward", "quality", "rebuffer_penalty", "smooth_penalty", "data_penalty"]


def run_policy(policy, env, n_traces):
    """Stream one video per trace; returns per-chunk rows and per-trace summaries."""
    rows, per_trace = [], []
    for t in range(n_traces):
        policy.reset()
        state, done, chunk = env.reset(t), False, 0
        rewards, bitrates, rebuf, switches = [], [], 0.0, 0
        while not done:
            action = policy.select(state, env)
            prev = env.last_action
            state, reward, done, info = env.step(action)
            rows.append({k: info.get(k) for k in LOG_FIELDS} | {"chunk": chunk, "reward": reward})
            rewards.append(reward)
            bitrates.append(info["bitrate_kbps"])
            rebuf += info["rebuffer_s"]
            switches += int(chunk > 0 and action != prev)
            chunk += 1
        per_trace.append({
            "qoe": float(np.mean(rewards)),
            "bitrate_mbps": float(np.mean(bitrates)) / 1000,
            "rebuffer_s": rebuf,
            "rebuffer_ratio": rebuf / (chunk * C.CHUNK_LEN_S),
            "switches": switches,
        })
    return rows, per_trace


def summarize(name, per_trace):
    keys = per_trace[0].keys()
    out = {"policy": name}
    for k in keys:
        vals = np.array([p[k] for p in per_trace])
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std())
    return out


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traces", help="trace directory (default: 20 synthetic traces)")
    p.add_argument("--qoe", default="lin", choices=["lin", "log", "hd"])
    p.add_argument("--data-lambda", type=float, default=0.0)
    p.add_argument("--out", default="results/baselines")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    traces = load_traces(args.traces) if args.traces else [synthetic_trace(seed=1000 + i) for i in range(20)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summaries = []
    for policy in all_baselines():
        env = ABREnv(traces, qoe=args.qoe, data_lambda=args.data_lambda, seed=args.seed)
        rows, per_trace = run_policy(policy, env, len(traces))
        write_csv(out / f"{policy.name}.csv", rows)
        summaries.append(summarize(policy.name, per_trace))

    write_csv(out / "summary.csv", summaries)
    print(f"{len(traces)} traces, QoE_{args.qoe}\n")
    print(f"{'policy':<12}{'QoE/chunk':>14}{'bitrate Mbps':>14}{'rebuffer s':>12}{'switches':>10}")
    for s in sorted(summaries, key=lambda s: -s["qoe_mean"]):
        print(f"{s['policy']:<12}{s['qoe_mean']:>8.3f} ±{s['qoe_std']:<5.2f}{s['bitrate_mbps_mean']:>12.2f}"
              f"{s['rebuffer_s_mean']:>12.2f}{s['switches_mean']:>10.1f}")
    print(f"\nlogs written to {out}/")


if __name__ == "__main__":
    main()
