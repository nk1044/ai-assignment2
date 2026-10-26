"""Run ABR policies on a set of traces and report QoE metrics (README §9).

    python evaluate.py                                # HSDPA test traces, all baselines
    python evaluate.py --traces fcc/test --qoe log    # dataset name or a directory path
    python evaluate.py --traces synthetic
    python evaluate.py --agent PPO=runs/ppo_asym/best.pt    # add trained agent(s) to the comparison
    python evaluate.py --oracle                              # also run the offline optimal (~1.5 min)

Test-time simulation is deterministic (every video starts at the beginning of its trace, no delay
noise — as in Pensieve's test env), so every policy sees exactly the same network. Writes per-chunk logs to ``<out>/<policy>.csv``, per-trace metrics to
``<out>/<policy>_traces.csv`` and a summary to ``<out>/summary.csv``.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from abrsim import ABREnv, load_dataset, synthetic_trace
from abrsim import config as C
from baselines import OfflineOptimal, all_baselines

LOG_FIELDS = ["trace", "chunk", "wall_time_s", "trace_time_s", "bitrate_kbps", "buffer_s", "rebuffer_s",
              "delay_s", "sleep_s", "throughput_mbps", "chunk_size", "reward", "quality", "rebuffer_penalty",
              "smooth_penalty", "data_penalty"]


def run_policy(policy, env, n_traces):
    """Stream one video per trace; returns per-chunk rows and per-trace summaries.

    Following Pensieve/MPC evaluation practice, the first chunk's wait is reported separately as
    startup delay and per-trace metrics are computed over chunks 1..N-1 (playback phase).
    """
    rows, per_trace = [], []
    for t in range(n_traces):
        policy.reset()
        state, done, chunk, wall = env.reset(t), False, 0, 0.0
        infos, rewards = [], []
        while not done:
            action = policy.select(state, env)
            state, reward, done, info = env.step(action)
            wall += info["delay_s"] + info["sleep_s"]
            rows.append({k: info.get(k) for k in LOG_FIELDS}
                        | {"chunk": chunk, "reward": reward, "wall_time_s": wall})
            infos.append(info)
            rewards.append(reward)
            chunk += 1
        play, bitrates = infos[1:], [i["bitrate_kbps"] for i in infos]
        rebuf = sum(i["rebuffer_s"] for i in play)
        per_trace.append({
            "trace": infos[0]["trace"],
            "qoe": float(np.mean(rewards[1:])),
            "bitrate_mbps": float(np.mean(bitrates[1:])) / 1000,
            "rebuffer_s": rebuf,
            "rebuffer_ratio": rebuf / (len(play) * C.CHUNK_LEN_S),
            "switches": int(np.sum(np.diff(bitrates) != 0)),
            "startup_s": infos[0]["rebuffer_s"],
            "total_qoe": float(np.sum(rewards)),
        })
    return rows, per_trace


def summarize(name, per_trace):
    keys = [k for k in per_trace[0] if k != "trace"]
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
    p.add_argument("--traces", default="hsdpa/test",
                   help="dataset (hsdpa/test, fcc/test, test, ...), a directory, or 'synthetic'")
    p.add_argument("--qoe", default="lin", choices=["lin", "log", "hd"])
    p.add_argument("--data-lambda", type=float, default=0.0)
    p.add_argument("--out", help="output dir (default: results/baselines/<traces>_<qoe>)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--agent", action="append", default=[], metavar="NAME=CKPT",
                   help="trained agent checkpoint to evaluate (repeatable)")
    p.add_argument("--no-baselines", action="store_true")
    p.add_argument("--oracle", action="store_true", help="also run the offline-optimal upper bound")
    args = p.parse_args()

    if args.traces == "synthetic":
        traces = [synthetic_trace(seed=1000 + i) for i in range(20)]
    else:
        traces = load_dataset(args.traces)
    out = Path(args.out or f"results/baselines/{args.traces.replace('/', '_')}_{args.qoe}")
    out.mkdir(parents=True, exist_ok=True)

    summaries = []
    policies = [] if args.no_baselines else all_baselines()
    if args.oracle:
        policies.append(OfflineOptimal())
    for spec in args.agent:
        from agents import RLPolicy                      # imported lazily: baselines need no torch
        name, _, path = spec.rpartition("=")
        policies.append(RLPolicy.from_checkpoint(path, name or "PPO"))
    for policy in policies:
        env = ABREnv(traces, qoe=args.qoe, data_lambda=args.data_lambda, seed=args.seed, noise=False,
                     random_start=False)
        rows, per_trace = run_policy(policy, env, len(traces))
        write_csv(out / f"{policy.name}.csv", rows)
        write_csv(out / f"{policy.name}_traces.csv", per_trace)
        summaries.append(summarize(policy.name, per_trace))

    write_csv(out / "summary.csv", summaries)
    print(f"{args.traces}: {len(traces)} traces, QoE_{args.qoe}\n")
    print(f"{'policy':<12}{'QoE/chunk':>14}{'bitrate Mbps':>14}{'rebuffer s':>12}{'switches':>10}{'startup s':>11}")
    for s in sorted(summaries, key=lambda s: -s["qoe_mean"]):
        print(f"{s['policy']:<12}{s['qoe_mean']:>8.3f} ±{s['qoe_std']:<5.2f}{s['bitrate_mbps_mean']:>12.2f}"
              f"{s['rebuffer_s_mean']:>12.2f}{s['switches_mean']:>10.1f}{s['startup_s_mean']:>11.2f}")
    print(f"\nlogs written to {out}/")


if __name__ == "__main__":
    main()
