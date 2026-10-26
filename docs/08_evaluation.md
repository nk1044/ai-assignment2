# 08 — Evaluation, Metrics, Figures, Results

## 8.1 Test protocol (`evaluate.py`)

- **Deterministic environment:** `ABREnv(..., noise=False, random_start=False)`. Every video starts at
  the beginning of its trace, and there is no delay noise. This is identical to Pensieve's test simulator.
- **Why deterministic?**
  1. Every algorithm faces *exactly* the same network, so differences come only from decisions.
  2. The oracle's plan replays exactly.
  3. Random start offsets made the 12-trace FCC average swing from −0.3 to +0.9, depending only on
     whether the 3-minute window overlapped an outage. That is luck, not skill.
- **Startup delay is excluded from QoE.** The first chunk always stalls (empty buffer). Like Pensieve
  and MPC, its wait is reported as `startup_s` and QoE is averaged over chunks 1–47.
  `total_qoe` (all 48 rewards summed) is also stored.
- `run_policy(policy, env, n)`: for each trace, `policy.reset()`, `env.reset(t)`, then loop
  `select → step`, logging every chunk.

## 8.2 Metrics (per trace, then mean ± std over traces)

| Metric | Definition |
|---|---|
| `qoe` | mean per-chunk reward over chunks 1..47 (**main metric**) |
| `bitrate_mbps` | mean chosen bitrate over chunks 1..47 |
| `rebuffer_s` | total stall seconds over chunks 1..47 |
| `rebuffer_ratio` | rebuffer_s / (47 × 4 s) |
| `switches` | number of bitrate changes between consecutive chunks |
| `startup_s` | stall time of chunk 0 |
| `total_qoe` | sum of all 48 rewards (used by the oracle-bound test) |

## 8.3 Results so far

**HSDPA test (142 unseen traces), QoE_lin, first training run (A2C v1).** Every run, including PPO, is
compared in `experiments/README.md`.

| Policy | QoE/chunk | Bitrate (Mbps) | Stall (s/video) | Switches | Startup (s) |
|---|---|---|---|---|---|
| Offline optimal | 1.145 | 1.18 | 0.01 | 2.4 | 1.70 |
| **A2C (ours)** | **0.934** | 1.10 | **0.17** | 12.1 | 1.70 |
| RobustMPC | 0.907 | 1.13 | 0.91 | 9.5 | 4.06 |
| BOLA | 0.771 | 0.94 | 0.01 | 13.8 | 1.70 |
| RB | 0.745 | 0.92 | 1.13 | 6.5 | 4.06 |
| BB | 0.626 | 1.14 | 1.80 | 25.3 | 1.70 |
| FastMPC | 0.431 | 1.20 | 6.82 | 10.6 | 4.06 |

How to read it:
- A2C has the best QoE of all real algorithms. It gets almost RobustMPC's bitrate with **5× fewer stalls**.
- It has learned to be cautious when it matters, without being as timid as BOLA.
- It switches more than RobustMPC (12 vs 9.5). That is the next thing to improve.
- It still sits 18 % below the optimum: room for better training (Ch. 07 §7.6).
- A2C's startup is 1.70 s because it starts at 300 kbps. MPC/RB start at the 750k default → 4.06 s.

**Stress-test scenarios (baselines only so far):**

| Policy | Metro tunnel | Stadium | Highway | Flaky Wi-Fi | Rural | All |
|---|---|---|---|---|---|---|
| Offline optimal | 2.11 | 1.84 | 1.96 | 2.03 | 0.65 | 1.72 |
| BOLA | 1.27 | 1.42 | 1.24 | 1.24 | 0.40 | 1.11 |
| RB | 0.37 | 1.64 | 1.58 | 1.19 | 0.44 | 1.04 |
| RobustMPC | −1.79 | 1.64 | 1.72 | 1.56 | 0.56 | 0.74 |
| BB | −1.12 | 1.42 | 1.30 | 1.11 | 0.30 | 0.60 |

No heuristic wins every scenario. RobustMPC is the best on real traces but the worst in tunnels.

## 8.4 Figures (`plots/make_figures.py` → `plots/figures/`)

The style is fixed: each algorithm always has the same colour (PPO blue, A2C red, RobustMPC orange,
BOLA aqua, RB yellow, BB pink, FastMPC green, fixed violet). The oracle is grey and hatched/dashed, as a reference.
Each figure is written as PNG (slides) and PDF (LaTeX). PDF timestamps are stripped so regenerating does
not create git noise.

| File | What it shows | Talking point |
|---|---|---|
| `data_traces` | 3 HSDPA + 3 FCC traces against the 6 bitrate levels | bandwidth crosses several levels within seconds |
| `g1_training` | validation QoE vs iteration (+ RobustMPC/optimum lines) and entropy | learning happens once entropy drops |
| `g2_qoe_bars` | mean QoE ± s.e. per algorithm, HSDPA and FCC | headline comparison |
| `g3_breakdown_*` | per-chunk quality (up) vs stall and smoothness penalties (down), net QoE diamond | *why* each algorithm wins or loses |
| `g4_qoe_cdf` | CDF of per-trace QoE | robustness across all traces, not just the average |
| `g5_timeseries_hsdpa_test` | for the trace with the largest QoE spread: bandwidth, chosen bitrate, buffer, stall markers | the optimum builds a 50 s buffer just before a collapse; baselines stall repeatedly |
| `g6_tradeoff_*` | average bitrate vs % time stalled | the quality–stall frontier |
| `g7_scenarios` | per-scenario bandwidth sparkline + QoE bars | the tunnel story |

To add the agent: run the three `evaluate.py --oracle --agent PPO=...` commands (`commands.md`), then
`make_figures.py`. Every chart picks up `PPO_traces.csv` (and `A2C_traces.csv`, if present) automatically.
