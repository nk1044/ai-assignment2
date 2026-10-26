# Project Notes — Adaptive Bitrate Streaming with Deep RL

Study notes for explaining, defending, and rebuilding this project. They assume you know RL basics
(MDP, policy, value function, policy gradient). Read in numeric order.

| # | Chapter | Read it to be able to… |
|---|---|---|
| 01 | [Problem](01_problem.md) | state the problem, the QoE trade-off, and why it is hard |
| 02 | [Project structure](02_project_structure.md) | navigate every folder/file and explain how data flows end-to-end |
| 03 | [MDP formulation](03_mdp_formulation.md) | define S, A, P, R, γ precisely and justify every design choice |
| 04 | [Simulator](04_simulator.md) | explain (and re-implement) the trace-driven streaming simulator line by line |
| 05 | [Data](05_data.md) | explain where every trace/video file comes from, its format, and the splits |
| 06 | [Baselines and the oracle](06_baselines.md) | explain RB, BB, BOLA, MPC, RobustMPC and the offline-optimal DP |
| 07 | [The RL agent](07_agent.md) | explain PPO, the asymmetric (privileged) critic, GAE, the training loop, the A2C history |
| 08 | [Evaluation and results](08_evaluation.md) | explain the test protocol, metrics, every figure, and the numbers |
| 09 | [Technology choices](09_tech_choices.md) | defend every tool/algorithm choice against alternatives |
| 10 | [Rebuild guide](10_rebuild_guide.md) | rewrite the whole project from scratch in the right order |
| 11 | [Viva Q&A](11_viva_qa.md) | answer the questions examiners are likely to ask |
| 12 | [Limitations and extensions](12_limitations_extensions.md) | discuss weaknesses honestly and what the bonus parts add |

## The project in one paragraph

A video player must choose, for every 4-second chunk of a video, one of 6 qualities (300–4300 kbps),
while the network bandwidth fluctuates unpredictably. Too high a quality → the buffer drains and the
video stalls. Too low → a blurry picture. Switching too often → visible flicker. We model this as an MDP:
- **State:** a 6×8 matrix holding the recent throughput and download-time history, the buffer level, the
  next chunk's size at each quality, the last quality, and the chunks remaining.
- **Action:** one of 6 bitrates.
- **Reward:** the per-chunk QoE, `q(R) − 4.3·stall − |Δq|`.

We build a trace-driven simulator that replays **real recorded bandwidth traces** (Norway 3G, FCC
broadband) together with the real chunk sizes of a test video. We then train a **PPO** agent with
16 parallel simulators and GAE. Its **asymmetric critic** sees the true future bandwidth during training
only; the deployed actor sees just the normal observation. The first agents were A2C with a symmetric
critic, and they are kept as an experiment history. It is compared against 5 classic ABR algorithms and a
dynamic-programming **offline optimal** that knows the future. On 142 unseen 3G traces:
- the best classic algorithm (RobustMPC) scores 0.907 QoE per chunk;
- the optimum scores 1.145;
- the final PPO agent scores **0.971**, closing 27 % of that gap. It beats every baseline on every test set.

## Key numbers to remember

| Item | Value |
|---|---|
| Bitrates (actions) | 300, 750, 1200, 1850, 2850, 4300 kbps (6) |
| Chunk length / chunks per video | 4 s / 48 (≈ 3.2 min) |
| State | 6 × 8 matrix (25 meaningful numbers) |
| Reward | `q(R_t) − 4.3·T_t − \|q(R_t) − q(R_{t−1})\|` (q in Mbps, T = stall seconds) |
| Buffer cap | 60 s |
| Train / validation / test traces | 115 → 103 train + 12 val / 142 HSDPA + 12 FCC test |
| Network size | actor ≈ 266 k parameters, privileged critic ≈ 348 k |
| Training (PPO) | 10 000 rollouts × 16 videos, 4 epochs each, ≈ 1.05 h CPU (A2C: 30 000 updates ≈ 53 min) |
| HSDPA test QoE | **PPO 0.971** (+7 % vs RobustMPC 0.907), A2C v1 0.934, A2C v2 0.924, optimum 1.145 |
| Stress tests / metro tunnel | PPO 1.393 / 1.37 vs RobustMPC 0.738 / −1.79 |
