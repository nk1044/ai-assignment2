# Adaptive Bitrate (ABR) Streaming with Deep Reinforcement Learning

## Quick Start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                      # simulator + baseline tests
python evaluate.py          # run all baselines (synthetic traces until real ones are added)
```

**Implementation progress**

- [x] Part 1 – Trace-driven streaming simulator + MDP environment (`abrsim/`)
- [x] Part 2 – Baselines (RB, BB, BOLA, MPC) + evaluation script (`baselines/`, `evaluate.py`)
- [ ] Part 3 – Real traces (FCC, HSDPA) download & preprocessing
- [ ] Part 4 – Actor-critic agent + training
- [ ] Part 5 – Graphs, explainability, web demo, own traces

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Background: How Video Streaming Works](#2-background-how-video-streaming-works)
3. [Why Reinforcement Learning?](#3-why-reinforcement-learning)
4. [MDP Formulation](#4-mdp-formulation)
   - 4.1 [State Space (S)](#41-state-space-s)
   - 4.2 [Action Space (A)](#42-action-space-a)
   - 4.3 [Transition Dynamics (P)](#43-transition-dynamics-p)
   - 4.4 [Reward Function (R)](#44-reward-function-r)
   - 4.5 [Discount Factor and Episode](#45-discount-factor-and-episode)
5. [Environment (Simulator)](#5-environment-simulator)
6. [Network Traces (Data)](#6-network-traces-data)
7. [RL Algorithm: Actor-Critic (A2C / A3C)](#7-rl-algorithm-actor-critic-a2c--a3c)
8. [Baseline Algorithms](#8-baseline-algorithms)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Planned Experiments and Graphs](#10-planned-experiments-and-graphs)
11. [Expected Results](#11-expected-results)
12. [What Makes This Project Unique (Bonus Plan)](#12-what-makes-this-project-unique-bonus-plan)
13. [Planned Repository Structure](#13-planned-repository-structure)
14. [Assumptions and Limitations](#14-assumptions-and-limitations)
15. [References](#15-references)

---

## 1. Problem Statement

When we watch a video on YouTube, Netflix or Hotstar, the network bandwidth is **not constant** —
it changes every second because of Wi-Fi interference, mobile handovers, congestion, moving
vehicles, etc. The video player must keep deciding **which quality (bitrate)** to download next.

- If it picks a **quality that is too high**, the download is slower than playback, the buffer
  empties, and the video **stalls (rebuffering)** — the spinning wheel.
- If it picks a **quality that is too low**, the video plays smoothly but looks **blurry**, wasting
  available bandwidth.
- If it **changes quality too often**, the viewer sees annoying jumps in sharpness
  (**quality switches / flicker**).

These three goals conflict with each other. The algorithm that makes this decision is called an
**Adaptive Bitrate (ABR) algorithm**.

**Goal of this project:** Learn an ABR policy using **Reinforcement Learning** that maximizes the
viewer's **Quality of Experience (QoE)** — high quality, few stalls, smooth switching — and show
that it outperforms classic hand-designed ABR algorithms on **real-world network traces**.

---

## 2. Background: How Video Streaming Works

Modern streaming uses **HTTP Adaptive Streaming** (DASH / HLS):

1. The video is split into small **chunks** (also called segments), each of fixed duration
   `L` seconds (we use `L = 4 s`).
2. Each chunk is encoded at several **bitrate levels** (e.g. 240p … 1080p).
3. The client downloads chunks **one at a time**. Before each download, the ABR algorithm chooses
   the bitrate for the next chunk.
4. Downloaded chunks are stored in a **playback buffer**. The player drains the buffer in real
   time (1 second of video per 1 second of wall clock).
5. If the buffer becomes empty before the next chunk arrives → **stall / rebuffering**.

```
 Network (variable bandwidth)
        │
        ▼
 ┌──────────────┐   choose bitrate   ┌────────────────┐
 │ ABR Agent    │ ─────────────────► │ Download chunk │
 │ (RL policy)  │ ◄───────────────── │ from server    │
 └──────────────┘  observe throughput└────────────────┘
        ▲            buffer, etc.            │
        │                                    ▼
        │                           ┌──────────────────┐
        └────────── state ───────── │ Playback buffer  │ ──► Video plays
                                    └──────────────────┘
```

So ABR is naturally a **sequential decision-making problem**: every decision changes the future
buffer level, which affects which decisions will be safe in the future.

---

## 3. Why Reinforcement Learning?

| Issue with classic ABR | How RL helps |
|---|---|
| **Rate-based** methods rely on throughput prediction, which is noisy and often wrong. | RL learns directly from experience how much to trust past throughput. |
| **Buffer-based** methods ignore throughput and react late. | RL uses *both* buffer and throughput history jointly. |
| **MPC** needs an accurate throughput model and is expensive to solve online. | RL learns a policy offline; inference is a single neural-network forward pass. |
| Hand-tuned rules are fixed for one QoE metric / network type. | RL can be retrained for any reward definition or network environment. |
| Decisions have **delayed consequences** (a greedy high bitrate now may cause a stall 20 s later). | RL optimizes the **long-term cumulative reward**, not just the next step. |

---

## 4. MDP Formulation

We model ABR as a **Markov Decision Process** `M = (S, A, P, R, γ)`.

- **Decision epoch (time step `t`)** = the moment just after chunk `t−1` finishes downloading and
  before chunk `t` is requested. So one step = one chunk.
- **Episode** = streaming one complete video (e.g. 48 chunks × 4 s ≈ 3.2 minutes) over one
  network trace.

### 4.1 State Space (S)

At step `t`, the agent observes a state vector `s_t` built from the player's local measurements:

| # | Feature | Symbol | Dim | Normalization | Meaning |
|---|---|---|---|---|---|
| 1 | Last chosen bitrate | `a_{t-1}` | 1 | `÷ max bitrate` | Needed to compute switching penalty |
| 2 | Current buffer occupancy | `B_t` | 1 | `÷ 10 s` | Seconds of video stored, safety margin against stalls |
| 3 | Past chunk throughputs | `x_{t-k}, k=1..8` | 8 | Mbps (chunk bits ÷ download time) | Measured download speed of last 8 chunks |
| 4 | Past chunk download times | `τ_{t-k}, k=1..8` | 8 | `÷ 10 s` | How long last 8 chunks took to download |
| 5 | Next chunk sizes (all bitrates) | `n_t ∈ ℝ^6` | 6 | MB | Size of the next chunk at each of 6 quality levels |
| 6 | Remaining chunks | `c_t` | 1 | `÷ total chunks` | How close the video is to the end |

**Total dimensionality:** `1 + 1 + 8 + 8 + 6 + 1 = 25` continuous features.

Implementation detail (following Pensieve): the state is stored as a **6 × 8 matrix**
(6 feature rows, history length 8) where scalar features occupy only the last column and
row 5 uses the first 6 columns. This lets us apply **1-D convolutions** over the time-series rows.

```
s_t ∈ ℝ^{6×8}
row 0: last bitrate         [ 0 0 0 0 0 0 0 a_{t-1} ]
row 1: buffer               [ 0 0 0 0 0 0 0 B_t     ]
row 2: throughput history   [ x_{t-8} ... x_{t-1}   ]
row 3: download-time hist.  [ τ_{t-8} ... τ_{t-1}   ]
row 4: next chunk sizes     [ n_1 ... n_6  0 0      ]
row 5: chunks remaining     [ 0 0 0 0 0 0 0 c_t     ]
```

**Why this state is (approximately) Markov:** the true next state depends on future bandwidth,
which is hidden. The throughput/download-time history acts as a *sufficient summary* of recent
network behaviour, so the agent can implicitly predict future bandwidth. Formally the problem is
a **POMDP**; using a history window is the standard way to approximate the Markov property.

The state space is **continuous and infinite** → tabular Q-learning is not feasible, which is
why we use a **neural network function approximator**.

### 4.2 Action Space (A)

**Discrete**, 6 actions — one per available bitrate level:

| Action `a` | Bitrate (kbps) | Approx. resolution |
|---|---|---|
| 0 | 300 | 240p |
| 1 | 750 | 360p |
| 2 | 1200 | 480p |
| 3 | 1850 | 720p |
| 4 | 2850 | 1080p |
| 5 | 4300 | 1440p |

`|A| = 6`. The policy `π(a | s)` outputs a **softmax probability distribution** over these 6
actions. (These are the bitrate ladder values from the standard "EnvivioDash3" test video used
in ABR research.)

### 4.3 Transition Dynamics (P)

The transition is driven by the hidden bandwidth process `C(τ)` (from a real network trace).
Given state `s_t` and action `a_t`:

1. **Chunk size:** `S_t = size(chunk t, bitrate a_t)` (bytes, from the real video's size table —
   sizes vary per chunk because of VBR encoding).

2. **Download time:** the simulator walks through the trace until `S_t` bytes are delivered:

   ```
   d_t = RTT + time needed such that ∫ C(τ)·η dτ = S_t
   ```
   where `RTT = 80 ms` and `η = 0.95` (packet payload fraction). A multiplicative noise
   `U(0.9, 1.1)` is applied to the delay to avoid overfitting.

3. **Rebuffering (stall) time:**

   ```
   T_t = max(d_t − B_t, 0)
   ```

4. **Buffer update:**

   ```
   B_{t+1} = max(B_t − d_t, 0) + L
   ```

5. **Buffer cap:** if `B_{t+1} > B_max = 60 s`, the player **waits (sleeps)** in steps of
   500 ms until the buffer drops below the threshold (time keeps advancing in the trace during
   sleep):

   ```
   Δ_sleep = ceil((B_{t+1} − B_max) / 0.5) · 0.5
   B_{t+1} ← B_{t+1} − Δ_sleep
   ```

6. **New observation:** measured throughput `x_t = S_t / d_t` and `τ_t = d_t` are pushed into the
   history; the oldest values drop out. `c_{t+1} = c_t − 1`.

Because bandwidth `C(τ)` is unknown and stochastic, the transition `P(s_{t+1} | s_t, a_t)` is
**unknown to the agent** → this is a **model-free RL** problem.

### 4.4 Reward Function (R)

We use the widely adopted linear QoE metric (from MPC, Yin et al. SIGCOMM'15):

```
                   ┌──── quality ────┐   ┌─ stall ─┐   ┌────────── smoothness ──────────┐
r_t  =  q(R_t)  −  μ · T_t  −  |q(R_t) − q(R_{t−1})|
```

Total QoE of a video with `N` chunks:

```
QoE = Σ_{t=1..N} q(R_t)  −  μ Σ_{t=1..N} T_t  −  Σ_{t=1..N-1} |q(R_{t+1}) − q(R_t)|
```

| Term | Meaning | Incentive |
|---|---|---|
| `q(R_t)` | Quality of chunk `t` at bitrate `R_t` | Encourages **higher bitrate** |
| `μ · T_t` | Rebuffering penalty (`T_t` in seconds) | Strongly discourages **stalls** |
| `\|q(R_t) − q(R_{t−1})\|` | Smoothness penalty | Discourages **frequent switches** |

**Three variants of `q(·)` (we will evaluate all three):**

| Variant | `q(R)` | `μ` | Intuition |
|---|---|---|---|
| **QoE_lin** (default) | `R` (in Mbps) | 4.3 | Quality grows linearly with bitrate. `μ = 4.3` = max bitrate, so 1 s of stall costs as much as one chunk of best quality is worth. |
| **QoE_log** | `log(R / R_min)` | 2.66 | Diminishing returns at high bitrates (more realistic for human perception). |
| **QoE_hd** | {300→1, 750→2, 1200→3, 1850→12, 2850→15, 4300→20} | 8 | Strongly rewards HD (≥720p). |

**Incentive design reasoning:**
- A large `μ` makes stalls the worst outcome — matches user studies showing viewers abandon videos
  after stalls.
- The smoothness term prevents the "oscillating" policy that greedily follows every bandwidth spike.
- Reward is given **per chunk** (dense), which makes credit assignment easier than a single
  end-of-episode reward.
- For training stability rewards may be scaled (e.g. divided by 10); this does not change the
  optimal policy.

### 4.5 Discount Factor and Episode

- **γ = 0.99** — ABR decisions have long-horizon effects (buffer built now protects against a
  bandwidth drop many chunks later).
- **Episode start:** random trace, random start offset within the trace, buffer = 0, first chunk
  at default (lowest/second-lowest) quality.
- **Episode end (terminal):** last chunk of the video downloaded. Then the next trace is loaded.

---

## 5. Environment (Simulator)

Training on a real video player is too slow (one video = several minutes of real time). We build a
**chunk-level trace-driven simulator** (like Pensieve's), which reproduces the player dynamics in
§4.3 in milliseconds:

- **Inputs:** network trace (timestamp, bandwidth) + video chunk-size table.
- **Per step:** given action, compute download time over the trace, update buffer, compute
  stall, sleep if buffer full, return `(next_state, reward, done, info)`.
- **Interface:** OpenAI Gym / Gymnasium-style `reset()` and `step(action)` API, so any RL
  library (our own PyTorch code or Stable-Baselines3) can be plugged in.
- **Speed:** ~1000× faster than real time → thousands of hours of streaming can be simulated in
  a few hours of training.

| Parameter | Value |
|---|---|
| Chunk length `L` | 4 s |
| Number of chunks per video | 48 |
| Bitrate levels | 6 (300–4300 kbps) |
| Max buffer `B_max` | 60 s |
| Sleep step when buffer full | 500 ms |
| RTT | 80 ms |
| Packet payload ratio | 0.95 |
| Delay noise | U(0.9, 1.1) |

---

## 6. Network Traces (Data)

We train and test on **real-world bandwidth traces** (publicly available):

| Dataset | Description | Typical bandwidth |
|---|---|---|
| **FCC** (US Federal Communications Commission, Measuring Broadband America) | Broadband / home Internet throughput logs | 0.2 – 6 Mbps (filtered) |
| **Norway HSDPA 3G** (Riiser et al., 2013) | Mobile 3G throughput recorded on buses, trains, ferries, cars in Oslo | 0.2 – 6 Mbps, highly variable |
| *(optional)* **Oboe / Puffer / 4G-LTE / 5G traces** | Newer traces for generalization testing | varies |

- Traces are pre-processed into `(time_s, bandwidth_Mbps)` pairs at 1-second granularity.
- Traces with average bandwidth > 6 Mbps are filtered out (otherwise every algorithm simply picks
  the max bitrate and the comparison is meaningless) and very low traces (< 0.2 Mbps) are removed.
- **Split:** ~80 % training traces / ~20 % test traces (no overlap). We also test on an
  *unseen* dataset (e.g. train on FCC, test on HSDPA) to measure generalization.

---

## 7. RL Algorithm: Actor-Critic (A2C / A3C)

### 7.1 Why Actor-Critic?

| Algorithm | Suitable? | Reason |
|---|---|---|
| Tabular Q-learning | ✗ | Continuous 25-dim state → infinite table. |
| DQN | possible | Works for discrete actions, but value-based methods are less stable with stochastic environments and give deterministic greedy policies. |
| **Actor-Critic (A2C/A3C)** | ✓ **chosen** | Directly learns a stochastic policy `π(a\|s)`; critic reduces variance; parallel workers explore many traces simultaneously. Proven on ABR (Pensieve). |
| PPO | ✓ (comparison) | A more stable actor-critic variant; we will compare against it. |

### 7.2 Network Architecture

```
          s_t (6 × 8)
              │
 ┌────────────┼─────────────────────────────────────────┐
 │ last bitrate, buffer, chunks left → Dense(128)       │
 │ throughput history (8)  → Conv1D(128, kernel=4)      │
 │ download-time hist. (8) → Conv1D(128, kernel=4)      │
 │ next chunk sizes (6)    → Conv1D(128, kernel=4)      │
 └────────────┬─────────────────────────────────────────┘
              │ concatenate + flatten
           Dense(128, ReLU)
              │
   ┌──────────┴──────────┐
 Actor head             Critic head
 Dense(6) + Softmax     Dense(1)
 π_θ(a | s)             V_φ(s)
```

Actor and critic have the **same structure but separate parameters** (θ and φ).

### 7.3 Training Objective

**Advantage** (n-step / one-step TD):

```
A(s_t, a_t) = r_t + γ V_φ(s_{t+1}) − V_φ(s_t)
```

**Actor (policy gradient) update with entropy regularization:**

```
θ ← θ + α_θ Σ_t ∇_θ log π_θ(a_t | s_t) · A(s_t, a_t)  +  β ∇_θ H(π_θ(· | s_t))
```

**Critic (value regression) update:**

```
φ ← φ − α_φ Σ_t ∇_φ ( r_t + γ V_φ(s_{t+1}) − V_φ(s_t) )²
```

- `H(π) = −Σ_a π(a|s) log π(a|s)` is the **entropy** bonus — encourages exploration early on.
  `β` is decayed from ~1 → ~0.1 (or lower) over training so the policy becomes more decisive.

### 7.4 Hyperparameters (initial)

| Hyperparameter | Value |
|---|---|
| Actor learning rate `α_θ` | 1e-4 |
| Critic learning rate `α_φ` | 1e-3 |
| Discount `γ` | 0.99 |
| Entropy weight `β` | 1.0 → 0.1 (decay) |
| Parallel workers (A3C) | 8–16 |
| Rollout length per update | 100 steps (or end of video) |
| Optimizer | Adam / RMSProp |
| Training epochs | ~50k–100k updates (until test QoE plateaus) |

### 7.5 Training Loop (pseudo-code)

```
initialize actor θ, critic φ
for each worker in parallel:
    loop:
        trace ← random training trace; s ← env.reset(trace)
        while not done:
            a ~ π_θ(·|s)
            s', r, done ← env.step(a)
            store (s, a, r, s')
            s ← s'
        compute advantages with critic
        send gradients to central learner → update θ, φ
        pull latest parameters
periodically: evaluate greedy policy (argmax π) on test traces, save best model
```

At **test time** the agent picks `a = argmax_a π_θ(a|s)` (deterministic).

---

## 8. Baseline Algorithms

We compare the RL agent against standard ABR algorithms, all running in the same simulator on the
same test traces:

| Baseline | Type | Rule |
|---|---|---|
| **Fixed lowest / random** | Sanity check | Always 300 kbps / uniform random action. |
| **Rate-Based (RB)** | Throughput | Predict bandwidth = harmonic mean of last 5 chunk throughputs; pick highest bitrate ≤ prediction. |
| **Buffer-Based (BB / BBA)** (Huang et al. SIGCOMM'14) | Buffer | Reservoir = 5 s, cushion = 10 s. Below reservoir → lowest bitrate; above reservoir+cushion → highest; linear mapping in between. |
| **BOLA** (Spiteri et al. INFOCOM'16) | Buffer (Lyapunov optimization) | Picks bitrate maximizing `(V·(utility + γp) − B) / size`. |
| **MPC / RobustMPC** (Yin et al. SIGCOMM'15) | Model-predictive control | Predicts throughput for next 5 chunks (RobustMPC discounts by max past prediction error), brute-forces all 6⁵ bitrate sequences and picks the first action of the best sequence. |
| **DQN** *(RL baseline)* | Value-based RL | Same state/reward, Q-network instead of actor-critic. |
| **PPO** *(RL baseline)* | Policy-gradient RL | Clipped-surrogate actor-critic. |
| **Offline Optimal** *(upper bound)* | Oracle | Dynamic programming with full future knowledge of the trace (not achievable in practice — shows how far each method is from ideal). |

---

## 9. Evaluation Metrics

For every test trace we record:

1. **Average QoE per chunk** (main metric) — total QoE ÷ number of chunks.
2. **Average bitrate** (Mbps) — utility component.
3. **Total rebuffering time** (s) and **rebuffering ratio** (stall time / playback time).
4. **Average bitrate change** (Mbps per switch) and **number of switches**.
5. **Normalized QoE** w.r.t. offline optimal.

Results are reported as **mean ± std** over all test traces and as **CDFs**.

---

## 10. Planned Experiments and Graphs

| # | Graph | Purpose |
|---|---|---|
| G1 | **Training curve:** average test QoE vs training epoch (+ entropy, critic loss) | Shows convergence of the actor-critic agent. |
| G2 | **Bar chart:** average QoE of all algorithms (QoE_lin, QoE_log, QoE_hd) | Main comparison. |
| G3 | **QoE breakdown:** stacked bars of bitrate utility, rebuffer penalty, smoothness penalty per algorithm | Shows *where* RL gains (e.g. fewer stalls vs higher quality). |
| G4 | **CDF of per-trace QoE** for each algorithm on FCC and HSDPA | Robustness across many network conditions. |
| G5 | **Time-series example on one trace:** bandwidth, chosen bitrate, and buffer level over time for RL vs BB vs MPC | Intuitive visualization of behaviour (RL builds buffer early, avoids stalls). |
| G6 | **Scatter:** avg bitrate vs rebuffer ratio for each algorithm | Visualizes the quality–stall trade-off frontier. |
| G7 | **Generalization:** train on FCC → test on HSDPA (and vice-versa) | Tests overfitting to a dataset. |
| G8 | **Ablation:** remove a state feature (e.g. no throughput history / no buffer / no chunk sizes) and measure QoE drop | Shows importance of each state component. |
| G9 | **Sensitivity to μ:** QoE components as stall penalty varies | Shows incentive design effect. |
| G10 | **A2C vs PPO vs DQN learning curves** | Justifies algorithm choice. |

---

## 11. Expected Results

Based on prior literature (Pensieve, SIGCOMM'17), we expect:

- The RL agent to achieve roughly **10–25 % higher average QoE** than the best heuristic (RobustMPC)
  on HSDPA and FCC traces.
- Gains mostly come from **reduced rebuffering** — the agent learns to build up buffer when
  bandwidth is good and lower quality early when throughput starts to drop.
- BB achieves smooth playback but under-uses bandwidth; RB has frequent stalls and switches.
- The RL agent should be within ~10–15 % of the offline optimal.

*(These are hypotheses; actual numbers will be filled in after experiments.)*

---

## 12. What Makes This Project Unique (Bonus Plan)

**Honest status:** the core (§4–§11) is a careful re-implementation of Pensieve. That is *correct*
but not *unique* — many teams can do it. The bonus has to come from **(a) one real research twist**
and **(b) a presentation that people remember**. Items are ranked by impact-per-effort.

### 12.1 Research Twists (what is *new* in our project)

| # | Idea | What we add | Why it is unique | Effort |
|---|---|---|---|---|
| U1 | **Our own Indian network traces** | Record bandwidth on our phones (Jio/Airtel 4G/5G) during real commutes — metro, bus, train, campus Wi-Fi, crowded market — using a small logging script (speedtest/iperf every second). Convert into the same trace format. | Nobody else has this dataset. "Trained on 3G Norway, tested on Delhi Metro" is a strong story; shows whether a model trained abroad works on Indian networks. | Medium |
| U2 | **"Data Saver ↔ Max Quality" preference-conditioned agent** | Add a data-cost term to the reward: `r_t = q(R_t) − μT_t − \|Δq\| − λ·S_t`. Feed `λ` as an extra state input and sample a random `λ` per episode, so **one agent** learns a whole family of policies. At test time a slider changes `λ`. | Real user need (limited mobile data plans). Turns single-objective RL into **multi-objective / conditioned RL** — goes beyond Pensieve. (Simpler fallback: train 3 separate agents with λ = 0, mid, high.) | Medium |
| U3 | **Explainable policy — "what did the agent learn?"** | (i) **Policy heatmap**: fix other features, sweep buffer (x-axis) × throughput (y-axis), colour = chosen bitrate. (ii) **Feature importance** via gradient saliency / permutation on the 25 state features. | Opens the black box; easy, beautiful slides; directly compares to the BB rule (which is just a straight line on the same plot). | Low |
| U4 | **Named stress-test scenarios** | Hand-crafted traces: *Metro tunnel* (sudden drop to ~0), *Stadium crowd* (slow decay), *Highway handover* (periodic dips), *Flaky Wi-Fi* (rapid oscillation). | Makes results relatable instead of only "average QoE on FCC". | Low |

> **Recommended combo:** U1 + U2 + U3 (+ U4 for free). U1 = unique data, U2 = unique method,
> U3 = unique insight.

### 12.2 Presentation / "Wow" Features (do not need to be functionally deep)

| # | Feature | What the audience sees | How to build cheaply | Effort |
|---|---|---|---|---|
| P1 | **Side-by-side video player demo** (upgraded web demo) | Two (or three) video players playing the *same* clip over the *same* trace: **Buffer-Based vs MPC vs our RL agent**. Low quality is shown blurry/pixelated, stalls show a spinning loader, a live QoE score ticks up/down under each. | Don't stream real video — use one HD video, apply CSS `filter: blur()` / pixelation according to the logged bitrate, pause + spinner during logged stall times. Replays **pre-computed simulation logs** (JSON). Pure HTML/JS. | Low–Med |
| P2 | **"Draw your own network" sandbox** | The user drags the mouse to draw a bandwidth curve; the agent immediately streams over it and shows its decisions + buffer graph. | Export the trained actor to **ONNX** and run in the browser with `onnxruntime-web` (or TF.js); port the simulator step (~40 lines) to JS. | Medium |
| P3 | **Human vs AI game** | A classmate plays the ABR agent manually (picks quality with buttons/keys while bandwidth changes); at the end a leaderboard compares their QoE with the RL agent and baselines. | Same JS simulator as P2, with buttons instead of the model. Great for live audience interaction. | Low (after P2) |
| P4 | **"Watch it learn" timelapse** | An animation/GIF: the agent's behaviour on one trace at epoch 0 (random, many stalls) → 1k → 10k → final (smooth). | Save checkpoints during training, run each on the same trace, render frames with matplotlib → GIF/MP4. | Low |
| P5 | **Data-saver slider** (pairs with U2) | Move a slider from *Max Quality* to *Data Saver* — the video gets slightly less sharp and a counter shows "MB saved / ₹ saved on your data plan". | Precompute runs for a few λ values; slider swaps between them. | Low |
| P6 | **Live dashboard during the talk** | Real-time charts: bandwidth, chosen bitrate, buffer level, cumulative QoE; plus the policy probability bars `π(a\|s)` changing every chunk. | Same web page (Chart.js / Plotly). | Low |
| P7 | **Real-world map of our traces** (pairs with U1) | A map of the city with the commute routes coloured by measured bandwidth, and where stalls would have happened for each algorithm. | Log GPS with the speed test; plot with Leaflet / folium. | Low–Med |
| P8 | **Polished visuals in the report** | Clean system diagram, MDP diagram, consistent colour per algorithm across all graphs, one "hero" figure on the first page (side-by-side frames from P1). | Draw.io / TikZ; matplotlib style sheet. | Low |

### 12.3 Suggested Demo Flow (for a ~5 min presentation)

1. **Hook (30 s):** play P1 on the *Metro tunnel* trace — the baseline player freezes, our agent keeps
   playing (slightly lower quality). Ask: "how did it know?"
2. **Formulation (1.5 min):** MDP diagram, state/action/reward.
3. **Results (1.5 min):** QoE bar chart, CDF, P4 learning timelapse, results on *our own* Indian traces (U1).
4. **Inside the agent (45 s):** U3 policy heatmap vs the BB rule.
5. **Interactive finale (45 s):** P2/P3 — a classmate draws a network or plays against the AI;
   P5 slider shows data saved.

### 12.4 Other Optional Extensions (if time remains)

- **Perceptual quality reward** — VMAF/SSIM per chunk instead of raw bitrate in `q(·)`.
- **Recurrent policy (LSTM/GRU)** instead of a fixed 8-step history (handles the POMDP better).
- **Robust / adversarial training** — an adversary perturbs bandwidth during training.
- **Live-streaming mode** — small buffer cap (4–8 s) plus a latency penalty.
- **5G / Starlink public traces** for a further generalization test.

---

## 13. Planned Repository Structure

```
ai-assignment-2/
├── README.md
├── data/
│   ├── traces/           # FCC, HSDPA network traces (train/ test/)
│   ├── traces_india/     # our own recorded traces (U1)
│   ├── traces_scenarios/ # hand-crafted stress tests (U4)
│   └── video/            # chunk size tables for each bitrate
├── abrsim/
│   └── abr_env.py        # Gym-style trace-driven streaming simulator
├── agents/
│   ├── actor_critic.py   # A2C/A3C model + training
│   ├── ppo.py            # PPO comparison (optional)
│   └── dqn.py            # DQN comparison (optional)
├── baselines/
│   ├── rate_based.py
│   ├── buffer_based.py
│   ├── bola.py
│   ├── mpc.py
│   └── offline_optimal.py
├── train.py
├── evaluate.py           # runs all algorithms on test traces
├── explain/              # policy heatmaps, saliency (U3)
├── plots/                # scripts + generated graphs, learning timelapse GIF (P4)
├── web/                  # side-by-side player, draw-your-network, human-vs-AI (P1–P6)
├── tools/trace_logger.py # records our own bandwidth traces on phone/laptop (U1)
└── report/               # two-column LaTeX (Overleaf) report
```

---

## 14. Assumptions and Limitations

- **Simulation vs reality:** chunk-level simulation ignores TCP slow-start and HTTP dynamics in
  detail; Pensieve showed such a simulator still transfers well to real players, but some gap exists.
- **Partial observability:** the agent cannot see future bandwidth; the history window is only an
  approximation of a Markov state.
- **Single video:** trained on one video's chunk-size table; different videos/encodings may need
  retraining or more diverse training videos.
- **QoE metric:** the linear QoE is a simplified model of human perception.
- **Bandwidth filtering:** traces are filtered to the 0.2–6 Mbps range to match the bitrate ladder.

---

## 15. References

1. H. Mao, R. Netravali, M. Alizadeh. *Neural Adaptive Video Streaming with Pensieve.* ACM SIGCOMM 2017.
2. X. Yin, A. Jindal, V. Sekar, B. Sinopoli. *A Control-Theoretic Approach for Dynamic Adaptive Video Streaming over HTTP (MPC).* ACM SIGCOMM 2015.
3. T.-Y. Huang et al. *A Buffer-Based Approach to Rate Adaptation (BBA).* ACM SIGCOMM 2014.
4. K. Spiteri, R. Urgaonkar, R. Sitaraman. *BOLA: Near-Optimal Bitrate Adaptation for Online Videos.* IEEE INFOCOM 2016.
5. H. Riiser et al. *Commute Path Bandwidth Traces from 3G Networks.* ACM MMSys 2013.
6. FCC. *Measuring Broadband America* dataset.
7. V. Mnih et al. *Asynchronous Methods for Deep Reinforcement Learning (A3C).* ICML 2016.
8. J. Schulman et al. *Proximal Policy Optimization Algorithms.* arXiv 2017.
9. F. Y. Yan et al. *Learning in situ: a randomized experiment in video streaming (Puffer).* NSDI 2020.
