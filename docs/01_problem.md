# 01 — The Problem

## 1.1 Setting: HTTP adaptive streaming (DASH / HLS)

Every major video service (YouTube, Netflix, Hotstar) streams video this way:

1. The video is cut into **chunks** of fixed playback length `L` (here `L = 4 s`).
2. Each chunk is encoded at several **bitrates** (the "bitrate ladder"). Here there are 6:
   300 / 750 / 1200 / 1850 / 2850 / 4300 kbps, roughly 240p … 1440p.
3. The client downloads chunks **one at a time, sequentially**. Before each download it must decide
   which bitrate to request. The component that decides is the **ABR (Adaptive BitRate) algorithm**.
4. Downloaded chunks go into a **playback buffer**. Playback drains the buffer in real time: 1 s of video
   per 1 s of wall-clock time.
5. If the buffer empties before the next chunk arrives, playback freezes. This is a **stall** or
   **rebuffering event**.

The ABR algorithm only controls **which bitrate** to request. It does not control the network.

## 1.2 Why it is hard

The available bandwidth changes every second. On mobile networks it can go from 3 Mbps to 0 in a
tunnel within a second. The ABR decision has three conflicting goals:

| Goal | Failure mode if ignored |
|---|---|
| High quality (high bitrate) | Blurry video, wasted bandwidth |
| No stalls | Video freezes. User studies show stalls are the main reason viewers abandon a video |
| Smoothness (few switches) | Visible quality flicker |

On top of that:

- **The future is unknown.** The decision depends on how fast the *next* download will be, but the
  player only sees past throughput.
- **Decisions have delayed consequences.** A greedy high bitrate now may look fine, but it drains the
  buffer, and a bandwidth dip 20 s later then causes a stall. A low bitrate now "invests" in buffer that
  protects against future dips. This is a *sequential* decision problem, not a one-step prediction.
- **Measurements are noisy and biased.** Throughput measured on small chunks underestimates capacity,
  and chunk sizes vary with scene complexity (VBR encoding).
- **Network conditions differ wildly** (3G bus, home broadband, café Wi-Fi). A rule tuned for one is
  bad for another.

## 1.3 How quality of experience is measured (QoE)

We use the standard linear QoE from the MPC paper (Yin et al., SIGCOMM 2015). Pensieve uses the same one.
For a video of `N` chunks:

```
QoE = Σ_{n=1..N} q(R_n)  −  μ · Σ_{n=1..N} T_n  −  Σ_{n=1..N−1} | q(R_{n+1}) − q(R_n) |
        quality              stall penalty              smoothness penalty
```

- `R_n` = bitrate of chunk n, `q(·)` = quality function. For QoE_lin, `q(R) = R` in Mbps.
- `T_n` = stall time (s) caused while downloading chunk n.
- `μ = 4.3` = the top bitrate in Mbps. So **1 s of stall costs as much as one chunk at the best quality
  is worth**.

Two other variants are supported: QoE_log (`q = log(R/R_min)`, μ = 2.66) and QoE_hd (a table that strongly
rewards ≥ 720p, μ = 8). They model different user preferences.

## 1.4 Classic approaches and their weaknesses

| Family | Idea | Weakness |
|---|---|---|
| Rate-based (RB) | Predict bandwidth from past throughput, pick the highest bitrate below the prediction | Predictions are wrong exactly when it matters (sudden drops) → stalls |
| Buffer-based (BB, BOLA) | Map buffer level → bitrate | Ignores throughput, reacts late, oscillates (BB ≈ 25 switches per video) |
| Model predictive control (MPC) | Predict bandwidth and optimise QoE over the next 5 chunks | Only as good as its predictor; collapses when the bandwidth disappears (metro tunnel: QoE −1.79) |

All of them are **hand-designed rules with fixed assumptions**. None adapts its strategy to the
statistics of real networks.

## 1.5 Our solution in one sentence

Formulate ABR as an MDP and let an **actor-critic RL agent learn the policy directly from experience**.
The agent replays thousands of hours of **real recorded network traces** in a fast simulator, so it learns
what real bandwidth does (dips, recoveries, outages) and how much buffer is worth. No bandwidth model is
written by hand.

## 1.6 How we know if it worked

- Test on **unseen** real traces: 142 Norway 3G + 12 FCC.
- Compare against RB, BB, BOLA, FastMPC and RobustMPC, plus sanity baselines (fixed lowest quality, random).
- Compare against the **offline optimal**, a dynamic program that knows the whole future. It is an upper
  bound no real player can reach. Measured gap on HSDPA: best baseline 0.907 vs optimal 1.145. **The agent
  has 21 % headroom to win.**
- Stress-test on named scenarios: metro tunnel, stadium crowd, highway handover, flaky Wi-Fi, rural edge.
