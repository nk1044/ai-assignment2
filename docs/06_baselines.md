# 06 — Baselines and the Offline Optimal (`baselines/`)

All of them implement `Policy.select(state, env) -> int` (Ch. 02). Helpers in `base.py`:
- `buffer_s(state) = state[1, −1] · 10`
- `past_throughputs(state) = row 2 without zeros` (oldest first)
- `harmonic_mean(x)`

## 6.1 Sanity baselines

- `ConstantPolicy(0)` → "fixed_300": always the lowest quality. It almost never stalls but has low
  quality. QoE ≈ 0.30.
- `RandomPolicy`: uniform random. QoE ≈ −15, which shows that decisions matter a lot.

## 6.2 Rate-Based (RB) — `heuristics.RateBased`

```
tp = last 5 measured throughputs (Mbps)
if none yet: return default (750k)
pred = harmonic_mean(tp)
return highest bitrate ≤ pred (else 0)
```
The harmonic mean is used because it is dominated by small values, so one fast outlier does not cause
overshoot (the FESTIVE/MPC convention). Weakness: no buffer awareness, and it trusts the prediction fully.

## 6.3 Buffer-Based (BB, "BBA-0", Huang et al. SIGCOMM'14) — `heuristics.BufferBased`

```
B < reservoir (5 s)            → lowest
B ≥ reservoir + cushion (15 s) → highest
else  int(5 · (B − 5) / 10)     → linear map
```
No throughput at all. It never stalls in steady state, but it oscillates (≈ 25 switches per video) and
reacts late.

## 6.4 BOLA (Spiteri et al. INFOCOM'16) — `heuristics.Bola`

BOLA is derived from Lyapunov optimisation and maximises a utility-minus-buffer-pressure score:

```
v_m  = ln(S_m / S_1)                 utility of level m (S = bitrate as size proxy)
Q    = buffer in chunks = B / 4
Qmax = 60 / 4 = 15,  gp = 5
V    = (Qmax − 1) / (v_M + gp)
choose argmax_m  (V · (v_m + gp) − Q) / S_m
```
The quality rises monotonically with the buffer (a test checks this). In practice it almost never stalls
(0.01 s per video), but it under-uses bandwidth (0.94 Mbps average).

## 6.5 MPC / RobustMPC (Yin et al. SIGCOMM'15) — `mpc.MPC`

At every chunk:

1. **Predict** bandwidth `ŵ` = harmonic mean of the last 5 throughputs.
2. **RobustMPC:** keep a history of relative prediction errors `|ŵ_prev − x|/x` and divide `ŵ` by
   `1 + max(last 5 errors)`. This makes it conservative after surprises. FastMPC skips this step.
3. **Enumerate** all `6^5 = 7776` bitrate sequences for the next `h = 5` chunks (fewer near the end of
   the video).
4. **Simulate** each sequence with `ŵ` and the real future chunk sizes from the manifest:
   `delay = size·8/1e6/ŵ`, `rebuf = max(delay − buf, 0)`, `buf = max(buf − delay, 0) + 4`.
   Score it with the *same reward* as the RL agent (quality − μ·rebuf − |Δq| − λ·size).
5. **Apply** only the first action of the best sequence (receding horizon).

**Implementation detail:** the 7776 sequences are a pre-computed `(7776, 5)` array, and all of them are
simulated at once with NumPy vectors. That takes ~1 ms per decision instead of ~100 ms with Python loops.

**Why it is the strongest baseline:** it combines throughput and buffer and looks ahead.
**Why it fails in tunnels:** the predictor extrapolates the past. When the bandwidth drops to zero, it is
in the middle of downloading a big chunk → long stall (metro tunnel QoE −1.79).

## 6.6 Offline optimal (upper bound) — `oracle.OfflineOptimal`

Knows the entire future trace. It is not a real algorithm, but it is the **ceiling** that measures headroom.

**Dynamic programming over chunks:**
```
frontier = {(last_action, −1): (total_reward=0, simulator_state, actions=[])}
for each chunk k:
    next = {}
    for each (prev, _) → (total, sim_state, path) in frontier:
        for a in 0..5:
            restore sim_state; out = download_chunk(a)       (noise-free copy of the simulator)
            r = q[a] − μ·rebuf − |q[a] − q[prev]| − λ·size
            key = (a, round(buffer_after / 0.5 s))
            keep the entry with the highest total + r for that key
    frontier = next
return path of the best final entry
```

- **Why it is (near-)optimal.** Two histories that end with the same last quality and the same buffer
  (±0.25 s) are interchangeable for the future: the reward of the remaining chunks only depends on
  these. (Their wall-clock positions differ only by stall time, which is already paid for in the reward.)
  Keeping the best one per key is the Bellman principle on a discretised state.
- **Measured convergence:** bin widths of 1.0 / 0.5 / 0.25 s give QoE within 0.01 of each other, so 0.5 s
  is used. That is ≤ 720 states × 6 actions per chunk, ≈ 0.6 s per trace.
- **Test:** `test_oracle_is_an_upper_bound` checks that the oracle's total QoE ≥ every baseline's on every trace.
- **Needs a noise-free env** to replay its plan exactly. That is one reason test-time simulation is
  noise-free (Ch. 08). The plan is computed once at chunk 0 (`select` caches it).

## 6.7 Baseline results (HSDPA test, 142 traces, QoE_lin)

| Policy | QoE/chunk | Bitrate | Stall s/video | Switches |
|---|---|---|---|---|
| Offline optimal | 1.145 | 1.18 | 0.01 | 2.4 |
| RobustMPC | 0.907 | 1.13 | 0.91 | 9.5 |
| BOLA | 0.771 | 0.94 | 0.01 | 13.8 |
| RB | 0.745 | 0.92 | 1.13 | 6.5 |
| BB | 0.626 | 1.14 | 1.80 | 25.3 |
| FastMPC | 0.431 | 1.20 | 6.82 | 10.6 |
| Fixed 300k | 0.299 | 0.30 | 0.01 | 0 |
| Random | −15.38 | 1.87 | 172 | 39 |

The ranking RobustMPC > BOLA ≈ RB > BB > FastMPC matches the literature. This is evidence that the
simulator and the baselines are implemented correctly.
