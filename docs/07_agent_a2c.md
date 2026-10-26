# 07 — The A2C Agent (`agents/`, `train.py`)

## 7.1 Network (`agents/networks.py`)

Actor and critic have **identical architecture but separate parameters**. They do not share a trunk: if
the critic's large-magnitude value loss flowed through a shared trunk, it would distort the policy features.

```
input s: (B, 6, 8)
FeatureExtractor
  row 0 last value (last bitrate)  → Linear(1→128) + ReLU                       128
  row 1 last value (buffer)        → Linear(1→128) + ReLU                       128
  row 5 last value (chunks left)   → Linear(1→128) + ReLU                       128
  row 2 (8 throughputs)            → Conv1d(1→128, k=4) + ReLU → 128×5 flatten  640
  row 3 (8 download times)         → Conv1d(1→128, k=4) + ReLU → 128×5 flatten  640
  row 4 first 6 (next sizes)       → Conv1d(1→128, k=4) + ReLU → 128×3 flatten  384
  [optional λ]                     → Linear(1→128) + ReLU                       128
  concat (2048, or 2176 with λ) → Linear(→128) + ReLU
Actor head:  Linear(128→6)  → logits  (softmax = π(a|s))
Critic head: Linear(128→1)  → V(s)
```

≈ 265 k parameters each (the merge layer 2048×128 is 262 k of them).

**Why 1-D convolutions** on the histories: a kernel of 4 slides over time and detects local patterns
(drop, rise, variance) wherever they occur in the window. Its weights are shared over time, so it uses
fewer parameters than a dense layer and has the right inductive bias. This architecture is Pensieve's,
proven on this exact task.

## 7.2 Algorithm: synchronous A2C with GAE (`agents/a2c.py`)

### Acting
`act(states, extras, greedy)`: compute softmax probabilities. Greedy → argmax. Otherwise sample via the
inverse CDF (`(cumsum < u).sum()`), which is fast and batched for 16 envs.

### Advantages (GAE)
```
δ_t = r_t/scale + γ·V(s_{t+1})·(1 − done_t) − V(s_t)
A_t = δ_t + γλ·(1 − done_t)·A_{t+1}          (computed backwards, λ = 0.95)
G_t = A_t + V(s_t)                            (critic target)
```
After the last chunk `V = 0` (terminal). λ = 0.95 trades off variance (λ = 1, Monte-Carlo) and bias
(λ = 0, one-step TD). There is a unit test: with a zero critic and λ = 1, the advantages equal the
discounted returns.

### Losses (one update per iteration, 768 transitions)
```
Â        = (A − mean A) / (std A + 1e−8)                       advantage normalisation
L_actor  = −mean[ log π(a_t|s_t) · Â_t ]  −  β · mean H(π(·|s_t))
L_critic = mean[ (G_t − V(s_t))² ]
```
- Adam for each network (actor lr 1e−4, critic lr 1e−3). The critic learns faster so that it gives
  useful advantages early.
- Gradients are clipped to norm 5.
- **Entropy weight β** decays linearly from `entropy_start` to `entropy_end` over `entropy_decay_iters`
  (defaults 1.0 → 0.05 over 20 000). It forces exploration early, then lets the policy become decisive.
- **Why normalise advantages:** stalls create rewards of −20 to −40 next to +1 for quality, so raw
  advantages vary by orders of magnitude between batches. Normalising keeps the step size stable.

### Checkpoints
`save(path, **meta)` stores the config, iteration, actor, critic, both optimiser states and meta.
`load` restores everything, so training can be resumed and evaluated identically.

## 7.3 Rollouts (`agents/rollout.py`)

`RolloutCollector(traces, n_envs=16, qoe, data_lambda, lambda_range, seed)`:
- 16 `ABREnv`s with seeds `seed+i`, random trace starts and noise on;
- each `collect()` resets all 16 and steps them **in lock-step** with one batched forward pass per step.
  All videos are 48 chunks long, so they finish together. The rollout is then a clean `(T=48, N=16, …)` batch:
  `states (48,16,6,8)`, `actions`, `rewards`, `dones`, `infos` (+ `extras` if λ-conditioned).
- **Synchronous A2C vs A3C:** the 16 parallel environments give the same decorrelated, diverse
  experience as A3C's asynchronous workers. They are deterministic, easier to debug, and the batched
  forward pass is efficient on one CPU. Pensieve used A3C with 16 workers; A2C is known to perform as well
  or better (OpenAI baselines).

## 7.4 Training loop (`train.py`)

```
traces = load_dataset("train")                   # 115
train, val = split_validation(traces, 0.10, seed)  # 103 / 12, deterministic permutation
agent = A2CAgent(A2CConfig(...)); collector = RolloutCollector(train, 16, ...)
write runs/<name>/config.json (all args + val trace names)
for it in 1..iters (30 000):
    rollout = collector.collect(agent)
    stats   = agent.update(rollout)
    if it == 1 or it % 1000 == 0: save ckpt_<it>.pt       # for the learning timelapse
    if it % 250 == 0 or last:
        val = greedy RLPolicy on val traces via evaluate.run_policy   (fixed seed → same conditions each time)
        append row to log.csv; print
        if val.qoe > best: save best.pt
save final.pt
```

- **The validation set comes from the training pool**, never from the test set. This prevents test
  leakage through model selection.
- Every hyperparameter is a CLI flag (generated automatically from the `A2CConfig` dataclass fields).
- `--lambda-range lo hi` → `extra_dim = 1`, and λ ~ U(lo, hi) per video is fed to both networks
  (Data Saver, Ch. 12).
- Cost: ≈ 105 ms per update (the backward pass dominates) + 30 ms collection. 30 k iterations take
  ≈ 53 min (measured).

## 7.5 Hyperparameters

| Name | Value | Reason |
|---|---|---|
| actor_lr / critic_lr | 1e−4 / 1e−3 | Pensieve's values; the critic must lead |
| γ | 0.99 | long-horizon buffer effects |
| gae_lambda | 0.95 | standard bias/variance trade-off |
| entropy 1.0 → 0.05 over 20 k | exploration schedule (see 7.6) |
| n_envs | 16 | as in Pensieve's 16 A3C agents; batch of 768 |
| max_grad_norm | 5 | stability against stall-induced outliers |
| reward_scale | 1 | advantages are normalised anyway |
| iters | 30 000 | ≈ 23 M decisions |

## 7.6 What the first training run showed (runs/a2c_lin)

| Iteration | Val QoE | Val bitrate | Entropy |
|---|---|---|---|
| 250 | −0.56 | 0.87 | 1.78 |
| 10 250 | −0.36 | 0.65 | 1.77 |
| 20 250 | −0.11 | 1.01 | 1.09 |
| 29 750 (best) | −0.07 | 1.02 | 1.19 |

- **Why validation QoE is negative:** the 12 validation traces include FCC traces with complete outages.
  Validation stall time stays at ≈ 10.3 s/video *whatever bitrate is chosen*, so the stalls are
  unavoidable. Validation is used only to *compare checkpoints with each other*, so the offset does not
  matter.
- **Test result of best.pt** on 142 HSDPA test traces: **0.934** QoE/chunk. RobustMPC gets 0.907 (+3 %),
  and stalls drop 0.91 → 0.17 s/video. The agent beats every heuristic.
- **Diagnosed weakness:** entropy stayed ≈ ln 6 = 1.79 (a uniform random policy) for the first ~15 k
  iterations. With normalised advantages (unit scale), an entropy weight of 1.0 dominates the policy-gradient
  term, so the agent could not commit until β had decayed. Roughly half of the budget was spent on
  near-random behaviour. A smaller `--entropy-start` (e.g. 0.1–0.2) and/or more iterations should improve
  the result. This is a good example of hyperparameter diagnosis from training curves.
