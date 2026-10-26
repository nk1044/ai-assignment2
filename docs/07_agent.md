# 07 — The RL Agent: PPO with an Asymmetric Critic (`agents/`, `train.py`)

The project went through two agents. Both are documented here, because the comparison is part of the
results.

| Version | Algorithm | Critic sees | Where |
|---|---|---|---|
| A2C v1, v2 | synchronous A2C | the same 6×8 observation as the actor (symmetric) | code at commit `6d8faf4`; results in `experiments/` |
| **Current** | **PPO** | observation **+ true future bandwidth** (asymmetric / privileged) | `agents/ppo.py` |

## 7.1 Networks (`agents/networks.py`)

```
FeatureExtractor(extra_dim, priv_len)
  row 0 last value (last bitrate)  → Linear(1→128) + ReLU                       128
  row 1 last value (buffer)        → Linear(1→128) + ReLU                       128
  row 5 last value (chunks left)   → Linear(1→128) + ReLU                       128
  row 2 (8 throughputs)            → Conv1d(1→128, k=4) + ReLU → 128×5 flatten  640
  row 3 (8 download times)         → Conv1d(1→128, k=4) + ReLU → 128×5 flatten  640
  row 4 first 6 (next sizes)       → Conv1d(1→128, k=4) + ReLU → 128×3 flatten  384
  [optional λ, data saver]         → Linear(1→128) + ReLU                       128
  [critic only: 8 future-bw windows] → Conv1d(1→128, k=4) + ReLU → 128×5        640
  concat → Linear(→128) + ReLU
Actor  = FeatureExtractor(priv_len=0) + Linear(128→6)   logits → softmax π(a|s)     ≈ 266 k params
Critic = FeatureExtractor(priv_len=8) + Linear(128→1)   V(s, future)                ≈ 348 k params
```

- **Separate actor and critic.** If they shared a trunk, the critic's large value loss (stalls give
  returns of −20 … −40) would distort the policy features. More importantly, a shared trunk would leak
  privileged information into the actor.
- **1-D convolutions** detect temporal patterns (drops, trends, variance) in the 8-step windows with
  shared weights. This is Pensieve's proven architecture. The same branch type is used for the
  future-bandwidth input.

## 7.2 The asymmetric (privileged) critic

**Observation.** The actor sees only the past (throughput and download-time history), so ABR is a POMDP.
Two situations with an identical past (e.g. "train about to enter a tunnel" vs "train about to leave
one") have very different futures, and therefore different values. A critic that sees only the past must
average them. That blurs `V`, and therefore makes the advantages noisy.

**Idea** (Pinto et al., 2017, "asymmetric actor-critic"). The critic exists only during training, and
training happens in the simulator, which knows the entire trace. So the critic may use information that
is unavailable at deployment.

**What it sees.** `ABREnv.privileged()` → `StreamingSimulator.future_bandwidth()`:
- the true mean bandwidth in each of the next 8 windows of 4 s, so 32 s of the future;
- read from the trace at the current position (`ptr`, `last_time`) *without moving it*;
- it wraps the trace like the download does.

It is recorded **at decision time**, before `env.step`: `RolloutCollector` stores it as
`rollout["privileged"]` with shape `(48, 16, 8)`.

**Why the critic also gets the normal observation.** A critic conditioned only on the hidden state can
bias the policy gradient under partial observability (Baisero & Amato, 2022). Conditioning on
observation + privileged information avoids that.

**What does not change:**
- The actor's input is exactly as before.
- `RLPolicy`, evaluation and deployment use only the actor.
- Test results are fully comparable to the A2C runs.

## 7.3 PPO (`agents/ppo.py`)

**Rollout.** As before: 16 envs in lock-step, one video each, giving 768 transitions.

**Advantages (GAE)** with the privileged critic:
```
δ_t = r_t/scale + γ·V(s_{t+1}, f_{t+1})·(1 − done_t) − V(s_t, f_t)
A_t = δ_t + γλ·(1 − done_t)·A_{t+1}        (λ = 0.95, backwards)
G_t = A_t + V(s_t, f_t)
Â   = (A − mean) / (std + 1e−8)
```

**Update** (`PPOAgent.update`):
```
old_logp = log π_θ(a|s) for all 768 samples, computed once (no grad)  ← the policy that collected the data
for epoch in 1..4:
    shuffle; for each of 4 minibatches (192):
        ρ = exp(log π_θ(a|s) − old_logp)
        L_actor  = −mean min(ρÂ, clip(ρ, 0.8, 1.2)Â) − β·mean H(π_θ)     → Adam step (actor)
        L_critic = mean (G − V(s, f))²                                    → Adam step (critic)
    if mean approx-KL(old‖new) of this epoch > 0.03: stop the epochs early
```

- **Why the clipping matters.** Plain policy gradient (A2C) can only take one step per rollout: after
  that step the data was produced by a different policy, so more steps would use stale data and could
  destroy the policy. PPO's clip removes the incentive to move `ρ` outside [0.8, 1.2], so the same
  rollout can be reused safely for several epochs. That means more learning per simulated video, and
  smoother learning.
- **Target KL** is a second safety net. If a single epoch moves the policy too far, the remaining epochs
  are skipped.
- **Logged per update:** `policy_loss`, `critic_loss`, `entropy`, `approx_kl`, `clip_frac` (the fraction
  of samples where clipping was active), `entropy_weight`, `epochs`. Healthy values from the speed test:
  KL ≈ 0.011, clip_frac ≈ 0.11, all 4 epochs used.

**Acting.** `act(states, extras, greedy)` computes softmax probabilities. Greedy takes the argmax (test
time); otherwise it samples by inverse CDF (`(cumsum < u).sum()`), batched for the 16 envs.

**Checkpoints.** `save` stores the config, iteration, both networks, both optimiser states and meta.
`load` restores everything.

## 7.4 Training loop (`train.py`)

```
traces = load_dataset("train")                       # 115 (FCC + HSDPA train)
train, val = split_validation(traces, 0.10, seed)    # 103 / 12
for it in 1..10 000:
    rollout = collector.collect(agent)               # 16 videos, with privileged future bandwidth
    stats   = agent.update(rollout)                  # PPO epochs
    it == 1 or it % 500 == 0  → ckpt_<it>.pt
    it % 100 == 0             → validate: greedy policy on 12 val traces × 4 random starts (48 videos,
                                 fixed seed → identical conditions every time) → log.csv, best.pt
save final.pt
```

- **Validation uses 4 starts per trace** (`--val-repeats 4`). A2C v2 showed that 12 single videos are too
  noisy to select the best checkpoint: v2 had the better validation score but tested worse.
- **Cost:** ≈ 34 ms collection + ≈ 345 ms update per iteration. 10 000 iterations ≈ 1.05 h, about the
  same wall-clock budget as one A2C run (30 000 iterations ≈ 53 min). PPO uses ⅓ of the videos but
  learns 4× from each.

## 7.5 Hyperparameters (`PPOConfig`, every field is a CLI flag)

| Name | Value | Reason |
|---|---|---|
| actor_lr / critic_lr | 1e−4 / 1e−3 | the critic must lead; the actor moves slowly |
| γ / gae_lambda | 0.99 / 0.95 | long-horizon buffer effects; standard bias/variance trade-off |
| clip_eps | 0.2 | PPO default |
| ppo_epochs × minibatches | 4 × 4 (192 each) | reuse each rollout 4 times |
| target_kl | 0.03 | early-stop safety |
| entropy 0.1 → 0.01 over 6 000 | lower than A2C v1's 1.0, which kept the policy uniform for half the run |
| n_envs | 16 | 768 transitions per rollout |
| max_grad_norm | 5 | stall-induced outliers |
| iters | 10 000 | same wall-clock as the A2C runs |
| priv_len | 8 (× 4 s) | 32 s of look-ahead ≈ the time scale of buffer decisions |

## 7.6 History: the A2C runs (symmetric critic)

The A2C agent used the same networks without the privileged branch. Its update was one step per rollout:
`L = −mean(log π · Â) − β·H` plus critic MSE.

| Run | Entropy schedule | Test HSDPA | FCC | Scenarios | Tunnel | Observation |
|---|---|---|---|---|---|---|
| A2C v1 | 1.0 → 0.05 / 20 k | 0.934 | 0.644 | 1.047 | +0.16 | entropy ≈ ln 6 (uniform) for ~15 k of 30 k updates: half the run wasted |
| A2C v2 | 0.2 → 0.05 / 15 k | 0.924 | 0.645 | 0.926 | −0.50 | learned faster (val −0.20 at 2.75 k vs −0.56), but the noisy validation picked a weaker checkpoint |

Both beat RobustMPC (0.907 / 0.604 / 0.738 / −1.79). Lessons that shaped the PPO version:
1. Entropy bonus must be small relative to normalised advantages.
2. Checkpoint selection needs a less noisy validation set.
3. Learning is limited by noisy advantages, which motivated the privileged critic.

## 7.7 Why validation QoE is negative

The 12 validation traces include FCC traces with complete outages: ≈ 10 s of stall per video that no
policy can avoid. Validation is only used to compare checkpoints of the same run, so the offset does not
matter. Judge models on the test sets (Ch. 08) or in `experiments/README.md`.
