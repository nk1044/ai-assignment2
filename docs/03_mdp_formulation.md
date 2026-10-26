# 03 — MDP Formulation

`M = (S, A, P, R, γ)`. Code: `abrsim/abr_env.py` (`ABREnv`), constants in `abrsim/config.py`.

## 3.1 Time step and episode

- **One step = one chunk decision.** Step `t` happens after chunk `t−1` has finished downloading and before
  chunk `t` is requested. Steps are therefore *not* equally spaced in wall-clock time: a step lasts
  download time + any sleep.
- **Episode = one full video** of 48 chunks (≈ 3.2 min of content) streamed over one network trace.
  Every episode has exactly 48 steps. That is why 16 parallel environments finish together (Ch. 07).
- **Terminal:** after the 48th chunk. There is no bootstrapping past it (V = 0).

## 3.2 State space S — a 6 × 8 matrix

```
row 0  last bitrate        [0 0 0 0 0 0 0  a_{t-1}/4300 ]      1 value
row 1  buffer              [0 0 0 0 0 0 0  B_t / 10 s   ]      1 value
row 2  throughput history  [x_{t-8} … x_{t-1}]  (Mbps)          8 values
row 3  download-time hist. [d_{t-8} … d_{t-1}] / 10 s           8 values
row 4  next chunk sizes    [n_0 … n_5, 0, 0]  (MB)              6 values
row 5  chunks remaining    [0 0 0 0 0 0 0  c_t / 48     ]      1 value
```

25 meaningful numbers in a fixed 6×8 container (`S_INFO = 6`, `S_LEN = 8`). Zeros mean "unused" or
"no history yet".

| Feature | Why the agent needs it |
|---|---|
| Last bitrate | The smoothness penalty depends on it; without it the reward would not be Markov in the state |
| Buffer | The safety margin: how long playback survives if the next download is slow |
| Throughput history | The only evidence about future bandwidth (trend, variance, recent drops) |
| Download-time history | Complements throughput: a long download warns of trouble even when the throughput number looks OK, and it is informative about RTT-dominated small chunks |
| Next chunk sizes | VBR: the same bitrate can be a big or small chunk depending on scene complexity; tells the real cost of each action |
| Chunks remaining | Near the end, a large buffer is worthless (it never gets played), so the agent can spend it on quality |

**Normalisation.** Everything is roughly in [0, 5] so no input dominates the first layer:
- bitrate is divided by the max bitrate;
- buffer and delay are divided by 10 s;
- sizes are in MB, throughput in Mbps;
- chunks remaining is divided by 48.

**Update rule** (`ABREnv.step`): `np.roll(state, -1, axis=1)` shifts every row one step left (oldest
falls out), the last column is zeroed, then the new values are written into column −1. Row 4 is
overwritten completely with the next chunk's sizes. At the end of the video those are zeros.

**Initial state** (`ABREnv.reset`): all zeros except:
- row 0 = default action 750 kbps (normalised);
- row 4 = sizes of chunk 0;
- row 5 = 1.0.

**Is it Markov?** Not exactly. The true next state depends on future bandwidth, which the agent never
sees, so the problem is a **POMDP**. The 8-step histories act as a sufficient-in-practice summary of the
network's recent behaviour, so the agent can implicitly predict the future. This is the standard
approximation (Pensieve uses the same one). An LSTM over the full history is a possible extension.

**Why not a tabular method?** The state is continuous and 25-dimensional, so the state space is
infinite → we need function approximation (a neural network).

## 3.3 Action space A — discrete, |A| = 6

| a | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| kbps | 300 | 750 | 1200 | 1850 | 2850 | 4300 |

The policy outputs a softmax over these 6 actions. During training the agent samples from it
(exploration); at test time it takes the argmax (deterministic).

The action space is discrete because real DASH manifests offer a finite ladder. A continuous bitrate is
not requestable.

## 3.4 Transition dynamics P (implemented by the simulator, Ch. 04)

Given `s_t, a_t` and the hidden bandwidth trace `C(τ)`:

```
S_t   = size(chunk t, quality a_t)                                   bytes, from the real video
d_t   = [time to push S_t bytes through C(τ)·0.95] + RTT(0.08 s)     × noise U(0.9, 1.1) in training
T_t   = max(d_t − B_t, 0)                                             stall time
B'    = max(B_t − d_t, 0) + 4 s                                       buffer after download
if B' > 60 s:  sleep Δ = ceil((B' − 60)/0.5)·0.5 ;  B_{t+1} = B' − Δ   (trace clock advances by Δ)
x_t   = S_t·8 / d_t / 1e6 Mbps ;  history rows shift
```

`P` is **unknown to the agent and stochastic** (from its point of view, because future bandwidth is
hidden) → this is **model-free RL**.

## 3.5 Reward function R

Per chunk (`ABREnv.step`):

```
r_t = q(R_t)  −  μ · T_t  −  | q(R_t) − q(R_{t−1}) |  −  λ · S_t[MB]
```

| QoE variant | q(R) | μ |
|---|---|---|
| `lin` (default) | R in Mbps (0.3 … 4.3) | 4.3 |
| `log` | ln(R / 300 kbps) | 2.66 |
| `hd` | [1, 2, 3, 12, 15, 20] per level | 8 |

- `λ` is the data-cost weight, 0 by default. It is used for the **Data Saver** extension (Ch. 12).
- The reward is **dense**, one per chunk. Summed over the video it equals the QoE metric exactly
  (`R_{−1}` = the default 750 kbps). So maximising return = maximising the metric used to judge us.
- **Incentive design:**
  - μ = max bitrate means one second of stall wipes out a whole top-quality chunk, so stalls are always
    worse than a quality drop.
  - The smoothness term punishes oscillation proportionally to the jump size.
  - All three terms are in the same unit (Mbps-equivalents), so they are comparable.
- `info` returned by `step` contains the individual components (`quality`, `rebuffer_penalty`,
  `smooth_penalty`, `data_penalty`). This is used for the QoE breakdown figure.

Typical scale: quality term 0.3 … 4.3 per step. A 5 s stall costs −21.5, so returns are highly skewed by
stalls. This is why advantages are normalised in training (Ch. 07).

## 3.6 Discount γ = 0.99

The effective horizon is 1/(1−γ) = 100 steps, longer than a video (48). Buffer decisions pay off tens of
chunks later (build buffer now → survive a drop later), so the agent must value the future almost as
much as the present. A small γ would produce a greedy, stall-prone policy.

## 3.7 Training-time vs test-time environment

| | Training (`RolloutCollector`) | Test (`evaluate.py`) |
|---|---|---|
| Start position in trace | random | beginning of trace |
| Delay noise | U(0.9, 1.1) | none |
| Traces | 90 % of fcc/train + hsdpa/train | hsdpa/test, fcc/test, scenarios |

Randomness in training gives more diverse experience and acts as regularisation. Determinism in testing
means every algorithm sees exactly the same network. This matches Pensieve's train and test simulators.
