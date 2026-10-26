# 10 — Rebuild Guide: Writing the Project from Scratch

Build in this order. Each step has a **check** that must pass before you move on, so a later bug can
never hide an earlier one. Code references point to the real implementation.

## Step 1 — Environment setup
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install numpy torch matplotlib pytest
printf '[pytest]\npythonpath = .\ntestpaths = tests\n' > pytest.ini
```

## Step 2 — Constants (`abrsim/config.py`)
Write all numbers from Ch. 04 §4.2 in one file. Never hard-code them elsewhere.

## Step 3 — Data
1. `tools/prepare_traces.py`:
   - sparse-clone Pensieve-PPO at the pinned commit;
   - copy `src/train` (split into `trace_*` = FCC and the rest = HSDPA), `src/test` (HSDPA test) and
     `src/envivio`;
   - shuffle the FCC traces with seed 0 and take 20 % as test.
2. `abrsim/traces.py`: `Trace(name, times, bw_mbps)`, `load_trace` (np.loadtxt), `load_traces(dir)`
   (skip hidden/.json/.md), and `load_dataset(spec)` as in Ch. 04 §4.5.
3. `abrsim/video.py`: `load_chunk_sizes()` → (6, 48) array.

**Check:** 142 HSDPA test traces load; chunk sizes increase with quality.

## Step 4 — Simulator (`abrsim/abr_env.py: StreamingSimulator`)
Implement `reset`, `_download_time`, `_advance_time` and `download_chunk` exactly as the pseudo-code in
Ch. 04 §4.3. Then add `get_state/set_state`.

**Check (write these tests first):**
- constant 4 Mbps, no noise → delay = size·8/(4e6·0.95) + 0.08;
- the first chunk's rebuffer = delay, and the buffer = 4;
- on a 50 Mbps link the buffer never exceeds 60 and sleep > 0;
- a 10 s zero-bandwidth gap doesn't hang;
- a 30 s trace still gives 48 chunks.

## Step 5 — MDP wrapper (`ABREnv`)
- `reset` builds the initial 6×8 state.
- `step`:
  1. `download_chunk`;
  2. reward = q − μ·rebuf − |Δq| − λ·MB;
  3. `np.roll(state, −1, axis=1)`, zero the last column, write the new values;
  4. return `(s, r, done, info)` with the reward components in `info`.

**Check:** reward-decomposition test; 48 steps per episode; always-lowest QoE > always-highest QoE on a slow link.

## Step 6 — Policy interface + baselines
1. `baselines/base.py`: `Policy`, `buffer_s`, `past_throughputs`, `harmonic_mean`, `ConstantPolicy`, `RandomPolicy`.
2. `heuristics.py`: RB, BB, BOLA (formulas in Ch. 06).
3. `mpc.py`:
   - pre-compute `itertools.product(range(6), repeat=5)` → (7776, 5);
   - each step: predict (harmonic mean of the last 5, robust error discount);
   - vectorised simulation of all sequences over the real future chunk sizes;
   - take the argmax of the summed reward and return the first action.

**Check:** BB returns 0 below 5 s and 5 above 15 s; BOLA is monotone in the buffer; RB on a 10 Mbps link → 4300k.

## Step 7 — Evaluation harness (`evaluate.py`)
- `run_policy(policy, env, n_traces)`: per-chunk rows + per-trace metrics (Ch. 08 §8.2), with chunk 0
  as startup.
- The CLI:
  - loads a dataset;
  - builds `ABREnv(noise=False, random_start=False)` for each policy;
  - writes `<policy>.csv`, `<policy>_traces.csv` and `summary.csv`;
  - prints a table sorted by QoE.

**Check:** on hsdpa/test the ranking is RobustMPC > BOLA ≈ RB > BB > FastMPC, with RobustMPC ≈ 0.91.
**If this does not match, the simulator has a bug.**

## Step 8 — Offline optimal (`baselines/oracle.py`)
DP over `(last_action, buffer_bin)` keeping the best (total, sim_state, path) per key (Ch. 06 §6.6).
It uses a fresh noise-free `StreamingSimulator` with `set_state` to branch.

**Check:** oracle total QoE ≥ every baseline's on every trace; results stable across bin sizes.

## Step 9 — Networks (`agents/networks.py`)
`FeatureExtractor` (3 Linear(1→128) for the scalars, 3 Conv1d(1→128, k=4) for the rows, concat → 2048 →
Linear(128)), then `Actor` (→6 logits) and `Critic` (→1).

The critic gets one more Conv1d branch over the 8 future-bandwidth windows (`priv_len = 8`); the actor does not.

**Check:** output shapes (B, 6) and (B,); parameter count ≈ 266 k (actor) / 348 k (critic); the critic output changes when only the privileged input changes.

## Step 10 — Privileged information + PPO (`abrsim` + `agents/ppo.py`)
1. `StreamingSimulator.future_bandwidth()`: integrate the trace forward from `(ptr, last_time)` into 8 windows of 4 s **without moving the simulator**; `ABREnv.privileged()` returns it.
2. `PPOConfig` dataclass; `PPOAgent` with:
   - `act`;
   - `advantages(rollout)` (GAE with `V(s, f)`);
   - `update` (old log-probs once, then 4 epochs × 4 minibatches of the clipped surrogate + entropy, critic MSE, grad clipping, target-KL early stop);
   - `entropy_weight` (linear decay);
   - `save/load`.

**Check:**
- on a hand-made trace the windows equal hand-computed means, and the simulator state is unchanged;
- with a zero critic and λ = 1 the GAE equals discounted returns;
- one update gives finite losses, KL < 0.5 and clip_frac in [0, 1];
- a tiny target KL stops the epochs early;
- save→load reproduces the probabilities.

## Step 11 — Rollouts and the training script
1. `RolloutCollector`: 16 envs, lock-step, batched `act`. Record `privileged()` **before** `env.step` (decision time), then stack into (48, 16, …).
2. `RLPolicy`: wraps the agent (greedy) in the `Policy` interface.
3. `train.py`:
   - 90/10 split of the training pool;
   - loop collect→update;
   - validate every 250 iterations with `run_policy` → `log.csv` / `best.pt`;
   - checkpoint every 1000 iterations.

**Check:** a 2-iteration smoke run writes all files; then a 500-iteration trial runs without errors.

## Step 12 — Train and evaluate
```bash
python train.py --run-name ppo_asym                        # ≈ 1.05 h
python evaluate.py --oracle --agent PPO=runs/ppo_asym/best.pt
python tools/record_experiment.py --run runs/ppo_asym --name PPO_asym --algo "PPO, asymmetric critic"
```
Watch entropy in the log. If it stays at ≈ 1.79 (uniform) for a long time, lower `--entropy-start`.

## Step 13 — Figures, scenarios, logger
`plots/style.py` (colour map) → `plots/make_figures.py`; `tools/make_scenarios.py`; `tools/trace_logger.py`
(background download thread + per-second sampler + rate cap).

## Common pitfalls (all hit during this project)
| Pitfall | Symptom | Fix |
|---|---|---|
| A folder named `env/` | git silently ignores it (`ENV/` rule + case-insensitive macOS) | named the package `abrsim/` |
| Noise and start offsets sharing one RNG | switching noise off changes which traces are tested | two RNG streams |
| Random start offsets at test time | FCC average swung −0.3 → +0.9 | deterministic test protocol |
| Counting the first chunk in QoE | every policy pays a huge constant penalty | report it as startup delay |
| Logger timeout ≤ TLS handshake | trace of all zeros | background thread + sampler |
| Uncapped logger | 43 MB in 4 s on fast links | `--cap-mbps 8` |
| Entropy weight too high for normalised advantages | the policy stays uniform for 15 k iterations | lower `entropy_start` |
| Recording privileged info *after* `env.step` | the critic sees the future of the *next* state (off by one decision) | read it before stepping; test checks it |
| Validation on 12 single videos | the better-validating A2C v2 tested worse than v1 | 4 random starts per validation trace |
| Matplotlib PDF timestamps | figures always appear as modified in git | `metadata={"CreationDate": None}` |
