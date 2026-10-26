# 09 — Technology and Design Choices (and why)

## 9.1 Stack

| Tool | Used for | Why this and not the alternative |
|---|---|---|
| **Python 3.12** | everything | The RL/ML ecosystem standard. 3.12 rather than the system's 3.15, because PyTorch wheels lag new Python releases |
| **NumPy** | simulator, baselines, oracle | The env and baselines don't need autodiff. Pure NumPy keeps them fast to import, dependency-light and usable without torch. MPC's 7776-sequence search is vectorised |
| **PyTorch** | actor/critic, training | Dynamic graphs make debugging easy, and it is the research standard. TensorFlow 1 (Pensieve's original) is obsolete; JAX is overkill for a 266 k-parameter model |
| **Own PPO (~170 lines)** instead of Stable-Baselines3 | algorithm | Every line can be explained in a viva. It needs a custom architecture (per-row conv branches, λ conditioning, a **privileged critic input that the actor must not see**) and a custom validation/checkpoint loop. SB3 assumes actor and critic share the same observation and would hide the maths |
| **No gym/gymnasium dependency** | env API | Same `reset/step` contract without the dependency. It can be wrapped in a few lines if needed |
| **matplotlib** | figures | Publication-quality PDF for LaTeX plus PNG for slides from one script |
| **pytest** (38 tests) | correctness | The physics, the reward formula and the oracle bound are verified, not assumed |
| **CSV** outputs | results | Human-readable, diff-able, opens in Excel/pandas. No database needed |
| **Git LFS?** No | data | The data is only 2.9 MB, so it is committed directly. `prepare_traces.py` can rebuild it from a pinned commit |

## 9.2 Algorithm choice: why actor-critic, A2C first, then PPO

| Candidate | Verdict |
|---|---|
| Tabular Q-learning / SARSA | ✗ The 25-dim continuous state cannot be tabulated; discretising loses the throughput-history signal |
| DQN | Possible (discrete actions). But value-based methods with a max-operator overestimate under the heavy-tailed, stall-dominated rewards here, need a replay buffer and target networks, and give a deterministic greedy policy during learning (exploration only via ε) |
| REINFORCE | Unbiased but very high variance (a single stall dominates a return). The critic in A2C fixes this |
| A2C ✓ (first agent) | Policy gradient for a direct stochastic policy, a learned baseline (critic) for variance reduction, GAE, entropy-driven exploration, on-policy with parallel envs. Pensieve showed this family works on exactly this task. Beat every baseline (0.934) |
| A3C | Same objective, asynchronous. A2C is deterministic and equally good; asynchrony gives no benefit on one machine |
| **PPO** ✓ (current) | Same actor-critic structure, but the clipped surrogate makes it safe to reuse each rollout for several epochs: more learning per simulated video and steadier updates (target-KL early stop). It is the de-facto standard on-policy method |
| Model-based / MPC | That is the RobustMPC baseline. It needs a bandwidth model, which is exactly what fails |

## 9.3 Design choices worth defending

| Choice | Reason |
|---|---|
| Trace-driven simulator | ≈ 30 000× faster than real time (≈ 23 M chunks ≈ 1 000 days of video in 53 min); exact replay for fair comparison; standard in ABR research |
| Real traces, not synthetic | the agent must learn the structure of real networks (Ch. 05) |
| Real video sizes (VBR) | chunk size varies with content; the state includes it |
| Separate actor and critic networks | the critic's large-magnitude loss would disturb shared features, and a shared trunk would leak privileged information into the actor |
| Asymmetric (privileged) critic | the simulator knows the future bandwidth; the critic is training-only, so it may use it. Sharper V gives lower-variance advantages. The critic also gets the observation (unbiasedness) |
| 1-D conv over history | detects temporal patterns with few weights (Pensieve architecture) |
| Advantage normalisation + grad clipping | stall rewards are heavy-tailed |
| Validation split taken from training data | model selection never touches the test set |
| Noise-free deterministic tests | identical conditions for all algorithms; reproducible numbers |
| Startup delay excluded from QoE | the standard convention; otherwise 1.7–4 s × 4.3 of unavoidable penalty dominates |
| Same reward used by MPC and the agent | a fair comparison: both optimise the same objective |
| Oracle by DP | turns "is 0.93 good?" into "how much of the possible gain did we get?" |
| Shared `Policy` interface | one evaluation path for all algorithms, so no accidental differences |
| Fixed colour per algorithm in plots | colour follows the entity across every figure; the oracle is neutral grey |
| Pinned-commit data download | reproducible data; nobody depends on a moving branch |
