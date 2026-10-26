# 11 — Viva Q&A

## Formulation

**Q: Why is this an RL problem and not supervised learning?**
There are no labels for the "correct" bitrate. The quality of a decision is only known later, through
its effect on the buffer and future stalls, so it is a sequential decision problem with delayed reward.
You could imitate the oracle (supervised learning), but the oracle uses future information the agent
never has, so its labels are partly unlearnable.

**Q: Is your state Markov?**
Not strictly. Future bandwidth is hidden, so it is a POMDP. The last 8 throughputs and download times
summarise the recent network behaviour, which makes it approximately Markov. It is the standard
approximation; an LSTM over the full history would be the principled fix.

**Q: Why include the last bitrate in the state?**
The reward has a switching term `|q(R_t) − q(R_{t−1})|`. Without `R_{t−1}` in the state, the reward would
not be a function of (s, a).

**Q: Why μ = 4.3?**
It equals the highest bitrate in Mbps, so one second of stall cancels the value of one top-quality chunk.
That keeps stalls always worse than dropping quality. It is the value used by MPC/Pensieve, so results
are comparable to the literature.

**Q: Why discrete actions?**
A DASH manifest offers a finite ladder; a player cannot request 2.1 Mbps if no such encoding exists.

**Q: Why γ = 0.99 when an episode is only 48 steps?**
Buffer decisions pay off many chunks later. A horizon of ~100 steps means the agent values the whole
remaining video almost equally.

**Q: Why is the reward per chunk rather than at the end?**
Dense rewards give much better credit assignment. And the sum of per-chunk rewards *is* the QoE
metric, so nothing changes in the objective.

## Simulator and data

**Q: If you train in simulation, what is the data for?**
The simulator is only arithmetic. The traces are the real-world part: how bandwidth actually evolves.
The agent learns the statistics of real networks through them. See Ch. 05 §5.1.

**Q: How realistic is the simulator?**
It is a chunk-level simulation: download time is integrated over the bandwidth trace, plus RTT, payload
overhead, buffer and sleeping. Pensieve showed this level transfers to real players. Main gap: the trace
doesn't react to our traffic (no TCP dynamics or congestion feedback).

**Q: How did you validate the simulator?**
- Unit tests pin the exact download-time formula and the buffer arithmetic.
- The baselines reproduce the ranking and the magnitude (RobustMPC ≈ 0.9) reported in the literature.
- The oracle is ≥ all baselines on every trace.

**Q: Could the test set leak into training?**
No:
- The HSDPA test traces are disjoint from train (checked).
- FCC test is a held-out 20 %.
- Validation for model selection comes from the training pool.
- The scenarios are never used in training.

## Algorithm

**Q: Why actor-critic and not DQN?** See Ch. 09 §9.2. In short: continuous state → function
approximation, and a stochastic policy with a critic baseline handles heavy-tailed stall rewards better
than Q-learning's max-operator.

**Q: Why did you move from A2C to PPO?**
A2C may only take one gradient step per rollout; after that the data is off-policy. PPO's clipped ratio
keeps the new policy close to the old one, so each rollout can be reused for 4 epochs. That gives more
learning per simulated video and steadier updates, with target-KL early stopping as a safety net.

**Q: What is an asymmetric actor-critic?**
The actor and critic get different inputs. Our critic also sees the *true* bandwidth of the next 32 s
(read from the trace by the simulator). The actor sees only what a real player sees.

**Q: Isn't that cheating?**
No. The critic is only used during training to compute advantages; it is thrown away at deployment.
The deployed actor never sees the future. The trick only improves the *training signal*.

**Q: Does it make the problem an MDP?**
Not for the agent. The actor still faces a POMDP. Only the critic's value estimate becomes closer to
Markov, so it can tell "about to enter a tunnel" from "about to leave one" (an identical past, a
different future) and give sharper advantages.

**Q: Why give the critic the observation too, not just the future?**
A critic conditioned only on hidden state can bias the policy gradient under partial observability
(Baisero & Amato, 2022). Conditioning on observation + privileged information avoids it.

**Q: Why didn't you start with the asymmetric critic?**
We first reproduced the standard, published design (Pensieve's symmetric A2C) to validate the
simulator and pipeline. The symmetric runs then serve as the control for measuring what the privileged
critic adds.

**Q: What does the critic do?**
It estimates V(s), which gives a baseline for the policy gradient (lower variance without bias) and the
TD errors for GAE.

**Q: What is GAE and why λ = 0.95?**
An exponentially weighted sum of TD errors. λ = 0 is low-variance but biased (trusts the critic); λ = 1 is
Monte-Carlo (unbiased, high variance). 0.95 is the standard compromise.

**Q: Why the entropy bonus and why decay it?**
Without it, the policy collapses early onto a safe action (e.g., always lowest) and never discovers
better ones. Decaying it lets the final policy become decisive. Our run showed the start value 1.0 was
too high: the policy stayed uniform for about 15 k iterations.

**Q: Why normalise advantages?**
A single 5 s stall gives a reward of −21.5 versus ~+1 for normal chunks. Normalisation keeps the
gradient scale stable between batches.

**Q: Why 1-D convolutions?**
They detect temporal patterns (drops, trends) in the 8-step histories with shared weights. This is the
proven Pensieve architecture.

**Q: Sync A2C vs A3C?** (first agent)
The same parallel experience; synchronous is deterministic and simpler. With batched inference it is
efficient on one CPU.

**Q: What do the PPO numbers in the log mean?**
- `KL`: how far one update moved the policy (healthy < 0.03).
- `clip_frac`: the share of samples where clipping was active (~0.1 is healthy).
- `ep`: PPO epochs actually used (4 unless target-KL stopped early).

**Q: How long does training take / how much experience?**
30 k updates × 16 videos × 48 chunks ≈ 23 M decisions ≈ 1 000 days of video. 53 minutes on a laptop CPU.

## Results

**Q: Does RL beat the baselines?**
Yes. On 142 unseen 3G traces the first A2C agent scores 0.934 vs RobustMPC's 0.907, and it stalls 5× less
(0.17 vs 0.91 s per video). It also wins on FCC (0.644 vs 0.604) and the scenarios (1.047 vs 0.738;
metro tunnel +0.16 vs −1.79). PPO results are in `experiments/README.md`. It is still 18 % below the offline optimum (1.145).

**Q: Why is the gain smaller than Pensieve's 12–25 %?**
- Pensieve compared against MPC variants under its own setup.
- Our first run wasted about half its budget in near-uniform exploration (entropy too high).
- Our RobustMPC uses the exact same reward and the true chunk sizes, so it is a strong baseline.

Tuning `entropy_start` and training longer is the next step.

**Q: What is the oracle and why is it optimal?**
DP over (last quality, buffer bin) with full knowledge of the trace (Ch. 06 §6.6). Histories with the
same last quality and buffer are interchangeable for the future, so keeping the best per key is the
Bellman principle. It is converged in bin size.

**Q: Why exclude the first chunk?**
The buffer is empty, so every policy stalls by roughly the chunk download time. It is a startup delay,
reported separately, as in MPC and Pensieve.

**Q: Why is validation QoE negative but test QoE positive?**
The validation set contains FCC traces with full outages: ≈ 10 s of unavoidable stall per video
whatever the bitrate. Validation is only for comparing checkpoints with each other.

## Bonus / uniqueness

**Q: What is new compared to Pensieve?**
- Named stress-test scenarios that expose where each algorithm fails (RobustMPC −1.79 in a tunnel).
- A preference-conditioned Data Saver agent: λ is an input, so one model covers a whole quality-vs-data
  slider.
- A tool and protocol for recording our own Indian 4G/5G traces, to test cross-country generalisation.
- An offline-optimal bound to quantify headroom.
- Fully reproducible pipeline with tests.
