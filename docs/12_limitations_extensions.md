# 12 — Limitations and Extensions

## 12.1 Limitations (state them before the examiner does)

| Limitation | Impact | Mitigation / argument |
|---|---|---|
| Traces are open-loop (don't react to our downloads) | ignores TCP slow start, congestion caused by us | standard in ABR research; Pensieve transferred to real players |
| Partial observability, 8-step window | can't remember older events | LSTM extension |
| One video (EnvivioDash3) | sizes/ladder are specific to one video | could train on many videos; the state already includes the real chunk sizes |
| Linear QoE is a model of perception | real perception is non-linear | QoE_log / QoE_hd variants supported; VMAF-based reward is an extension |
| Pre-filtered bandwidth range (≈ 0.2–6 Mbps) | very fast links are uninteresting for this ladder | the trace logger caps at 8 Mbps for the same reason |
| A2C runs under-tuned (entropy) | only +3 % over RobustMPC | fixed in the final PPO run (+7 %) |
| FCC test set is only 12 traces | high variance (± 1.8) | HSDPA (142) is the main benchmark |

## 12.2 Extensions and their status

| Id | Extension | Status |
|---|---|---|
| U4 | Named stress-test scenarios | ✅ done (`tools/make_scenarios.py`, figure g7) |
| U1 | Own Indian traces | ✅ tool + protocol; recordings still to do |
| U2 | Data Saver (preference-conditioned on λ) | ✅ code (`--lambda-range`); needs a training run |
| — | Offline-optimal upper bound | ✅ done |
| — | Asymmetric (privileged) critic + PPO | ✅ done: 0.971 HSDPA vs A2C 0.934 / RobustMPC 0.907 |
| U3 | Explainability: policy heatmap (buffer × throughput → chosen bitrate, vs BB/RB rules), permutation feature importance | planned (Part 6b) |
| P4 | "Watch it learn" timelapse from `ckpt_*.pt` | planned (checkpoints saved every 500 PPO iterations; A2C runs saved every 1000) |
| P1–P2 | Web demo: side-by-side players, draw-your-network | ✅ done (`python demo/server.py`) |
| — | DQN comparison, LSTM policy, VMAF reward, live-streaming mode | ideas |

## 12.3 How the Data Saver works (U2)

- Reward `r = q − μT − |Δq| − λ·MB`. A larger λ makes every megabyte cost QoE.
- Training: `python train.py --run-name ppo_datasaver --lambda-range 0 2`. Each video samples
  λ ~ U(0, 2), and λ is fed as an extra input to both networks (a Linear(1→128) branch in the feature
  extractor).
- The agent learns a *family* of policies indexed by λ. At test time a slider sets λ, and the same model
  trades quality for data: no retraining per user preference. This is multi-objective /
  preference-conditioned RL, which goes beyond a single fixed reward.
