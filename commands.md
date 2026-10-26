# Commands

Run everything from the project root (`ai-assignment-2/`).

## 0. One-time setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# only needed if data/ is missing (it is already committed)
python tools/prepare_traces.py
python tools/make_scenarios.py
```

Check that everything works (≈ 5 s):

```bash
pytest -q          # expect: 33 passed
```

> Every command below assumes the venv is active (`source .venv/bin/activate`).

---

## 1. Train the agent  (≈ 1–1.5 h on a laptop CPU)

```bash
python train.py --run-name a2c_lin 2>&1 | tee runs_a2c_lin.log
```

- Prints a line every 250 iterations: training reward, **validation QoE**, best so far, entropy.
  Validation QoE should rise from negative values towards **~0.9–1.1**. RobustMPC scores ~0.9 on the test set.
- Output goes to `runs/a2c_lin/`:
  - `best.pt` (the model to present)
  - `final.pt`
  - `ckpt_*.pt` (one every 1000 iterations, for the timelapse)
  - `log.csv` (data for the learning curve)
- Safe to stop with `Ctrl+C`: `best.pt` keeps the best model saved so far.
- Quick trial run first (≈ 2 min), to see that it works on your machine:

  ```bash
  python train.py --run-name trial --iters 500 --eval-every 100
  ```

### Optional extra runs (for the bonus sections)

```bash
# Data-saver agent (U2): one model for the whole "max quality <-> data saver" slider
python train.py --run-name a2c_datasaver --lambda-range 0 2

# Generalization (G7): train only on FCC broadband, then test on Norway 3G
python train.py --run-name a2c_fcc_only --train-set fcc/train
```

---

## 2. Test the trained agent against all baselines

Each command prints a results table and saves logs to `results/baselines/<dataset>_lin/`.

```bash
# Main result: Norway 3G test set, 142 traces (≈ 2 min including the offline optimal)
python evaluate.py --oracle --agent A2C=runs/a2c_lin/best.pt

# FCC broadband test set, 12 traces
python evaluate.py --oracle --agent A2C=runs/a2c_lin/best.pt --traces fcc/test

# Stress-test scenarios: metro tunnel, stadium crowd, highway, flaky Wi-Fi, rural (30 traces)
python evaluate.py --oracle --agent A2C=runs/a2c_lin/best.pt --traces scenarios
```

The table columns are:
- QoE per chunk (higher is better)
- average bitrate
- rebuffering seconds per video
- quality switches per video
- startup delay

Other QoE definitions (README §4.4):

```bash
python evaluate.py --oracle --agent A2C=runs/a2c_lin/best.pt --qoe log
python evaluate.py --oracle --agent A2C=runs/a2c_lin/best.pt --qoe hd
```

> An agent trained with `--qoe lin` is optimized for QoE_lin. For a fair QoE_log/QoE_hd comparison, train
> with the matching `--qoe` flag.

Optional runs:

```bash
# Generalization run: compare the FCC-only agent with the full agent on Norway 3G
python evaluate.py --agent A2C=runs/a2c_lin/best.pt --agent A2C_fcc_only=runs/a2c_fcc_only/best.pt \
    --out results/generalization

# Data-saver agent at three slider positions (λ = data cost per MB)
for L in 0 0.5 1.5; do
  python evaluate.py --no-baselines --agent A2C_datasaver=runs/a2c_datasaver/best.pt \
      --data-lambda $L --out results/datasaver_$L
done
```

---

## 3. Make the report / presentation figures

Run this after the three `evaluate.py` commands in step 2:

```bash
python plots/make_figures.py
```

It writes PNG and PDF files to `plots/figures/`. A2C is added to every chart automatically.

| File | Shows |
|---|---|
| `g1_training.png` | learning curve: validation QoE vs. iteration, with RobustMPC and the optimum as reference lines |
| `g2_qoe_bars.png` | average QoE of every algorithm (HSDPA and FCC) |
| `g3_breakdown_*.png` | where QoE comes from: bitrate vs. stall penalty vs. switching penalty |
| `g4_qoe_cdf.png` | QoE distribution across all test traces |
| `g5_timeseries_hsdpa_test.png` | one video: bandwidth, chosen bitrate and buffer for A2C / RobustMPC / BB / optimum |
| `g6_tradeoff_*.png` | quality vs. stalls scatter |
| `g7_scenarios.png` | stress-test scenarios |
| `data_traces.png` | what real network traces look like |

---

## 4. Live demo during the presentation (≈ 3 min total)

```bash
source .venv/bin/activate

# 1) the simulator and baselines are verified
pytest -q

# 2) RL agent vs. classic algorithms on 142 real 3G traces
python evaluate.py --oracle --agent A2C=runs/a2c_lin/best.pt

# 3) the "metro tunnel" story: where RobustMPC breaks
python evaluate.py --agent A2C=runs/a2c_lin/best.pt --traces scenarios

# 4) open the figures
open plots/figures/g5_timeseries_hsdpa_test.png plots/figures/g7_scenarios.png plots/figures/g2_qoe_bars.png
```

(`open` is macOS; on Linux use `xdg-open`.)

---

## 5. Record your own network traces (U1)

```bash
python tools/trace_logger.py --name delhi_metro_yellow_line --duration 600 \
    --network "Jio 4G" --notes "Rajiv Chowk -> HUDA City Centre"

# after recording ~10 traces:
python evaluate.py --oracle --agent A2C=runs/a2c_lin/best.pt --traces india
```

The logger uses at most ~60 MB of mobile data per minute (8 Mbps cap), and stops at 400 MB by default
(`--max-mb`). The full recording protocol is in `data/traces/india/README.md`.
