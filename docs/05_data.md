# 05 — Data

## 5.1 What data is for, when the model learns in simulation

The simulator is only the **rules** (download time = size / bandwidth, buffer arithmetic). The thing it
cannot invent is **how real bandwidth behaves over time**. The traces provide exactly that: the hidden
part of the transition function. The agent learns the statistical structure of real networks (slow decays,
bursts, outages, recoveries) by replaying them thousands of times. Synthetic random noise would
teach it the wrong structure.

## 5.2 Network traces

**Format** (all datasets): plain text, one sample per line, `<time_s>\t<bandwidth_Mbps>`. This is the
"cooked" format released with Pensieve. Samples are ~1 s apart; HSDPA has irregular spacing.

| Dataset | Origin | Train | Test | Mean bw per trace | Duration |
|---|---|---|---|---|---|
| Norway HSDPA 3G | Riiser et al., MMSys 2013 — phones on buses, trams, metro, ferries, trains, cars around Oslo | 68 | 142 | train 0.13–3.59 Mbps (median 1.20); test 0.60–4.59 (median 1.35) | 36–4207 s |
| FCC | US FCC *Measuring Broadband America* — home broadband web-browsing logs | 47 | 12 | 0.19–2.92 Mbps (median 1.19) | 65–1860 s |

- **Source and reproducibility:** `tools/prepare_traces.py` sparse-clones github.com/godka/Pensieve-PPO at a
  **pinned commit** (`a1b2579…`). It copies `src/train`, `src/test` and `src/envivio`, then re-creates
  `data/` exactly.
- **Pre-filtering** (done by the Pensieve authors): traces outside the range where the decision matters are
  removed. If bandwidth is always > 6 Mbps, every algorithm just picks 4.3 Mbps.
- **Splits:**
  - HSDPA train/test as released. I verified that no test trace's first 10 samples appear inside any train
    trace, so there is no leakage.
  - FCC had no test split, so `prepare_traces.py` holds out 20 % with `np.random.default_rng(0)`.
- **Long HSDPA file names** (`bus.ljansbakken-oslo-report.2010-09-28_1407CEST.log`) are shortened to
  `bus_ljansbakken_2010-09-28_1407CEST`.
- **FCC quirk:** 15 of 59 FCC traces contain exact-zero bandwidth stretches (outages). Even the optimum
  stalls ≈ 4.8 s per video on FCC test. This also explains the negative validation QoE during training (Ch. 07).

**Training set** = `"train"` = fcc/train + hsdpa/train = 115 traces. `train.py` then holds out 10 %
(12 traces, seed 0) for validation → 103 for learning.

## 5.3 Video: EnvivioDash3 chunk sizes

`data/video/envivio/video_size_{0..5}`: one integer (bytes) per line, 49 lines (only the first 48 are used).
It is the standard DASH reference test video used in ABR papers. 4 s chunks, VBR-encoded, so sizes vary
per chunk.

| Level | 300k | 750k | 1200k | 1850k | 2850k | 4300k |
|---|---|---|---|---|---|---|
| mean MB / chunk | 0.151 | 0.375 | 0.599 | 0.921 | 1.419 | 2.140 |

These match bitrate × 4 s / 8 on average. Individual chunks deviate, which is why the state contains the
next chunk's real sizes.

## 5.4 Stress-test scenarios (`tools/make_scenarios.py` → `data/traces/scenarios/`)

Five named generators, 300 s each, 6 deterministic variants each (seed = 1000·scenario + variant), 30 traces:

| Scenario | Generator |
|---|---|
| `metro_tunnel` | 2.5–3.5 Mbps (±20 % log-normal jitter); every 40–70 s a **0 Mbps** gap of 15–30 s |
| `stadium_crowd` | geometric decay from 3.5–4.5 to 0.3–0.5 Mbps over 300 s, ±20 % jitter |
| `highway_handover` | 1.8–2.8 Mbps; every 15–30 s a 2–5 s dip to 0.05–0.2 Mbps |
| `flaky_wifi` | alternates 3–5 Mbps and 0.3–0.7 Mbps every 2–6 s |
| `rural_edge` | log random walk clipped to 0.35–1.3 Mbps |

They are used **only for testing**, never for training. So good results there mean genuine generalisation.

## 5.5 Our own traces (`tools/trace_logger.py` → `data/traces/india/`)

- A background thread downloads continuously from `speed.cloudflare.com/__down?bytes=…`. The main thread
  reads the byte counter every `--interval` (1 s) and writes Mbps. An outage therefore shows up correctly
  as ~0 rather than blocking the sampler (the first version used a socket timeout and recorded zeros).
- `--cap-mbps 8` throttles reading. This saves mobile data (≤ ~60 MB/min). The ladder tops out at 4.3 Mbps,
  so capping at 8 changes no decision. `--max-mb 400` is a hard stop.
- A `<name>.meta.json` sidecar records network, route, data used and the cap. The protocol is in
  `data/traces/india/README.md`.
