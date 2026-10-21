# Data

Created by `python tools/prepare_traces.py` (re-running it reproduces this folder exactly).

| Folder | Traces | Source |
|---|---|---|
| `traces/hsdpa/train` | 68 | Norway HSDPA 3G commute logs (bus, car, ferry, metro, train, tram around Oslo) — Riiser et al., MMSys 2013 |
| `traces/hsdpa/test` | 142 | Same dataset, disjoint test split |
| `traces/fcc/train` | 47 | FCC *Measuring Broadband America* web-browsing throughput logs |
| `traces/fcc/test` | 12 | 20 % of the FCC traces, held out with seed 0 |
| `video/envivio/` | 6 files × 48 chunks | Per-chunk sizes (bytes) of the EnvivioDash3 test video, 4 s chunks, 300–4300 kbps |

All traces are the pre-processed ("cooked") versions released with Pensieve (Mao et al., SIGCOMM 2017),
taken from the mirror at github.com/godka/Pensieve-PPO (commit `a1b2579`).

**Format:** one sample per line, `<time_s> <bandwidth_Mbps>`. Traces shorter than a video are looped
by the simulator.

| Dataset | Mean bandwidth per trace | Duration | Notes |
|---|---|---|---|
| FCC | 0.19 – 2.92 Mbps (median 1.19) | 65 – 1860 s | 15 traces contain zero-bandwidth outages |
| HSDPA train | 0.13 – 3.59 Mbps (median 1.20) | 36 – 4207 s | highly variable mobile 3G |
| HSDPA test | 0.60 – 4.59 Mbps (median 1.35) | 44 – 317 s | no overlap with train |
