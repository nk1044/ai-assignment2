# 04 — The Trace-Driven Simulator (`abrsim/`)

## 4.1 Why a simulator

Training needs ≈ 480 000 videos (30 k updates × 16). In real time that is about 3 years of streaming.
A **chunk-level, trace-driven** simulator computes each chunk download analytically from a recorded
bandwidth trace, which takes microseconds. It also allows **exact replay**: every algorithm can be tested
on the same network, which is impossible on a live network.

Chunk-level means we do not simulate packets or TCP. A chunk's download time = bytes ÷ available rate,
integrated over the trace. Pensieve validated that this level of detail is enough to transfer to real
players.

## 4.2 Constants (`abrsim/config.py`)

```python
BITRATES_KBPS = [300, 750, 1200, 1850, 2850, 4300]; A_DIM = 6; DEFAULT_ACTION = 1
CHUNK_LEN_S = 4.0;  TOTAL_CHUNKS = 48
BUFFER_MAX_S = 60.0; SLEEP_STEP_S = 0.5
LINK_RTT_S = 0.08;  PACKET_PAYLOAD_PORTION = 0.95
NOISE_LOW, NOISE_HIGH = 0.9, 1.1
S_INFO, S_LEN = 6, 8; BUFFER_NORM_S = DELAY_NORM_S = 10.0
REBUF_PENALTY = {"lin": 4.3, "log": 2.66, "hd": 8.0}; HD_REWARD = [1, 2, 3, 12, 15, 20]
```

- **RTT 80 ms:** each chunk is a separate HTTP request, so there is one request round trip of latency.
- **0.95 payload:** headers and retransmissions mean only ~95 % of link capacity carries video bytes.

## 4.3 `StreamingSimulator` — state and algorithm

Simulator state = `(ptr, last_time, chunk_idx, buffer_s)` (exposed via `get_state`/`set_state`, used by
the oracle):

- `trace.times[i], trace.bw_mbps[i]`: bandwidth `bw[i]` holds during the interval ending at `times[i]`.
- `ptr`: index of the current trace sample. `last_time`: current position on the trace clock.

### reset(trace_idx=None)
```
pick trace (random via self.rng unless given)
ptr = random integer in [1, len) if random_start else 1
last_time = times[ptr−1];  chunk_idx = 0;  buffer = 0
```

### _download_time(size) — integrate bandwidth over the trace
```
delay = 0; sent = 0
loop:
    throughput = bw[ptr] · 1e6/8 · 0.95                 # bytes/s
    duration   = times[ptr] − last_time                  # time left in this sample interval
    payload    = throughput · duration
    if throughput > 0 and sent + payload ≥ size:        # finishes inside this interval
        frac = (size − sent) / throughput
        delay += frac; last_time += frac; return delay
    sent += payload; delay += duration                  # consume whole interval
    last_time = times[ptr]; ptr += 1
    if ptr == len: ptr = 1; last_time = times[0]        # loop the trace
```
- A zero-bandwidth interval (tunnel) contributes time but no bytes. The `throughput > 0` guard prevents
  a division by zero.
- Traces shorter than a video simply loop.

### _advance_time(dt) — sleeping
Moves `ptr/last_time` forward by `dt` seconds without sending bytes. The same interval walk is used.

### download_chunk(quality) — one MDP transition
```
size  = chunk_sizes[quality, chunk_idx]
delay = _download_time(size) + RTT
if noise: delay *= noise_rng.uniform(0.9, 1.1)
rebuffer = max(delay − buffer, 0)
buffer   = max(buffer − delay, 0) + 4
if buffer > 60: sleep = ceil((buffer − 60)/0.5)·0.5; buffer −= sleep; _advance_time(sleep)
chunk_idx += 1; done = chunk_idx ≥ 48
return {delay_s, sleep_s, trace_time_s, rebuffer_s, buffer_s, chunk_size, next_chunk_sizes,
        chunks_remaining, done}
```

Subtle points (good viva material):

- **Why sleep?** Real players stop requesting when the buffer is full, otherwise they would buffer the
  whole video. While sleeping the playhead keeps playing, so the buffer drains, and the network keeps
  changing.
- **RTT and noise are added to the delay but do not advance the trace clock.** This is inherited from
  Pensieve's simulator. The effect is negligible (≤ 80 ms + 10 %).
- **Two random generators.** `self.rng` picks traces and start offsets; `self.noise_rng` (seed + 1 000 003)
  draws noise. Turning noise on/off therefore does not change which trace offsets get sampled, so
  experiments stay comparable.
- The first chunk always stalls (buffer = 0). Evaluation reports that as *startup delay* separately (Ch. 08).

## 4.4 `ABREnv` — the MDP wrapper

```python
ABREnv(traces, chunk_sizes=None, qoe="lin", data_lambda=0.0, seed=None, random_start=True, noise=True)
reset(trace_idx=None) -> state (6, 8) float32
step(action) -> (state, reward, done, info)
```

`step` does:
1. validate the action;
2. `out = sim.download_chunk(action)`;
3. compute the reward (Ch. 03);
4. measure throughput `= size·8/1e6/delay`;
5. roll the state and write the new column;
6. remember `last_action`;
7. return `info = out ∪ reward parts ∪ {action, bitrate_kbps, throughput_mbps, trace}`.

The API is Gym-style but does **not** depend on `gym`/`gymnasium`. Any RL library can still wrap it in
about 10 lines.

## 4.5 Video and traces helpers

- `video.load_chunk_sizes()` → a `(6, 48)` int array of bytes. It reads `data/video/envivio/video_size_{0..5}`
  (truncated to 48 chunks). If those are missing it falls back to `synthetic_chunk_sizes()`, which models
  nominal bitrate × L / 8 × a per-chunk "scene complexity" factor (log-normal σ = 0.15, shared by all
  levels) × per-level noise (σ = 0.05).
- `traces.load_dataset(spec)`:
  - `"hsdpa/test"` → that folder.
  - `"train"` / `"test"` → every `*/train` (or `*/test`) folder, with names prefixed by dataset.
  - `"a+b"` → union of several specs.
  - Anything else is treated as a path.
  - Files that are hidden or end in `.json`/`.md` are skipped (metadata). Extensions in general cannot be
    skipped, because FCC trace names end in `.com`.
- `traces.synthetic_trace()`: random walk in log-bandwidth, clipped to [0.2, 6] Mbps. Used in tests.

## 4.6 Tests that pin the physics (`tests/test_env.py`)

| Test | Checks |
|---|---|
| download time matches bandwidth | 4 Mbps, no noise: delay = size·8/(4e6·0.95) + 0.08 exactly |
| first chunk always rebuffers | rebuffer = delay, buffer = 4 s after the first chunk |
| high bitrate on a slow link stalls more | sanity of the dynamics |
| buffer never exceeds cap | on a 50 Mbps link the buffer ≤ 60 s and sleep > 0 happens |
| reward decomposition | the reward equals the formula, including λ |
| zero-bandwidth gap survived | a 10 s outage does not hang or crash |
| trace wraps around | a 30 s trace still yields 48 chunks |
| datasets load | 142 HSDPA test traces; 47 + 68 train; sizes increase with quality |
| scenarios/metadata | 30 scenarios, the tunnel has zeros, `.json`/`.md` are ignored |
