"""Trace-driven, chunk-level video streaming simulator and the RL environment on top of it.

``StreamingSimulator`` implements the player/network dynamics of README §4.3 (download over a bandwidth
trace, rebuffering, buffer update, sleeping when the buffer is full).

``ABREnv`` wraps it as an MDP with a Gym-style API:
    state  = 6 x 8 matrix (README §4.1)
    action = bitrate index in [0, 6)
    reward = q(R_t) - mu * T_t - |q(R_t) - q(R_{t-1})| - lambda * size_MB   (README §4.4, §12 U2)
"""

import numpy as np

from . import config as C
from .video import load_chunk_sizes


class StreamingSimulator:
    def __init__(self, traces, chunk_sizes=None, seed=None, random_start=True, noise=True):
        self.traces = traces
        self.chunk_sizes = load_chunk_sizes() if chunk_sizes is None else np.asarray(chunk_sizes)
        self.n_chunks = self.chunk_sizes.shape[1]
        self.rng = np.random.default_rng(seed)
        self.random_start = random_start
        self.noise = noise
        self.trace_idx = -1

    def reset(self, trace_idx=None):
        """Start a new video on a trace (random one unless ``trace_idx`` is given)."""
        if trace_idx is None:
            trace_idx = int(self.rng.integers(len(self.traces)))
        self.trace_idx = trace_idx
        self.trace = self.traces[trace_idx]
        self.ptr = int(self.rng.integers(1, len(self.trace))) if self.random_start else 1
        self.last_time = self.trace.times[self.ptr - 1]
        self.chunk_idx = 0
        self.buffer_s = 0.0

    def _next_ptr(self):
        self.ptr += 1
        if self.ptr >= len(self.trace):        # loop the trace if the video outlasts it
            self.ptr = 1
            self.last_time = self.trace.times[0]

    def _download_time(self, size_bytes):
        """Wall-clock seconds to push ``size_bytes`` through the trace from the current time."""
        delay, sent = 0.0, 0.0
        while True:
            throughput = self.trace.bw_mbps[self.ptr] * 1e6 / 8 * C.PACKET_PAYLOAD_PORTION   # bytes/s
            duration = self.trace.times[self.ptr] - self.last_time
            payload = throughput * duration
            if throughput > 0 and sent + payload >= size_bytes:
                frac = (size_bytes - sent) / throughput
                delay += frac
                self.last_time += frac
                return delay
            sent += payload
            delay += duration
            self.last_time = self.trace.times[self.ptr]
            self._next_ptr()

    def _advance_time(self, dt):
        """Let ``dt`` seconds pass on the trace without downloading (player sleeping)."""
        while True:
            duration = self.trace.times[self.ptr] - self.last_time
            if duration > dt:
                self.last_time += dt
                return
            dt -= duration
            self.last_time = self.trace.times[self.ptr]
            self._next_ptr()

    def download_chunk(self, quality):
        """Download the next chunk at ``quality``; returns a dict describing what happened."""
        size = int(self.chunk_sizes[quality, self.chunk_idx])
        delay = self._download_time(size) + C.LINK_RTT_S
        if self.noise:
            delay *= self.rng.uniform(C.NOISE_LOW, C.NOISE_HIGH)

        rebuffer = max(delay - self.buffer_s, 0.0)
        self.buffer_s = max(self.buffer_s - delay, 0.0) + C.CHUNK_LEN_S

        sleep = 0.0
        if self.buffer_s > C.BUFFER_MAX_S:
            sleep = np.ceil((self.buffer_s - C.BUFFER_MAX_S) / C.SLEEP_STEP_S) * C.SLEEP_STEP_S
            self.buffer_s -= sleep
            self._advance_time(sleep)

        self.chunk_idx += 1
        done = self.chunk_idx >= self.n_chunks
        next_sizes = (self.chunk_sizes[:, self.chunk_idx] if not done
                      else np.zeros(self.chunk_sizes.shape[0], dtype=np.int64))
        return {
            "delay_s": delay,
            "sleep_s": sleep,
            "rebuffer_s": rebuffer,
            "buffer_s": self.buffer_s,
            "chunk_size": size,
            "next_chunk_sizes": next_sizes,
            "chunks_remaining": self.n_chunks - self.chunk_idx,
            "done": done,
        }


def quality_value(action, qoe="lin"):
    """q(R) for the chosen QoE variant."""
    if qoe == "lin":
        return C.BITRATES_KBPS[action] / 1000.0
    if qoe == "log":
        return float(np.log(C.BITRATES_KBPS[action] / C.BITRATES_KBPS[0]))
    if qoe == "hd":
        return float(C.HD_REWARD[action])
    raise ValueError(f"unknown QoE variant {qoe!r}")


class ABREnv:
    """MDP wrapper: ``reset() -> state``, ``step(action) -> (state, reward, done, info)``."""

    observation_shape = (C.S_INFO, C.S_LEN)
    n_actions = C.A_DIM

    def __init__(self, traces, chunk_sizes=None, qoe="lin", data_lambda=0.0, seed=None,
                 random_start=True, noise=True):
        self.sim = StreamingSimulator(traces, chunk_sizes, seed, random_start, noise)
        self.qoe = qoe
        self.mu = C.REBUF_PENALTY[qoe]
        self.data_lambda = data_lambda

    def reset(self, trace_idx=None):
        self.sim.reset(trace_idx)
        self.last_action = C.DEFAULT_ACTION
        self.state = np.zeros(self.observation_shape, dtype=np.float32)
        self.state[0, -1] = C.BITRATES_KBPS[self.last_action] / max(C.BITRATES_KBPS)
        self.state[4, :C.A_DIM] = self.sim.chunk_sizes[:, 0] / 1e6
        self.state[5, -1] = 1.0
        return self.state.copy()

    def step(self, action):
        action = int(action)
        if not 0 <= action < C.A_DIM:
            raise ValueError(f"action must be in [0, {C.A_DIM}), got {action}")
        out = self.sim.download_chunk(action)

        q, q_prev = quality_value(action, self.qoe), quality_value(self.last_action, self.qoe)
        size_mb = out["chunk_size"] / 1e6
        parts = {
            "quality": q,
            "rebuffer_penalty": self.mu * out["rebuffer_s"],
            "smooth_penalty": abs(q - q_prev),
            "data_penalty": self.data_lambda * size_mb,
        }
        reward = parts["quality"] - parts["rebuffer_penalty"] - parts["smooth_penalty"] - parts["data_penalty"]

        throughput_mbps = out["chunk_size"] * 8 / 1e6 / out["delay_s"]
        s = np.roll(self.state, -1, axis=1)               # shift history one step left
        s[:, -1] = 0.0
        s[0, -1] = C.BITRATES_KBPS[action] / max(C.BITRATES_KBPS)
        s[1, -1] = out["buffer_s"] / C.BUFFER_NORM_S
        s[2, -1] = throughput_mbps
        s[3, -1] = out["delay_s"] / C.DELAY_NORM_S
        s[4, :] = 0.0
        s[4, :C.A_DIM] = out["next_chunk_sizes"] / 1e6
        s[5, -1] = out["chunks_remaining"] / self.sim.n_chunks
        self.state = s
        self.last_action = action

        info = {**out, **parts, "action": action, "bitrate_kbps": C.BITRATES_KBPS[action],
                "throughput_mbps": throughput_mbps, "trace": self.sim.trace.name}
        return s.copy(), float(reward), out["done"], info
