"""Model Predictive Control (Yin et al., SIGCOMM'15).

At every chunk: predict bandwidth for the next ``horizon`` chunks, enumerate all 6^horizon bitrate
sequences, simulate buffer/rebuffering for each, and apply the first action of the best sequence.
RobustMPC divides the prediction by (1 + max recent prediction error) to be conservative.
"""

import itertools

import numpy as np

from abrsim import config as C
from abrsim.abr_env import quality_value

from .base import Policy, buffer_s, harmonic_mean, past_throughputs


class MPC(Policy):
    def __init__(self, horizon=5, robust=True, window=5):
        self.horizon = horizon
        self.robust = robust
        self.window = window
        self.name = "RobustMPC" if robust else "FastMPC"
        self.combos = np.array(list(itertools.product(range(C.A_DIM), repeat=horizon)))

    def reset(self):
        self.last_prediction = None
        self.errors = []

    def predict_mbps(self, state):
        tp = past_throughputs(state)
        if len(tp) == 0:
            return None
        if self.last_prediction is not None:
            self.errors.append(abs(self.last_prediction - tp[-1]) / tp[-1])
        prediction = harmonic_mean(tp[-self.window:])
        self.last_prediction = prediction
        if self.robust and self.errors:
            prediction /= 1 + max(self.errors[-self.window:])
        return prediction

    def select(self, state, env):
        bw = self.predict_mbps(state)
        if bw is None:
            return C.DEFAULT_ACTION

        start = env.sim.chunk_idx
        h = min(self.horizon, env.sim.n_chunks - start)
        combos = self.combos if h == self.horizon else np.unique(self.combos[:, :h], axis=0)
        q = np.array([quality_value(a, env.qoe) for a in range(C.A_DIM)])

        buf = np.full(len(combos), buffer_s(state))
        total = np.zeros(len(combos))
        prev = np.full(len(combos), env.last_action)
        for k in range(h):
            acts = combos[:, k]
            size = env.sim.chunk_sizes[acts, start + k]
            delay = size * 8 / 1e6 / bw
            rebuf = np.maximum(delay - buf, 0.0)
            buf = np.maximum(buf - delay, 0.0) + C.CHUNK_LEN_S
            total += q[acts] - env.mu * rebuf - np.abs(q[acts] - q[prev]) - env.data_lambda * size / 1e6
            prev = acts
        return int(combos[np.argmax(total), 0])
