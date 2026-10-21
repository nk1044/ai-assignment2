"""Classic hand-designed ABR algorithms (README §8)."""

import numpy as np

from abrsim import config as C

from .base import Policy, buffer_s, harmonic_mean, past_throughputs


class RateBased(Policy):
    """Predict bandwidth as the harmonic mean of the last ``window`` throughputs; pick the highest
    bitrate not exceeding it."""

    name = "RB"

    def __init__(self, window=5):
        self.window = window

    def select(self, state, env):
        tp = past_throughputs(state)[-self.window:]
        if len(tp) == 0:
            return C.DEFAULT_ACTION
        predicted_kbps = harmonic_mean(tp) * 1000
        fitting = [a for a, r in enumerate(C.BITRATES_KBPS) if r <= predicted_kbps]
        return fitting[-1] if fitting else 0


class BufferBased(Policy):
    """BBA-0 (Huang et al., SIGCOMM'14): lowest quality below the reservoir, highest above
    reservoir + cushion, linear in between."""

    name = "BB"

    def __init__(self, reservoir_s=5.0, cushion_s=10.0):
        self.reservoir_s = reservoir_s
        self.cushion_s = cushion_s

    def select(self, state, env):
        b = buffer_s(state)
        if b < self.reservoir_s:
            return 0
        if b >= self.reservoir_s + self.cushion_s:
            return C.A_DIM - 1
        return int((C.A_DIM - 1) * (b - self.reservoir_s) / self.cushion_s)


class Bola(Policy):
    """BOLA-BASIC (Spiteri et al., INFOCOM'16): Lyapunov-based, maximizes
    (V * (v_m + gp) - Q) / S_m with utility v_m = ln(S_m / S_min) and Q = buffer in chunks."""

    name = "BOLA"

    def __init__(self, gp=5.0, buffer_target_s=C.BUFFER_MAX_S):
        self.gp = gp
        self.q_max = buffer_target_s / C.CHUNK_LEN_S
        sizes = np.array(C.BITRATES_KBPS, dtype=float)
        self.utility = np.log(sizes / sizes[0])
        self.V = (self.q_max - 1) / (self.utility[-1] + gp)

    def select(self, state, env):
        q = buffer_s(state) / C.CHUNK_LEN_S
        sizes = np.array(C.BITRATES_KBPS, dtype=float)
        score = (self.V * (self.utility + self.gp) - q) / sizes
        return int(np.argmax(score))
