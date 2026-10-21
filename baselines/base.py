"""Common interface for ABR policies (baselines and, later, the RL agent).

A policy sees the same 6 x 8 observation as the RL agent plus read-only access to the env, which it
uses only for information a real player also has (the video manifest = chunk sizes, the last quality).
"""

import numpy as np

from abrsim import config as C


class Policy:
    name = "policy"

    def reset(self):
        """Called at the start of every video."""

    def select(self, state, env):
        raise NotImplementedError


def buffer_s(state):
    return float(state[1, -1] * C.BUFFER_NORM_S)


def past_throughputs(state):
    """Measured throughputs (Mbps) of downloaded chunks, oldest first (zeros = not yet downloaded)."""
    tp = state[2]
    return tp[tp > 0]


def harmonic_mean(x):
    return len(x) / np.sum(1.0 / np.asarray(x))


class ConstantPolicy(Policy):
    def __init__(self, action):
        self.action = action
        self.name = f"fixed_{C.BITRATES_KBPS[action]}"

    def select(self, state, env):
        return self.action


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def select(self, state, env):
        return int(self.rng.integers(C.A_DIM))
