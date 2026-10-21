"""Offline optimal (upper bound): dynamic programming with full knowledge of the future bandwidth.

Not achievable by a real player — it tells us how much QoE is left on the table.

DP state after chunk k = (last quality, buffer level rounded to ``buffer_bin`` seconds). For each state we
keep only the highest-reward history reaching it (plus its exact simulator state). Rebuffering is already
paid for inside the reward, so two histories that end in the same quality and buffer level are
interchangeable up to the bin width — this makes the search near-optimal while staying tractable. With
``max_states`` the frontier is additionally capped to the best states by reward.

Uses a noise-free copy of the simulator, so evaluate it in a noise-free env (``evaluate.py`` does).
"""

import numpy as np

from abrsim import StreamingSimulator
from abrsim import config as C
from abrsim.abr_env import quality_value

from .base import Policy


class OfflineOptimal(Policy):
    name = "Oracle"

    def __init__(self, buffer_bin=0.5, max_states=None):
        self.buffer_bin = buffer_bin
        self.max_states = max_states

    def reset(self):
        self.plan = None

    def select(self, state, env):
        if self.plan is None:
            self.plan = self.solve(env)
        return self.plan[env.sim.chunk_idx]

    def solve(self, env):
        """Best action for every remaining chunk, from the env's current position."""
        sim = StreamingSimulator([env.sim.trace], env.sim.chunk_sizes, noise=False)
        sim.trace = env.sim.trace
        q = np.array([quality_value(a, env.qoe) for a in range(C.A_DIM)])
        start = env.sim.chunk_idx
        plan = [0] * env.sim.n_chunks

        # frontier: key -> (total reward, simulator state, actions so far)
        frontier = {(env.last_action, -1): (0.0, env.sim.get_state(), [])}
        for _ in range(start, env.sim.n_chunks):
            nxt = {}
            for (prev, _), (total, sim_state, path) in frontier.items():
                for a in range(C.A_DIM):
                    sim.set_state(sim_state)
                    out = sim.download_chunk(a)
                    r = (q[a] - env.mu * out["rebuffer_s"] - abs(q[a] - q[prev])
                         - env.data_lambda * out["chunk_size"] / 1e6)
                    key = (a, int(round(out["buffer_s"] / self.buffer_bin)))
                    if key not in nxt or total + r > nxt[key][0]:
                        nxt[key] = (total + r, sim.get_state(), path + [a])
            if self.max_states and len(nxt) > self.max_states:
                nxt = dict(sorted(nxt.items(), key=lambda kv: -kv[1][0])[:self.max_states])
            frontier = nxt

        best = max(frontier.values(), key=lambda v: v[0])[2]
        plan[start:] = best
        return plan
