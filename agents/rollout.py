"""Collect complete videos from ``n_envs`` simulators in lock-step.

Besides the actor's observations it records, for every step, the privileged critic input: the true
bandwidth of the next 32 s at the moment of the decision (``ABREnv.privileged``).

Every video has the same number of chunks, so all envs finish together and a rollout is a clean
(T = TOTAL_CHUNKS, N = n_envs) batch.
"""

import numpy as np

from abrsim import ABREnv


class RolloutCollector:
    def __init__(self, traces, n_envs=16, qoe="lin", data_lambda=0.0, lambda_range=None, seed=0):
        """``lambda_range=(lo, hi)`` samples a data-saver weight per video and feeds it to the agent
        (preference-conditioned training, README §12 U2)."""
        self.envs = [ABREnv(traces, qoe=qoe, data_lambda=data_lambda, seed=seed + i) for i in range(n_envs)]
        self.lambda_range = lambda_range
        self.rng = np.random.default_rng(seed)

    def collect(self, agent):
        n = len(self.envs)
        lambdas = None
        if self.lambda_range is not None:
            lambdas = self.rng.uniform(*self.lambda_range, size=n).astype(np.float32)
            for env, lam in zip(self.envs, lambdas):
                env.data_lambda = float(lam)
        states = np.stack([env.reset() for env in self.envs])

        buf = {k: [] for k in ("states", "privileged", "actions", "rewards", "dones")}
        infos = []
        done = False
        while not done:
            extras = None if lambdas is None else lambdas[:, None]
            actions = agent.act(states, extras, rng=self.rng)
            privileged = np.stack([env.privileged() for env in self.envs])   # at decision time; critic only
            steps = [env.step(a) for env, a in zip(self.envs, actions)]
            buf["states"].append(states)
            buf["privileged"].append(privileged)
            buf["actions"].append(actions)
            buf["rewards"].append(np.array([s[1] for s in steps], dtype=np.float32))
            dones = np.array([s[2] for s in steps], dtype=np.float32)
            buf["dones"].append(dones)
            infos.append([s[3] for s in steps])
            states = np.stack([s[0] for s in steps])
            done = bool(dones.all())

        rollout = {k: np.stack(v) for k, v in buf.items()}
        if lambdas is not None:
            rollout["extras"] = np.broadcast_to(lambdas[None, :, None], (len(buf["states"]), n, 1)).copy()
        rollout["infos"] = infos
        return rollout
