"""Wrap a trained agent in the baseline ``Policy`` interface so ``evaluate.py`` can compare them."""

from baselines.base import Policy

from .ppo import PPOAgent


class RLPolicy(Policy):
    def __init__(self, agent, name="PPO", data_lambda=None, greedy=True):
        self.agent = agent
        self.name = name
        self.data_lambda = data_lambda
        self.greedy = greedy

    @classmethod
    def from_checkpoint(cls, path, name="PPO", **kw):
        return cls(PPOAgent.load(path), name, **kw)

    def select(self, state, env):
        extras = None
        if self.agent.cfg.extra_dim:
            lam = env.data_lambda if self.data_lambda is None else self.data_lambda
            extras = [[lam]]
        return int(self.agent.act(state[None], extras, greedy=self.greedy)[0])
