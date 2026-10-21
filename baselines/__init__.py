from .base import ConstantPolicy, Policy, RandomPolicy
from .heuristics import Bola, BufferBased, RateBased
from .mpc import MPC


def all_baselines():
    return [
        RandomPolicy(),
        ConstantPolicy(0),
        RateBased(),
        BufferBased(),
        Bola(),
        MPC(robust=False),
        MPC(robust=True),
    ]
