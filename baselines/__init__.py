from .base import ConstantPolicy, Policy, RandomPolicy
from .heuristics import Bola, BufferBased, RateBased
from .mpc import MPC
from .oracle import OfflineOptimal


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
