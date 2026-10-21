"""Smoke tests for the A2C agent (no real training here)."""

import numpy as np
import pytest
import torch

from abrsim import ABREnv, synthetic_trace
from abrsim import config as C
from agents import A2CAgent, A2CConfig, RLPolicy, RolloutCollector
from evaluate import run_policy

TRACES = [synthetic_trace(seed=i) for i in range(3)]


@pytest.mark.parametrize("extra_dim", [0, 1])
def test_network_shapes(extra_dim):
    agent = A2CAgent(A2CConfig(extra_dim=extra_dim))
    s = torch.zeros(5, C.S_INFO, C.S_LEN)
    e = torch.zeros(5, 1) if extra_dim else None
    assert agent.actor(s, e).shape == (5, C.A_DIM)
    assert agent.critic(s, e).shape == (5,)


def test_act_returns_valid_actions():
    agent = A2CAgent()
    states = np.random.default_rng(0).random((64, C.S_INFO, C.S_LEN)).astype(np.float32)
    a = agent.act(states, rng=np.random.default_rng(0))
    assert a.shape == (64,) and a.min() >= 0 and a.max() < C.A_DIM
    assert (agent.act(states, greedy=True) == agent.probs(states).argmax(1)).all()


def test_rollout_shapes():
    rollout = RolloutCollector(TRACES, n_envs=3).collect(A2CAgent())
    assert rollout["states"].shape == (C.TOTAL_CHUNKS, 3, C.S_INFO, C.S_LEN)
    assert rollout["actions"].shape == rollout["rewards"].shape == (C.TOTAL_CHUNKS, 3)
    assert rollout["dones"][-1].all() and not rollout["dones"][:-1].any()


def test_gae_with_zero_critic_equals_discounted_return():
    agent = A2CAgent(A2CConfig(gamma=0.9, gae_lambda=1.0))
    for p in agent.critic.head.parameters():
        torch.nn.init.zeros_(p)
    rewards = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    dones = np.array([[0.0], [0.0], [1.0]], dtype=np.float32)
    states = np.zeros((3, 1, C.S_INFO, C.S_LEN), dtype=np.float32)
    adv, ret = agent.advantages(states, None, rewards, dones)
    expected = [1 + 0.9 * 2 + 0.81 * 3, 2 + 0.9 * 3, 3]
    assert np.allclose(adv[:, 0], expected) and np.allclose(ret[:, 0], expected)


def test_update_changes_parameters_and_is_finite():
    agent = A2CAgent()
    before = [p.clone() for p in agent.actor.parameters()]
    stats = agent.update(RolloutCollector(TRACES, n_envs=2).collect(agent))
    assert all(np.isfinite(v) for v in stats.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.actor.parameters()))
    assert agent.iteration == 1


def test_entropy_weight_decays_linearly():
    agent = A2CAgent(A2CConfig(entropy_start=1.0, entropy_end=0.0, entropy_decay_iters=10))
    agent.iteration = 5
    assert agent.entropy_weight() == pytest.approx(0.5)
    agent.iteration = 100
    assert agent.entropy_weight() == pytest.approx(0.0)


def test_checkpoint_roundtrip(tmp_path):
    agent = A2CAgent(A2CConfig(extra_dim=1))
    agent.update(RolloutCollector(TRACES, n_envs=2, lambda_range=(0, 1)).collect(agent))
    agent.save(tmp_path / "a.pt", val_qoe=1.23)
    loaded = A2CAgent.load(tmp_path / "a.pt")
    s = np.random.default_rng(1).random((4, C.S_INFO, C.S_LEN)).astype(np.float32)
    e = np.full((4, 1), 0.5, dtype=np.float32)
    assert np.allclose(agent.probs(s, e), loaded.probs(s, e))
    assert loaded.iteration == 1 and loaded.meta["val_qoe"] == 1.23


def test_rl_policy_plugs_into_evaluation():
    env = ABREnv(TRACES, seed=0)
    rows, per_trace = run_policy(RLPolicy(A2CAgent()), env, 2)
    assert len(rows) == 2 * C.TOTAL_CHUNKS and len(per_trace) == 2


def test_train_script_end_to_end(tmp_path):
    import train
    run_dir = train.main(["--train-set", "hsdpa/train", "--iters", "2", "--n-envs", "2",
                          "--eval-every", "2", "--ckpt-every", "1", "--out", str(tmp_path), "--run-name", "smoke"])
    for f in ("config.json", "log.csv", "best.pt", "final.pt", "ckpt_000001.pt", "ckpt_000002.pt"):
        assert (run_dir / f).exists(), f
