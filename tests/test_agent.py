"""Smoke tests for the PPO agent with an asymmetric critic (no real training here)."""

import numpy as np
import pytest
import torch

from abrsim import ABREnv, Trace, synthetic_trace
from abrsim import config as C
from agents import PPOAgent, PPOConfig, RLPolicy, RolloutCollector
from evaluate import run_policy

TRACES = [synthetic_trace(seed=i) for i in range(3)]


@pytest.mark.parametrize("extra_dim", [0, 1])
def test_network_shapes(extra_dim):
    agent = PPOAgent(PPOConfig(extra_dim=extra_dim))
    s = torch.zeros(5, C.S_INFO, C.S_LEN)
    e = torch.zeros(5, 1) if extra_dim else None
    p = torch.zeros(5, C.PRIV_WINDOWS)
    assert agent.actor(s, e).shape == (5, C.A_DIM)
    assert agent.critic(s, e, p).shape == (5,)


def test_actor_has_no_privileged_input_but_critic_does():
    agent = PPOAgent()
    assert agent.actor.features.conv_priv is None
    assert agent.critic.features.conv_priv is not None
    s = torch.rand(4, C.S_INFO, C.S_LEN)
    v1 = agent.critic(s, None, torch.zeros(4, C.PRIV_WINDOWS))
    v2 = agent.critic(s, None, torch.full((4, C.PRIV_WINDOWS), 5.0))
    assert not torch.allclose(v1, v2)                       # the critic actually uses the future


def test_future_bandwidth_reads_the_trace_without_moving_it():
    bw = np.array([1.0] * 10 + [3.0] * 10 + [0.0] * 20 + [2.0] * 100)
    env = ABREnv([Trace("t", np.arange(len(bw), dtype=float), bw)], noise=False, random_start=False)
    env.reset(0)
    before = env.sim.get_state()
    fut = env.privileged()
    assert env.sim.get_state() == before
    assert fut.shape == (C.PRIV_WINDOWS,)
    # bw[i] holds on (i-1, i]; 4 s windows from t=0: [0,4]=1, [4,8]=1, [8,12]=(1+3+3+3)/4, [12,16]=3
    assert fut[0] == pytest.approx(1.0) and fut[2] == pytest.approx(2.5) and fut[3] == pytest.approx(3.0)
    assert fut[-1] == pytest.approx(0.0)                     # inside the outage


def test_future_bandwidth_wraps_short_traces():
    env = ABREnv([Trace("c", np.arange(10, dtype=float), np.full(10, 2.0))], noise=False, random_start=False)
    env.reset(0)
    assert np.allclose(env.privileged(), 2.0)


def test_act_returns_valid_actions():
    agent = PPOAgent()
    states = np.random.default_rng(0).random((64, C.S_INFO, C.S_LEN)).astype(np.float32)
    a = agent.act(states, rng=np.random.default_rng(0))
    assert a.shape == (64,) and a.min() >= 0 and a.max() < C.A_DIM
    assert (agent.act(states, greedy=True) == agent.probs(states).argmax(1)).all()


def test_rollout_shapes_and_privileged_is_from_decision_time():
    collector = RolloutCollector(TRACES, n_envs=3)
    rollout = collector.collect(PPOAgent())
    assert rollout["states"].shape == (C.TOTAL_CHUNKS, 3, C.S_INFO, C.S_LEN)
    assert rollout["privileged"].shape == (C.TOTAL_CHUNKS, 3, C.PRIV_WINDOWS)
    assert rollout["actions"].shape == rollout["rewards"].shape == (C.TOTAL_CHUNKS, 3)
    assert rollout["dones"][-1].all() and not rollout["dones"][:-1].any()


def test_privileged_recorded_before_the_step():
    bw = np.concatenate([np.full(30, 4.0), np.zeros(30), np.full(400, 4.0)])
    collector = RolloutCollector([Trace("t", np.arange(len(bw), dtype=float), bw)], n_envs=1)
    env = collector.envs[0]
    env.sim.random_start = False
    rollout = collector.collect(PPOAgent())
    assert rollout["privileged"][0, 0, 0] == pytest.approx(4.0)  # at t=0 the next 4 s are at 4 Mbps


def test_gae_with_zero_critic_equals_discounted_return():
    agent = PPOAgent(PPOConfig(gamma=0.9, gae_lambda=1.0))
    for p in agent.critic.head.parameters():
        torch.nn.init.zeros_(p)
    rollout = {"rewards": np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
               "dones": np.array([[0.0], [0.0], [1.0]], dtype=np.float32),
               "states": np.zeros((3, 1, C.S_INFO, C.S_LEN), dtype=np.float32),
               "privileged": np.zeros((3, 1, C.PRIV_WINDOWS), dtype=np.float32)}
    adv, ret = agent.advantages(rollout)
    expected = [1 + 0.9 * 2 + 0.81 * 3, 2 + 0.9 * 3, 3]
    assert np.allclose(adv[:, 0], expected) and np.allclose(ret[:, 0], expected)


def test_ppo_update_changes_parameters_and_stays_near_old_policy():
    agent = PPOAgent(PPOConfig(ppo_epochs=3, target_kl=0))
    rollout = RolloutCollector(TRACES, n_envs=4).collect(agent)
    s = rollout["states"].reshape(-1, C.S_INFO, C.S_LEN)
    before = agent.probs(s)
    params = [p.clone() for p in agent.actor.parameters()]
    stats = agent.update(rollout)
    assert all(np.isfinite(v) for v in stats.values())
    assert stats["epochs"] == 3 and 0 <= stats["clip_frac"] <= 1
    assert any(not torch.equal(a, b) for a, b in zip(params, agent.actor.parameters()))
    after = agent.probs(s)
    kl = np.mean(np.sum(before * (np.log(before + 1e-12) - np.log(after + 1e-12)), axis=1))
    assert kl < 0.5                                           # clipping keeps the step small
    assert agent.iteration == 1


def test_target_kl_stops_epochs_early():
    agent = PPOAgent(PPOConfig(ppo_epochs=50, target_kl=1e-9, actor_lr=1e-2))
    stats = agent.update(RolloutCollector(TRACES, n_envs=2).collect(agent))
    assert stats["epochs"] < 50


def test_entropy_weight_decays_linearly():
    agent = PPOAgent(PPOConfig(entropy_start=1.0, entropy_end=0.0, entropy_decay_iters=10))
    agent.iteration = 5
    assert agent.entropy_weight() == pytest.approx(0.5)
    agent.iteration = 100
    assert agent.entropy_weight() == pytest.approx(0.0)


def test_checkpoint_roundtrip(tmp_path):
    agent = PPOAgent(PPOConfig(extra_dim=1))
    agent.update(RolloutCollector(TRACES, n_envs=2, lambda_range=(0, 1)).collect(agent))
    agent.save(tmp_path / "a.pt", val_qoe=1.23)
    loaded = PPOAgent.load(tmp_path / "a.pt")
    s = np.random.default_rng(1).random((4, C.S_INFO, C.S_LEN)).astype(np.float32)
    e = np.full((4, 1), 0.5, dtype=np.float32)
    assert np.allclose(agent.probs(s, e), loaded.probs(s, e))
    assert loaded.iteration == 1 and loaded.meta["val_qoe"] == 1.23


def test_rl_policy_plugs_into_evaluation():
    env = ABREnv(TRACES, seed=0)
    rows, per_trace = run_policy(RLPolicy(PPOAgent()), env, 2)
    assert len(rows) == 2 * C.TOTAL_CHUNKS and len(per_trace) == 2


def test_train_script_end_to_end(tmp_path):
    import train
    run_dir = train.main(["--train-set", "hsdpa/train", "--iters", "2", "--n-envs", "2", "--val-repeats", "1",
                          "--eval-every", "2", "--ckpt-every", "1", "--out", str(tmp_path), "--run-name", "smoke"])
    for f in ("config.json", "log.csv", "best.pt", "final.pt", "ckpt_000001.pt", "ckpt_000002.pt"):
        assert (run_dir / f).exists(), f
