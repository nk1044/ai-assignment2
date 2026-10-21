import numpy as np
import pytest

from abrsim import ABREnv, Trace, synthetic_trace
from abrsim import config as C
from baselines import MPC, Bola, BufferBased, ConstantPolicy, RateBased, all_baselines
from evaluate import run_policy


def constant_trace(bw_mbps, duration_s=2000):
    return Trace(f"const_{bw_mbps}", np.arange(duration_s, dtype=float), np.full(duration_s, float(bw_mbps)))


def state_with_buffer(b):
    s = np.zeros((C.S_INFO, C.S_LEN), dtype=np.float32)
    s[1, -1] = b / C.BUFFER_NORM_S
    return s


@pytest.mark.parametrize("policy", all_baselines(), ids=lambda p: p.name)
def test_every_baseline_completes_valid_episodes(policy):
    env = ABREnv([synthetic_trace(seed=3), synthetic_trace(seed=4)], seed=0)
    rows, per_trace = run_policy(policy, env, 2)
    assert len(rows) == 2 * C.TOTAL_CHUNKS
    assert all(r["bitrate_kbps"] in C.BITRATES_KBPS for r in rows)


def test_buffer_based_mapping():
    bb = BufferBased(reservoir_s=5, cushion_s=10)
    assert bb.select(state_with_buffer(2), None) == 0
    assert bb.select(state_with_buffer(20), None) == C.A_DIM - 1
    assert 0 < bb.select(state_with_buffer(10), None) < C.A_DIM - 1


def test_bola_increases_quality_with_buffer():
    bola = Bola()
    picks = [bola.select(state_with_buffer(b), None) for b in (0, 10, 20, 40, 59)]
    assert picks == sorted(picks) and picks[0] < picks[-1]


@pytest.mark.parametrize("bw, expected", [(10.0, C.A_DIM - 1), (0.5, 0)])
def test_rate_based_tracks_link(bw, expected):
    env = ABREnv([constant_trace(bw)], noise=False, random_start=False)
    rows, _ = run_policy(RateBased(), env, 1)
    assert rows[-1]["bitrate_kbps"] == C.BITRATES_KBPS[expected]


def test_mpc_beats_always_highest_on_slow_link():
    env = ABREnv([constant_trace(1.5)], noise=False, random_start=False)
    _, mpc = run_policy(MPC(), env, 1)
    _, high = run_policy(ConstantPolicy(C.A_DIM - 1), env, 1)
    assert mpc[0]["qoe"] > high[0]["qoe"]
    assert mpc[0]["rebuffer_s"] < high[0]["rebuffer_s"]
