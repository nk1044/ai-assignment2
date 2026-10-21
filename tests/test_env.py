import numpy as np
import pytest

from abrsim import ABREnv, Trace, synthetic_trace
from abrsim import config as C


def constant_trace(bw_mbps, duration_s=2000):
    return Trace(f"const_{bw_mbps}", np.arange(duration_s, dtype=float), np.full(duration_s, float(bw_mbps)))


def run_episode(env, policy, trace_idx=0):
    s = env.reset(trace_idx)
    total, infos, done = 0.0, [], False
    while not done:
        s, r, done, info = env.step(policy(s))
        total += r
        infos.append(info)
    return total, infos


def test_state_shape_and_episode_length():
    env = ABREnv([synthetic_trace(seed=1)], seed=0)
    s = env.reset()
    assert s.shape == (C.S_INFO, C.S_LEN)
    _, infos = run_episode(env, lambda s: 2)
    assert len(infos) == C.TOTAL_CHUNKS
    assert infos[-1]["done"] and infos[-1]["chunks_remaining"] == 0


def test_download_time_matches_bandwidth():
    # 4 Mbps, no noise: chunk of S bytes takes S*8/(4e6*0.95) s + RTT
    env = ABREnv([constant_trace(4.0)], noise=False, random_start=False)
    env.reset(0)
    _, _, _, info = env.step(0)
    expected = info["chunk_size"] * 8 / (4e6 * C.PACKET_PAYLOAD_PORTION) + C.LINK_RTT_S
    assert info["delay_s"] == pytest.approx(expected, rel=1e-9)


def test_first_chunk_always_rebuffers_and_buffer_updates():
    env = ABREnv([constant_trace(4.0)], noise=False, random_start=False)
    env.reset(0)
    _, _, _, info = env.step(0)
    assert info["rebuffer_s"] == pytest.approx(info["delay_s"])       # buffer starts empty
    assert info["buffer_s"] == pytest.approx(C.CHUNK_LEN_S)


def test_high_bitrate_on_slow_link_stalls_more():
    env = ABREnv([constant_trace(1.0)], noise=False, random_start=False)
    _, low = run_episode(env, lambda s: 0)
    _, high = run_episode(env, lambda s: 5)
    assert sum(i["rebuffer_s"] for i in high) > sum(i["rebuffer_s"] for i in low)


def test_buffer_never_exceeds_cap_on_fast_link():
    env = ABREnv([constant_trace(50.0)], noise=False, random_start=False)
    _, infos = run_episode(env, lambda s: 0)
    assert max(i["buffer_s"] for i in infos) <= C.BUFFER_MAX_S + 1e-9
    assert any(i["sleep_s"] > 0 for i in infos)


def test_reward_decomposition():
    env = ABREnv([constant_trace(2.0)], noise=False, random_start=False, data_lambda=0.5)
    env.reset(0)
    _, r, _, info = env.step(4)
    q, q_prev = 2.85, C.BITRATES_KBPS[C.DEFAULT_ACTION] / 1000
    expected = q - 4.3 * info["rebuffer_s"] - abs(q - q_prev) - 0.5 * info["chunk_size"] / 1e6
    assert r == pytest.approx(expected)


def test_zero_bandwidth_gap_is_survived():
    # 10 s outage (metro tunnel) in the middle of a 3 Mbps link
    bw = np.full(300, 3.0)
    bw[20:30] = 0.0
    env = ABREnv([Trace("tunnel", np.arange(300, dtype=float), bw)], noise=False, random_start=False)
    _, infos = run_episode(env, lambda s: 1)
    assert len(infos) == C.TOTAL_CHUNKS


def test_trace_wraps_around_when_video_is_longer():
    env = ABREnv([constant_trace(0.5, duration_s=30)], noise=False, random_start=False)
    _, infos = run_episode(env, lambda s: 0)
    assert len(infos) == C.TOTAL_CHUNKS
