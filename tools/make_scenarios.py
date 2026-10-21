"""Generate named stress-test network scenarios (README §12 U4) into ``data/traces/scenarios/``.

Each scenario is a recognisable real-life situation; ``--variants`` random versions of each are written
(deterministic seeds), 1 sample per second, same format as the real traces.

    python tools/make_scenarios.py
    python evaluate.py --traces scenarios --oracle
"""

import argparse
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "data" / "traces" / "scenarios"
DURATION_S = 300


def jitter(rng, base, rel=0.15):
    """Multiplicative log-normal noise around a base bandwidth curve."""
    return base * rng.lognormal(0.0, rel, size=len(base))


def metro_tunnel(rng):
    """Good 4G between stations, then 15–30 s of no signal in each tunnel."""
    bw = jitter(rng, np.full(DURATION_S, rng.uniform(2.5, 3.5)), 0.2)
    t = int(rng.uniform(25, 45))
    while t < DURATION_S:
        gap = int(rng.uniform(15, 30))
        bw[t:t + gap] = 0.0
        t += gap + int(rng.uniform(40, 70))
    return bw


def stadium_crowd(rng):
    """Thousands of phones join the cell: bandwidth decays from ~4 Mbps to ~0.4 Mbps."""
    start, end = rng.uniform(3.5, 4.5), rng.uniform(0.3, 0.5)
    base = start * (end / start) ** (np.arange(DURATION_S) / DURATION_S)
    return jitter(rng, base, 0.2)


def highway_handover(rng):
    """Fast car: decent bandwidth with a 2–4 s near-outage at every cell-tower handover."""
    bw = jitter(rng, np.full(DURATION_S, rng.uniform(1.8, 2.8)), 0.2)
    t = int(rng.uniform(10, 25))
    while t < DURATION_S:
        bw[t:t + int(rng.uniform(2, 5))] = rng.uniform(0.05, 0.2)
        t += int(rng.uniform(15, 30))
    return bw


def flaky_wifi(rng):
    """Crowded café Wi-Fi: bandwidth jumps between good and bad every few seconds."""
    bw, t, good = np.empty(DURATION_S), 0, True
    while t < DURATION_S:
        n = int(rng.uniform(2, 6))
        bw[t:t + n] = rng.uniform(3.0, 5.0) if good else rng.uniform(0.3, 0.7)
        good, t = not good, t + n
    return jitter(rng, bw, 0.1)


def rural_edge(rng):
    """Village on the edge of coverage: always low, slowly wandering bandwidth."""
    x, bw = np.log(0.8), np.empty(DURATION_S)
    for i in range(DURATION_S):
        x = np.clip(x + rng.normal(0, 0.08), np.log(0.35), np.log(1.3))
        bw[i] = np.exp(x)
    return bw


SCENARIOS = {f.__name__: f for f in (metro_tunnel, stadium_crowd, highway_handover, flaky_wifi, rural_edge)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants", type=int, default=6)
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.iterdir():
        f.unlink()
    for s_idx, (name, fn) in enumerate(SCENARIOS.items()):
        for v in range(args.variants):
            bw = np.maximum(fn(np.random.default_rng(1000 * s_idx + v)), 0.0)
            np.savetxt(out / f"{name}_{v}", np.column_stack([np.arange(DURATION_S, dtype=float), bw]),
                       fmt=["%.1f", "%.4f"], delimiter="\t")
        print(f"{name:18s} {args.variants} variants, mean {np.mean(bw):.2f} Mbps (last variant)")


if __name__ == "__main__":
    main()
