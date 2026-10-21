"""Network bandwidth traces.

A trace file is plain text, one sample per line: ``<time_s> <bandwidth_Mbps>`` (the "cooked" format
used by Pensieve for the FCC and Norway HSDPA datasets).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

TRACE_ROOT = Path(__file__).resolve().parent.parent / "data" / "traces"
NON_TRACE_SUFFIXES = {".json", ".md"}      # metadata / notes living next to trace files


@dataclass
class Trace:
    name: str
    times: np.ndarray     # seconds, increasing
    bw_mbps: np.ndarray   # bandwidth at each timestamp

    def __len__(self):
        return len(self.times)


def load_trace(path):
    data = np.loadtxt(path, ndmin=2)
    return Trace(Path(path).name, data[:, 0].astype(float), data[:, 1].astype(float))


def load_traces(trace_dir):
    paths = sorted(p for p in Path(trace_dir).iterdir()
                   if p.is_file() and not p.name.startswith(".") and p.suffix not in NON_TRACE_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"no trace files in {trace_dir}")
    return [load_trace(p) for p in paths]


def load_dataset(spec):
    """Load traces by dataset name or path.

    ``"hsdpa/test"`` -> data/traces/hsdpa/test; ``"train"`` -> every ``*/train`` dir (FCC + HSDPA);
    ``"a+b"`` combines several specs; anything else is treated as a directory path.
    """
    traces = []
    for part in spec.split("+"):
        if (TRACE_ROOT / part).is_dir():
            traces += load_traces(TRACE_ROOT / part)
        elif part in ("train", "test"):
            dirs = sorted(TRACE_ROOT.glob(f"*/{part}"))
            if not dirs:
                raise FileNotFoundError(f"no */{part} dirs in {TRACE_ROOT}; run tools/prepare_traces.py")
            for d in dirs:
                traces += [Trace(f"{d.parent.name}/{t.name}", t.times, t.bw_mbps) for t in load_traces(d)]
        else:
            traces += load_traces(part)
    return traces


def synthetic_trace(duration_s=600, seed=0, low=0.2, high=6.0, volatility=0.25, name=None):
    """Random walk in log-bandwidth space, 1 s granularity. Used for tests until real traces are added."""
    rng = np.random.default_rng(seed)
    log_low, log_high = np.log(low), np.log(high)
    x = rng.uniform(log_low, log_high)
    bw = np.empty(duration_s)
    for i in range(duration_s):
        x = np.clip(x + rng.normal(0.0, volatility), log_low, log_high)
        bw[i] = np.exp(x)
    return Trace(name or f"synthetic_{seed}", np.arange(duration_s, dtype=float), bw)
