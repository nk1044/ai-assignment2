"""Download the real network traces + EnvivioDash3 chunk sizes and lay them out under ``data/``.

Source: the "cooked" traces distributed with Pensieve, as mirrored in github.com/godka/Pensieve-PPO
(pinned commit below). Format: ``<time_s> <bandwidth_Mbps>`` per line.

    data/traces/fcc/{train,test}     FCC Measuring Broadband America (80/20 split done here)
    data/traces/hsdpa/{train,test}   Norway HSDPA 3G commute traces (split as provided)
    data/video/envivio/video_size_*  per-chunk sizes (bytes) of the EnvivioDash3 test video

    python tools/prepare_traces.py
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

REPO = "https://github.com/godka/Pensieve-PPO.git"
COMMIT = "a1b2579ca325625a23fe7d329a186ef09e32a3f1"
FCC_TEST_FRACTION = 0.2
SPLIT_SEED = 0

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def fetch(dest):
    run = lambda *cmd: subprocess.run(cmd, cwd=dest, check=True, capture_output=True)
    run("git", "init", "-q")
    run("git", "remote", "add", "origin", REPO)
    run("git", "sparse-checkout", "set", "src/train", "src/test", "src/envivio")
    run("git", "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", COMMIT)
    run("git", "checkout", "-q", "FETCH_HEAD")
    return Path(dest) / "src"


def copy(files, dest, rename=lambda f: f.name):
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy(f, dest / rename(f))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        src = fetch(tmp)
        train = sorted(p for p in (src / "train").iterdir() if p.is_file())
        fcc = [p for p in train if p.name.startswith("trace_")]
        hsdpa_train = [p for p in train if not p.name.startswith("trace_")]
        hsdpa_test = sorted(p for p in (src / "test").iterdir() if p.is_file())

        rng = np.random.default_rng(SPLIT_SEED)
        order = rng.permutation(len(fcc))
        n_test = round(len(fcc) * FCC_TEST_FRACTION)
        fcc_test = [fcc[i] for i in sorted(order[:n_test])]
        fcc_train = [fcc[i] for i in sorted(order[n_test:])]

        shutil.rmtree(DATA / "traces", ignore_errors=True)
        copy(fcc_train, DATA / "traces/fcc/train")
        copy(fcc_test, DATA / "traces/fcc/test")
        # strip the long Oslo "report" names: bus.ljansbakken-oslo-report.2010-...log -> bus_ljansbakken_2010-...
        short = lambda f: f.name.replace("-oslo-report.", "_").replace(".log", "").replace(".", "_", 1)
        copy(hsdpa_train, DATA / "traces/hsdpa/train", short)
        copy(hsdpa_test, DATA / "traces/hsdpa/test")
        copy(sorted((src / "envivio").glob("video_size_*")), DATA / "video/envivio")

    for d in sorted((DATA / "traces").glob("*/*")):
        print(f"{d.relative_to(DATA)}: {len(list(d.iterdir()))} traces")


if __name__ == "__main__":
    main()
