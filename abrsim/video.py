"""Video chunk-size table.

Real ABR work uses per-chunk sizes of an encoded test video (e.g. EnvivioDash3). Until we drop such a
table into ``data/video/``, we generate a synthetic VBR (variable bitrate) table: every chunk has a
"scene complexity" factor shared by all quality levels, so a complex scene is large at every bitrate.
"""

from pathlib import Path

import numpy as np

from .config import BITRATES_KBPS, CHUNK_LEN_S, TOTAL_CHUNKS


def synthetic_chunk_sizes(n_chunks=TOTAL_CHUNKS, seed=42, complexity_sigma=0.15, level_sigma=0.05):
    """Return an (A_DIM, n_chunks) array of chunk sizes in bytes."""
    rng = np.random.default_rng(seed)
    complexity = rng.lognormal(0.0, complexity_sigma, size=n_chunks)
    complexity /= complexity.mean()
    nominal = np.array(BITRATES_KBPS)[:, None] * 1000 * CHUNK_LEN_S / 8   # bytes
    level_noise = rng.lognormal(0.0, level_sigma, size=(len(BITRATES_KBPS), n_chunks))
    return (nominal * complexity[None, :] * level_noise).astype(np.int64)


def load_chunk_sizes(video_dir=None):
    """Load ``video_size_<level>`` files (one size in bytes per line) or fall back to synthetic sizes."""
    if video_dir is not None:
        files = [Path(video_dir) / f"video_size_{i}" for i in range(len(BITRATES_KBPS))]
        if all(f.exists() for f in files):
            return np.array([np.loadtxt(f, dtype=np.int64) for f in files])
    return synthetic_chunk_sizes()
