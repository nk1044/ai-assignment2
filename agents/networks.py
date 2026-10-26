"""Actor and critic networks (README §7.2).

Both use the Pensieve feature extractor on the 6 x 8 state:
    rows 0, 1, 5 (last bitrate, buffer, chunks left)  -> Dense(128) on the latest value
    rows 2, 3    (throughput / download-time history) -> Conv1D(128, kernel 4) over 8 steps
    row 4        (next chunk sizes, 6 values)         -> Conv1D(128, kernel 4)
    concat -> Dense(128) -> head
``extra_dim`` appends conditioning scalars (e.g. the data-saver weight lambda, README §12 U2).

Asymmetric actor-critic: the critic additionally receives privileged information — the true mean
bandwidth over the next 8 x 4 s (``priv``), read from the trace by the simulator — through one more
Conv1D branch. The actor never sees it, so the deployed policy only uses information a real player has;
the critic is used only during training, where it gives sharper value estimates (lower-variance
advantages). The critic gets the normal observation *and* the privileged input, which keeps the policy
gradient unbiased under partial observability (Baisero & Amato, 2022).
"""

import torch
from torch import nn

from abrsim import config as C

HIDDEN = 128
KERNEL = 4


class FeatureExtractor(nn.Module):
    def __init__(self, extra_dim=0, priv_len=0):
        super().__init__()
        self.extra_dim = extra_dim
        self.priv_len = priv_len
        self.scalar = nn.ModuleList(nn.Linear(1, HIDDEN) for _ in range(3))
        self.conv_tp = nn.Conv1d(1, HIDDEN, KERNEL)
        self.conv_delay = nn.Conv1d(1, HIDDEN, KERNEL)
        self.conv_size = nn.Conv1d(1, HIDDEN, KERNEL)
        self.extra = nn.Linear(extra_dim, HIDDEN) if extra_dim else None
        self.conv_priv = nn.Conv1d(1, HIDDEN, KERNEL) if priv_len else None
        n_in = (3 * HIDDEN
                + 2 * HIDDEN * (C.S_LEN - KERNEL + 1)
                + HIDDEN * (C.A_DIM - KERNEL + 1)
                + (HIDDEN if extra_dim else 0)
                + (HIDDEN * (priv_len - KERNEL + 1) if priv_len else 0))
        self.merge = nn.Linear(n_in, HIDDEN)
        self.act = nn.ReLU()

    def forward(self, s, extra=None, priv=None):
        """s: (B, 6, 8); extra: (B, extra_dim) or None; priv: (B, priv_len) or None."""
        parts = [self.act(layer(s[:, row, -1:])) for layer, row in zip(self.scalar, (0, 1, 5))]
        parts.append(self.act(self.conv_tp(s[:, 2:3, :])).flatten(1))
        parts.append(self.act(self.conv_delay(s[:, 3:4, :])).flatten(1))
        parts.append(self.act(self.conv_size(s[:, 4:5, :C.A_DIM])).flatten(1))
        if self.extra is not None:
            parts.append(self.act(self.extra(extra)))
        if self.conv_priv is not None:
            parts.append(self.act(self.conv_priv(priv[:, None, :])).flatten(1))
        return self.act(self.merge(torch.cat(parts, dim=1)))


class Actor(nn.Module):
    def __init__(self, extra_dim=0):
        super().__init__()
        self.features = FeatureExtractor(extra_dim)
        self.head = nn.Linear(HIDDEN, C.A_DIM)

    def forward(self, s, extra=None):
        """Returns action logits (B, A_DIM); softmax gives pi(a|s)."""
        return self.head(self.features(s, extra))


class Critic(nn.Module):
    """V(s, future bandwidth): sees the actor's observation plus privileged look-ahead."""

    def __init__(self, extra_dim=0, priv_len=C.PRIV_WINDOWS):
        super().__init__()
        self.features = FeatureExtractor(extra_dim, priv_len)
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, s, extra=None, priv=None):
        """Returns V with shape (B,)."""
        return self.head(self.features(s, extra, priv)).squeeze(-1)
