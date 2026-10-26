"""Shared figure style: one fixed colour per algorithm across every figure (colour follows the
entity, never its rank), recessive grid/axes, thin marks, text in ink colours."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical palette (light mode), assigned in fixed slot order.
POLICY_COLORS = {
    "PPO": "#2a78d6",        # slot 1 blue   — our agent (PPO, asymmetric critic)
    "RobustMPC": "#eb6834",  # slot 2 orange
    "BOLA": "#1baf7a",       # slot 3 aqua
    "RB": "#eda100",         # slot 4 yellow
    "BB": "#e87ba4",         # slot 5 magenta
    "FastMPC": "#008300",    # slot 6 green
    "fixed_300": "#4a3aa7",  # slot 7 violet
    "A2C": "#e34948",        # slot 8 red    — earlier symmetric A2C agent (comparison only)
    "random": "#e34948",     # never plotted (off-scale); tables only
    "Oracle": "#8a8984",     # neutral reference, drawn dashed
}
POLICY_LABELS = {"fixed_300": "Fixed 300k", "random": "Random", "Oracle": "Offline optimal"}
# Order used in every legend / bar chart.
POLICY_ORDER = ["Oracle", "PPO", "A2C", "RobustMPC", "BOLA", "RB", "BB", "FastMPC", "fixed_300", "random"]

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def color(policy):
    return POLICY_COLORS.get(policy, INK_2)


def label(policy):
    return POLICY_LABELS.get(policy, policy)


def ordered(policies):
    known = [p for p in POLICY_ORDER if p in policies]
    return known + sorted(p for p in policies if p not in POLICY_ORDER)


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "font.size": 9,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })
