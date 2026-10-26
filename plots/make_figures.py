"""Generate the report figures (README §10) from evaluation logs.

    python evaluate.py --oracle && python evaluate.py --oracle --traces fcc/test   # produce the logs
    python plots/make_figures.py                                                    # -> plots/figures/

Figures whose inputs are missing (e.g. the training curve before training) are skipped with a note.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import style  # noqa: E402
from style import INK, INK_2, color, label, ordered, plt  # noqa: E402

from abrsim import config as C  # noqa: E402
from abrsim import load_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HIDDEN = {"random"}               # off-scale sanity baseline: tables only


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def load_results(result_dir):
    """{policy: {"chunks": [...], "traces": [...]}} for every policy logged in ``result_dir``."""
    out = {}
    for f in sorted(Path(result_dir).glob("*_traces.csv")):
        name = f.name[: -len("_traces.csv")]
        if name in HIDDEN:
            continue
        out[name] = {"traces": read_csv(f), "chunks": read_csv(f.with_name(f"{name}.csv"))}
    return out


def save(fig, out_dir, name):
    fig.savefig(out_dir / f"{name}.png")
    fig.savefig(out_dir / f"{name}.pdf", metadata={"CreationDate": None})
    plt.close(fig)
    print(f"  wrote {name}.png/.pdf")


# --------------------------------------------------------------------------- G2
def fig_qoe_bars(results_by_set, out_dir):
    sets = list(results_by_set)
    fig, axes = plt.subplots(1, len(sets), figsize=(3.4 * len(sets), 3.0), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, sets):
        res = results_by_set[name]
        pols = ordered(res)
        means = [np.mean([t["qoe"] for t in res[p]["traces"]]) for p in pols]
        sems = [np.std([t["qoe"] for t in res[p]["traces"]]) / np.sqrt(len(res[p]["traces"])) for p in pols]
        y = np.arange(len(pols))[::-1]
        for yi, p, m, e in zip(y, pols, means, sems):
            hatch = "////" if p == "Oracle" else None
            ax.barh(yi, m, height=0.62, color="none" if p == "Oracle" else color(p),
                    edgecolor=color(p), hatch=hatch, linewidth=1.0)
            ax.errorbar(m, yi, xerr=e, color=INK_2, capsize=2, linewidth=0.8)
            ax.text(1.02, yi, f"{m:.2f}", va="center", fontsize=7.5, color=INK,
                    transform=ax.get_yaxis_transform())
        ax.set_yticks(y, [label(p) for p in pols])
        ax.set_xlabel("Mean QoE per chunk (± s.e.)")
        ax.set_title(f"{name} ({len(res[pols[0]]['traces'])} traces)")
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color=INK_2, linewidth=0.6)
        lo = min(0, min(m - e for m, e in zip(means, sems)))
        ax.set_xlim(lo - 0.05, max(m + e for m, e in zip(means, sems)) + 0.05)
    fig.suptitle("Average QoE (QoE_lin, higher is better)", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, out_dir, "g2_qoe_bars")


# --------------------------------------------------------------------------- G3
def fig_breakdown(res, set_name, out_dir):
    pols = ordered(res)
    comp = {k: [] for k in ("quality", "rebuffer_penalty", "smooth_penalty")}
    for p in pols:
        rows = [r for r in res[p]["chunks"] if r["chunk"] > 0]
        for k in comp:
            comp[k].append(np.mean([r[k] for r in rows]))
    x = np.arange(len(pols))
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(x, comp["quality"], width=0.62, color=style.SEQ_BLUE[3], label="Bitrate utility  q(R)")
    ax.bar(x, -np.array(comp["rebuffer_penalty"]), width=0.62, color="#e34948", label="Rebuffer penalty  μ·T")
    ax.bar(x, -np.array(comp["smooth_penalty"]), width=0.62, bottom=-np.array(comp["rebuffer_penalty"]),
           color="#eda100", edgecolor=style.SURFACE, linewidth=1.0, label="Smoothness penalty  |Δq|")
    net = np.array(comp["quality"]) - comp["rebuffer_penalty"] - np.array(comp["smooth_penalty"])
    ax.scatter(x, net, marker="D", s=26, color=INK, zorder=3, label="Net QoE")
    ax.axhline(0, color=INK_2, linewidth=0.6)
    ax.set_xticks(x, [label(p) for p in pols], rotation=20, ha="right")
    ax.set_ylabel("Per-chunk contribution")
    ax.set_title(f"Where the QoE comes from — {set_name}")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=2, loc="upper right")
    ax.set_ylim(min(-np.array(comp["rebuffer_penalty"]) - comp["smooth_penalty"]) * 1.25,
                max(comp["quality"]) * 1.45)
    save(fig, out_dir, f"g3_breakdown_{set_name.replace('/', '_')}")


# --------------------------------------------------------------------------- G4
def fig_cdf(results_by_set, out_dir):
    sets = list(results_by_set)
    fig, axes = plt.subplots(1, len(sets), figsize=(3.4 * len(sets), 3.0))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, sets):
        res = results_by_set[name]
        for p in ordered(res):
            q = np.sort([t["qoe"] for t in res[p]["traces"]])
            ax.step(q, np.arange(1, len(q) + 1) / len(q), where="post", color=color(p),
                    linestyle="--" if p == "Oracle" else "-", label=label(p))
        ax.set_xlabel("Per-trace mean QoE")
        ax.set_ylabel("CDF")
        ax.set_title(name)
        lo = np.percentile(np.concatenate([[t["qoe"] for t in res[p]["traces"]] for p in res]), 2)
        ax.set_xlim(left=lo)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Distribution of QoE across traces (further right is better)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, out_dir, "g4_qoe_cdf")


# --------------------------------------------------------------------------- G6
def fig_tradeoff(res, set_name, out_dir):
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    pts = {p: (100 * np.mean([r["rebuffer_ratio"] for r in res[p]["traces"]]),
               np.mean([r["bitrate_mbps"] for r in res[p]["traces"]])) for p in ordered(res)}
    xs, ys = np.array(list(pts.values())).T
    lo, span = np.array([xs.min(), ys.min()]), np.array([np.ptp(xs) or 1, np.ptp(ys) or 1])
    norm = {p: (np.array(v) - lo) / span for p, v in pts.items()}
    taken = list(norm.values())                     # points + already-placed label centres
    for p, (x, y) in pts.items():
        ax.scatter(x, y, s=70, color="none" if p == "Oracle" else color(p), edgecolor=color(p),
                   linewidth=1.6, zorder=3)
        w = 0.018 * len(label(p))
        best = None
        for dx, dy, ha in ((8, -10, "left"), (8, 6, "left"), (-8, 6, "right"), (-8, -10, "right")):
            centre = norm[p] + np.array([np.sign(dx) * (0.03 + w / 2), 0.05 * np.sign(dy)])
            clearance = min(np.hypot(*((centre - q) * [0.6, 1.0])) for q in taken if q is not norm[p])
            if best is None or clearance > best[0] + 1e-9:
                best = (clearance, dx, dy, ha, centre)
        _, dx, dy, ha, centre = best
        ax.annotate(label(p), (x, y), textcoords="offset points", xytext=(dx, dy), ha=ha,
                    fontsize=8, color=INK)
        taken.append(centre)
    ax.margins(x=0.18, y=0.12)
    ax.set_xlabel("Time spent rebuffering (% of playback)")
    ax.set_ylabel("Average bitrate (Mbps)")
    ax.set_title(f"Quality vs. stalls — {set_name}")
    ax.text(0.98, 0.04, "← fewer stalls,  ↑ higher quality", transform=ax.transAxes, ha="right",
            fontsize=7.5, color=INK_2)
    save(fig, out_dir, f"g6_tradeoff_{set_name.replace('/', '_')}")


# --------------------------------------------------------------------------- G5
def fig_timeseries(res, set_name, out_dir, trace_name=None, policies=("BB", "RobustMPC", "PPO", "Oracle")):
    pols = [p for p in ordered(res) if p in policies]
    by_trace = defaultdict(dict)
    for p in pols:
        for r in res[p]["chunks"]:
            by_trace[r["trace"]].setdefault(p, []).append(r)
    if trace_name is None:   # most interesting trace: largest QoE spread between the plotted policies
        spread = {t: np.ptp([np.mean([r["reward"] for r in rows]) for rows in d.values()])
                  for t, d in by_trace.items() if len(d) == len(pols)}
        trace_name = max(spread, key=spread.get)
    trace = next(t for t in load_dataset(set_name) if t.name == trace_name)

    fig, axes = plt.subplots(3, 1, figsize=(6.6, 5.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.2, 1]})
    end = max(rows[-1]["wall_time_s"] for rows in by_trace[trace_name].values())
    m = trace.times - trace.times[0] <= end + 5
    axes[0].fill_between(trace.times[m] - trace.times[0], trace.bw_mbps[m], step="post",
                         color=style.SEQ_BLUE[1], linewidth=0)
    axes[0].plot(trace.times[m] - trace.times[0], trace.bw_mbps[m], drawstyle="steps-post",
                 color=style.SEQ_BLUE[4], linewidth=1.0)
    axes[0].set_ylabel("Bandwidth\n(Mbps)")
    axes[0].set_title(f"One video, {len(pols)} players — trace {trace_name}")
    for i, p in enumerate(pols):
        rows = by_trace[trace_name][p]
        start = np.array([r["wall_time_s"] - r["delay_s"] - r["sleep_s"] for r in rows])
        ls = "--" if p == "Oracle" else "-"
        axes[1].step(start, [r["bitrate_kbps"] / 1000 for r in rows], where="post", color=color(p),
                     linestyle=ls, label=label(p))
        axes[2].plot([r["wall_time_s"] for r in rows], [r["buffer_s"] for r in rows], color=color(p),
                     linestyle=ls, label=label(p))
        stalls = [r for r in rows if r["rebuffer_s"] > 0.05 and r["chunk"] > 0]
        axes[2].scatter([r["wall_time_s"] for r in stalls], [-3.5 * (i + 1)] * len(stalls), marker="v", s=34,
                        color=color(p), edgecolor=style.SURFACE, linewidth=1.0, zorder=4)
    axes[1].set_ylabel("Chosen bitrate\n(Mbps)")
    axes[1].set_yticks([b / 1000 for b in C.BITRATES_KBPS])
    axes[1].legend(ncol=len(pols), loc="upper left", bbox_to_anchor=(0, 1.22))
    axes[2].set_ylabel("Buffer (s)")
    axes[2].set_xlabel("Wall-clock time (s)        ▼ = stall")
    axes[2].set_ylim(bottom=-3.5 * (len(pols) + 1))
    axes[2].set_yticks([t for t in axes[2].get_yticks() if t >= 0])
    axes[2].axhline(0, color=INK_2, linewidth=0.6)
    fig.tight_layout()
    save(fig, out_dir, f"g5_timeseries_{set_name.replace('/', '_')}")


# --------------------------------------------------------------------------- data overview
def fig_traces(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, (name, spec) in zip(axes, [("Norway HSDPA 3G", "hsdpa/test"), ("FCC broadband", "fcc/test")]):
        traces = load_dataset(spec)
        rng = np.random.default_rng(1)
        for i, t in enumerate(rng.choice(len(traces), 3, replace=False)):
            tr = traces[t]
            m = tr.times - tr.times[0] <= 200
            ax.plot(tr.times[m] - tr.times[0], tr.bw_mbps[m], drawstyle="steps-post",
                    color=["#2a78d6", "#eb6834", "#1baf7a"][i], linewidth=1.1)
        for b in C.BITRATES_KBPS:
            ax.axhline(b / 1000, color=INK_2, linewidth=0.5, linestyle=":")
        ax.set_title(name)
        ax.set_xlabel("Time (s)")
    axes[0].set_ylabel("Bandwidth (Mbps)")
    fig.suptitle("Real network traces fluctuate around the bitrate ladder", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.text(0.01, 0.875, "3 random test traces per dataset · dotted lines = the 6 available bitrates",
             fontsize=8, color=INK_2)
    fig.tight_layout()
    save(fig, out_dir, "data_traces")


# --------------------------------------------------------------------------- scenarios (U4)
SCENARIO_TITLES = {"metro_tunnel": "Metro tunnel", "stadium_crowd": "Stadium crowd",
                   "highway_handover": "Highway handover", "flaky_wifi": "Flaky Wi-Fi", "rural_edge": "Rural edge"}


def fig_scenarios(res, out_dir, policies=("Oracle", "PPO", "RobustMPC", "BOLA", "RB", "BB")):
    pols = [p for p in ordered(res) if p in policies]
    scen = list(SCENARIO_TITLES)
    traces = {t.name: t for t in load_dataset("scenarios")}
    fig, axes = plt.subplots(2, len(scen), figsize=(10.5, 4.4), gridspec_kw={"height_ratios": [1, 2.3]})
    qoe_all = []
    for j, sc in enumerate(scen):
        tr = traces[f"{sc}_0"]
        ax = axes[0, j]
        ax.fill_between(tr.times, tr.bw_mbps, step="post", color=style.SEQ_BLUE[1], linewidth=0)
        ax.plot(tr.times, tr.bw_mbps, drawstyle="steps-post", color=style.SEQ_BLUE[4], linewidth=0.8)
        ax.set_title(SCENARIO_TITLES[sc], fontsize=10)
        ax.set_ylim(0, 6)
        ax.set_xticks([])
        ax.set_yticks([0, 3, 6] if j == 0 else [])
        if j == 0:
            ax.set_ylabel("Mbps")
        ax = axes[1, j]
        means = [np.mean([t["qoe"] for t in res[p]["traces"] if t["trace"].startswith(sc)]) for p in pols]
        qoe_all += means
        y = np.arange(len(pols))[::-1]
        for yi, p, m in zip(y, pols, means):
            ax.barh(yi, m, height=0.66, color="none" if p == "Oracle" else color(p), edgecolor=color(p),
                    hatch="////" if p == "Oracle" else None, linewidth=1.0)
            ax.text(m if m > 0 else 0, yi, f" {m:.2f}", va="center", fontsize=7, color=INK)
        ax.axvline(0, color=INK_2, linewidth=0.6)
        ax.set_yticks(y, [label(p) for p in pols] if j == 0 else [])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("QoE / chunk")
    lo, hi = min(0, min(qoe_all)), max(qoe_all)
    for ax in axes[1]:
        ax.set_xlim(lo - 0.1, hi * 1.35)
    fig.suptitle("Stress-test scenarios: example bandwidth (top) and mean QoE over 6 variants (bottom)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, out_dir, "g7_scenarios")


# --------------------------------------------------------------------------- G1
def fig_training(run_dir, out_dir, baselines=None):
    log = Path(run_dir) / "log.csv"
    if not log.exists():
        print(f"  skip g1_training: {log} not found (train first)")
        return
    rows = read_csv(log)
    it = [r["iter"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    axes[0].plot(it, [r["val_qoe"] for r in rows], color=color("PPO"), label="PPO (validation)")
    for p, v in (baselines or {}).items():
        axes[0].axhline(v, color=color(p), linestyle="--" if p == "Oracle" else ":", linewidth=1.2, label=label(p))
    axes[0].set_xlabel("Training iteration")
    axes[0].set_ylabel("Mean QoE per chunk")
    axes[0].set_title("Learning curve")
    axes[0].legend(loc="lower right")
    axes[1].plot(it, [r["entropy"] for r in rows], color=INK_2)
    axes[1].set_xlabel("Training iteration")
    axes[1].set_ylabel("Policy entropy (nats)")
    axes[1].set_title("Exploration decays")
    fig.tight_layout()
    save(fig, out_dir, "g1_training")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(ROOT / "results" / "baselines"))
    p.add_argument("--run", default=str(ROOT / "runs" / "ppo_asym"), help="training run for the learning curve")
    p.add_argument("--out", default=str(ROOT / "plots" / "figures"))
    args = p.parse_args()

    style.apply()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sets = {"hsdpa/test": "hsdpa_test_lin", "fcc/test": "fcc_test_lin"}
    results = {s: load_results(Path(args.results) / d) for s, d in sets.items()}
    results = {s: r for s, r in results.items() if r}
    if not results:
        sys.exit("no evaluation logs found — run `python evaluate.py --oracle` first")

    fig_traces(out)
    fig_qoe_bars(results, out)
    fig_cdf(results, out)
    for s, res in results.items():
        fig_breakdown(res, s, out)
        fig_tradeoff(res, s, out)
    scen = load_results(Path(args.results) / "scenarios_lin")
    if scen:
        fig_scenarios(scen, out)
    else:
        print("  skip g7_scenarios: run `python evaluate.py --traces scenarios --oracle`")
    if "hsdpa/test" in results:
        fig_timeseries(results["hsdpa/test"], "hsdpa/test", out)
        ref = {pol: np.mean([t["qoe"] for t in results["hsdpa/test"][pol]["traces"]])
               for pol in ("RobustMPC", "Oracle") if pol in results["hsdpa/test"]}
        fig_training(args.run, out, ref)


if __name__ == "__main__":
    main()
