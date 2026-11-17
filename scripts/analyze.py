#!/usr/bin/env python3
"""Turns the benchmark CSVs into the report's figures and summary tables.

usage: python scripts/analyze.py [--raw results_vanda/raw] [--out results]

Writes <out>/figs/*.png and <out>/tables/*.csv. Experiments whose CSV is missing are skipped.
All statistics are medians over repetitions; error bars show the interquartile range (robust to
the occasional stalled run). Speedups use the V0 baseline from the same job, i.e. the same node.
"""
import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

# Categorical slots 1-3 of the validated reference palette (light mode), plus chart ink.
# No figure uses more than three series; each series also gets its own marker shape.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
VERSIONS = {  # version -> (color, marker, label)
    "v1_openmp": (BLUE, "o", "V1 OpenMP"),
    "v2_mpi": (ORANGE, "s", "V2 MPI"),
    "v3_hybrid": (AQUA, "^", "V3 hybrid (4 threads/rank)"),
}
A40_PEAK_GBS = 696.0
SOCKET_PEAK_GBS = 307.2  # 8 x DDR5-4800 per Sapphire Rapids socket


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "figure.facecolor": "white",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


# ---------------------------------------------------------------- data helpers

def load(raw, name):
    path = raw / name
    if not path.exists():
        return None
    d = pd.read_csv(path)
    d["cores"] = d["ranks"] * d["threads"]
    d["cells"] = (d["nx"] - 2) * (d["ny"] - 2) * d["iters"]
    d["bw"] = 16 * d["cells"] / d["t_total"] / 1e9  # effective GB/s (one 8 B read + one 8 B write per cell)
    return d


def summarize(d, keys):
    q = lambda p: (lambda s: s.quantile(p))  # noqa: E731
    g = d.groupby(keys, dropna=False).agg(
        t=("t_total", "median"), t_q1=("t_total", q(0.25)), t_q3=("t_total", q(0.75)),
        comp=("t_compute", "median"), comm=("t_comm", "median"), bw=("bw", "median"),
        n=("t_total", "size"),
    ).reset_index()
    return g


def baseline(d):
    return d.loc[d["version"] == "v0_serial", "t_total"].median()


def add_speedup(g, t0):
    g["speedup"] = t0 / g["t"]
    g["sp_lo"] = t0 / g["t_q3"]
    g["sp_hi"] = t0 / g["t_q1"]
    g["efficiency"] = g["speedup"] / g["cores"]
    return g


def node_label(d):
    return f"{d['host'].iloc[0]}"


# ---------------------------------------------------------------- plot helpers

def log2_axis(ax, axis, values):
    ticks = sorted(set(int(v) for v in values))
    getattr(ax, f"set_{axis}scale")("log", base=2)
    loc = getattr(ax, f"{axis}axis")
    loc.set_major_locator(FixedLocator(ticks))
    loc.set_minor_locator(NullLocator())
    getattr(ax, f"set_{axis}ticklabels")([str(t) for t in ticks])


def ideal_line(ax, xs):
    xs = sorted(xs)
    ax.plot(xs, xs, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), label="ideal", zorder=1)


def series(ax, x, y, lo, hi, color, marker, label):
    ax.errorbar(x, y, yerr=[y - lo, hi - y], color=color, marker=marker, label=label,
                capsize=0, elinewidth=1.0, markeredgecolor="white", markeredgewidth=1.2, zorder=3)


def stacked(ax, labels, parts, colors, names, width=0.7):
    """Vertical stacked bars with a thin white gap between segments."""
    bottom = [0.0] * len(labels)
    x = range(len(labels))
    for vals, color, name in zip(parts, colors, names):
        ax.bar(x, vals, width, bottom=bottom, color=color, label=name, edgecolor="white", linewidth=1.0)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.grid(axis="x", visible=False)
    return bottom


def suptitle(fig, text):
    fig.suptitle(text, x=0.01, y=1.04, ha="left", fontsize=10, color=INK2)


def dodge(x, k, n, step=1.06):
    """Multiplicative x offset for series k of n on a log axis, so coinciding markers stay visible."""
    return x * step ** (k - (n - 1) / 2)


def save(fig, out, name):
    path = out / "figs" / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def write_table(g, out, name, cols):
    path = out / "tables" / name
    g[cols].round(4).to_csv(path, index=False)
    print(f"  wrote {path}")


# ---------------------------------------------------------------- experiments

def e1(raw, out):
    d64, d32 = load(raw, "e1_openmp_64c.csv"), load(raw, "e1_openmp_32c.csv")
    if d64 is None and d32 is None:
        return
    print("E1 OpenMP thread scaling")
    for d, tag in [(d64, "64c"), (d32, "32c")]:
        if d is None:
            continue
        g = add_speedup(summarize(d[d.version == "v1_openmp"], ["bind", "threads", "cores"]), baseline(d))
        write_table(g, out, f"e1_{tag}.csv", ["bind", "threads", "t", "speedup", "efficiency", "bw", "n"])

    d = d64 if d64 is not None else d32
    g = add_speedup(summarize(d[d.version == "v1_openmp"], ["bind", "threads", "cores"]), baseline(d))
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    styles = {"close": (BLUE, "o", "close (fills socket 0 first)"), "spread": (ORANGE, "s", "spread (both sockets)")}
    for bind, (color, marker, label) in styles.items():
        s = g[g.bind == bind].sort_values("threads")
        series(a, s.threads, s.speedup, s.sp_lo, s.sp_hi, color, marker, label)
        b.plot(s.threads, s.bw, color=color, marker=marker, label=label,
               markeredgecolor="white", markeredgewidth=1.2, zorder=3)
    ideal_line(a, g.threads.unique())
    log2_axis(a, "x", g.threads)
    log2_axis(a, "y", [1, 2, 4, 8, 16, 32, 64])
    a.set(xlabel="OpenMP threads", ylabel="speedup vs V0", title="(a) Speedup")
    a.legend(loc="upper left")
    for peak, txt in [(SOCKET_PEAK_GBS, "1-socket peak"), (2 * SOCKET_PEAK_GBS, "node peak")]:
        b.axhline(peak, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        b.text(1, peak, f"  {txt} {peak:.0f} GB/s", color=INK2, fontsize=7.5, va="bottom")
    log2_axis(b, "x", g.threads)
    b.set_ylim(0, 2 * SOCKET_PEAK_GBS * 1.08)
    b.set(xlabel="OpenMP threads", ylabel="effective bandwidth (GB/s)", title="(b) Memory bandwidth")
    b.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88))
    suptitle(fig, f"E1  V1 thread scaling, 4096², {node_label(d)}")
    save(fig, out, "e1_openmp.png")


def e2(raw, out):
    d = load(raw, "e2_strong_32c.csv")
    if d is None:
        return
    print("E2 strong scaling")
    g = add_speedup(summarize(d[d.version != "v0_serial"], ["version", "cores"]), baseline(d))
    g["comm_pct"] = 100 * g["comm"] / g["t"]
    write_table(g, out, "e2_strong.csv", ["version", "cores", "t", "speedup", "efficiency", "comm_pct", "bw", "n"])

    fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    for k, (v, (color, marker, label)) in enumerate(VERSIONS.items()):
        s = g[g.version == v].sort_values("cores")
        # The three versions nearly coincide (that is the finding); a small sideways offset
        # keeps every series visible.
        series(a, dodge(s.cores, k, 3), s.speedup, s.sp_lo, s.sp_hi, color, marker, label)
        if v != "v1_openmp":
            b.plot(s.cores, s.comm_pct, color=color, marker=marker, label=label,
                   markeredgecolor="white", markeredgewidth=1.2, zorder=3)
    ideal_line(a, g.cores.unique())
    log2_axis(a, "x", g.cores)
    log2_axis(a, "y", [1, 2, 4, 8, 16, 32])
    a.set(xlabel="cores", ylabel="speedup vs V0", title="(a) Speedup (series offset sideways)")
    a.legend(loc="upper left")
    log2_axis(b, "x", g.cores)
    b.set_ylim(0, None)
    b.set(xlabel="cores", ylabel="halo exchange incl. waiting (% of runtime)",
          title="(b) Communication share")
    b.legend(loc="upper left")
    suptitle(fig, f"E2  strong scaling, 8192², one socket of {node_label(d)}")
    save(fig, out, "e2_strong.png")


def e3(raw, out):
    d = load(raw, "e3_weak_32c.csv")
    if d is None:
        return
    print("E3 weak scaling")
    g = summarize(d[d.version != "v0_serial"], ["version", "cores", "ny"])
    t1 = g.loc[(g.version == "v2_mpi") & (g.cores == 1), "t"].iloc[0]
    g["weak_eff"] = t1 / g["t"]
    g["comm_pct"] = 100 * g["comm"] / g["t"]
    g["bw_per_core"] = g["bw"] / g["cores"]
    write_table(g, out, "e3_weak.csv", ["version", "cores", "ny", "t", "comp", "comm", "comm_pct",
                                        "weak_eff", "bw_per_core", "n"])

    g["comm_ms_per_iter"] = 1e3 * g["comm"] / d["iters"].iloc[0]
    # Compute and halo times are each the slowest rank's value (and halo time includes waiting
    # for neighbours), so they are plotted separately rather than stacked.
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(11.0, 3.3))
    for v in ["v2_mpi", "v3_hybrid"]:
        color, marker, label = VERSIONS[v]
        s = g[g.version == v].sort_values("cores")
        a.errorbar(s.cores, s.t, yerr=[s.t - s.t_q1, s.t_q3 - s.t], color=color, marker=marker, label=label,
                   capsize=0, elinewidth=1.0, markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        b.plot(s.cores, s.comm_ms_per_iter, color=color, marker=marker, label=label,
               markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        c.plot(s.cores, s.bw_per_core, color=color, marker=marker, label=label,
               markeredgecolor="white", markeredgewidth=1.2, zorder=3)
    for ax in (a, b, c):
        log2_axis(ax, "x", g.cores)
        ax.set_ylim(0, None)
        ax.set_xlabel("cores (512 rows each)")
    a.set(ylabel="runtime (s)", title="(a) Runtime (ideal: flat)")
    b.set(ylabel="halo exchange per iteration (ms)", title="(b) Halo exchange incl. waiting")
    c.set(ylabel="effective bandwidth per core (GB/s)", title="(c) Bandwidth per core")
    a.legend(loc="upper left")
    suptitle(fig, f"E3  weak scaling, nx = 4096, one socket of {node_label(d)}")
    save(fig, out, "e3_weak.png")


def e4(raw, out):
    files = [(f, load(raw, f)) for f in ["e4_hybrid_32c.csv", "e4_hybrid_64c.csv"]]
    files = [(f, d) for f, d in files if d is not None]
    if not files:
        return
    print("E4 ranks x threads")
    # Top row: total runtime per split. Bottom row: halo exchange time (slowest rank, incl.
    # waiting). Not stacked: the two maxima can come from different ranks.
    fig, axes = plt.subplots(2, len(files), figsize=(4.4 * len(files), 4.6), squeeze=False,
                             gridspec_kw={"height_ratios": [2, 1]}, sharex="col")
    for col, (f, d) in enumerate(files):
        top, bot = axes[0][col], axes[1][col]
        g = add_speedup(summarize(d[d.version == "v3_hybrid"], ["ranks", "threads", "cores"]), baseline(d))
        g = g.sort_values("ranks").reset_index(drop=True)
        g["comm_pct"] = 100 * g["comm"] / g["t"]
        tag = f.split("_")[-1].replace(".csv", "")
        write_table(g, out, f"e4_{tag}.csv", ["ranks", "threads", "t", "comp", "comm", "comm_pct", "speedup", "n"])
        x = list(range(len(g)))
        top.bar(x, g.t, width=0.65, color=BLUE)
        top.errorbar(x, g.t, yerr=[g.t - g.t_q1, g.t_q3 - g.t], fmt="none", ecolor=INK2, elinewidth=1.0)
        # Selective labels: the pure-OpenMP split and the fastest split only.
        best = int(g.speedup.idxmax())
        for k in sorted({0, best}):
            top.text(k, g.t_q3[k] * 1.01, f"{g.speedup[k]:.1f}×", ha="center", va="bottom", fontsize=7.5,
                     color=INK2)
        top.set_ylim(0, g.t.max() * 1.15)
        bot.bar(x, g.comm, width=0.65, color=ORANGE)
        for ax in (top, bot):
            ax.grid(axis="x", visible=False)
        bot.set_xticks(x)
        bot.set_xticklabels([f"{r}×{t}" for r, t in zip(g.ranks, g.threads)], fontsize=7.5)
        cores = int(g.cores.iloc[0])
        where = "one socket" if cores <= 36 else "two sockets"
        top.set(ylabel="runtime (s)", title=f"{cores} cores, {where} ({node_label(d)})")
        bot.set(xlabel="ranks × threads", ylabel="halo exch. (s)")
    suptitle(fig, "E4  hybrid splits, 8192² (labels: speedup vs V0; bottom: halo exchange incl. waiting)")
    save(fig, out, "e4_hybrid.png")


def e5(raw, out):
    gpu_file = next((f for f in ["e5_gpu.csv", "e5_gpu_1g.csv", "e5_gpu_interactive.csv"] if (raw / f).exists()), None)
    if gpu_file is None:
        print("E5: no GPU data yet - skipped")
        return
    print(f"E5 GPU (from {gpu_file})")
    d = load(raw, gpu_file)
    d["t_copy"] = pd.to_numeric(d["t_copy"], errors="coerce")
    d["gcells"] = d["cells"] / d["t_total"] / 1e9

    # E5a: block sizes
    a5 = d[d.exp == "E5a"].groupby(["bx", "by"]).agg(t=("t_total", "median"), bw=("bw", "median")).reset_index()
    if len(a5):
        a5["pct_peak"] = 100 * a5["bw"] / A40_PEAK_GBS
        a5 = a5.sort_values("pct_peak")
        write_table(a5, out, "e5a_blocks.csv", ["bx", "by", "t", "bw", "pct_peak"])
        fig, ax = plt.subplots(figsize=(5.0, 3.2))
        y = range(len(a5))
        ax.barh(list(y), a5.pct_peak, height=0.6, color=BLUE)
        ax.set_yticks(list(y))
        ax.set_yticklabels([f"{bx}×{by}" for bx, by in zip(a5.bx, a5.by)])
        for k, v in enumerate(a5.pct_peak):
            ax.text(v, k, f" {v:.0f}%", va="center", fontsize=7.5, color=INK2)
        ax.grid(axis="y", visible=False)
        ax.set(xlabel=f"effective bandwidth (% of A40 peak, {A40_PEAK_GBS:.0f} GB/s)", ylabel="block (x × y)",
               title="E5a  CUDA block size, 8192², 1 GPU")
        ax.set_xlim(0, 100)
        save(fig, out, "e5a_blocks.png")

    # E5b: throughput vs grid size, GPU vs CPU
    b5 = d[d.exp == "E5b"].groupby("nx").agg(gc=("gcells", "median")).reset_index()
    if len(b5):
        fig, ax = plt.subplots(figsize=(5.0, 3.2))
        ax.plot(b5.nx, b5.gc, color=BLUE, marker="o", label="V4, 1 A40", markeredgecolor="white",
                markeredgewidth=1.2, zorder=3)
        rows = [b5.assign(device="gpu_1")]
        d2 = d[d.exp == "E5d_strong"].groupby("nx").agg(gc=("gcells", "median")).reset_index()
        if len(d2):  # two real GPUs (not the shared-GPU fallback)
            ax.plot(d2.nx, d2.gc, color=BLUE, marker="D", linestyle=(0, (5, 2)), label="V4, 2 A40s",
                    markeredgecolor="white", markeredgewidth=1.2, zorder=3)
            rows.append(d2.assign(device="gpu_2"))
        # One line per (file, node): repeated jobs on different nodes are shown, not averaged together.
        styles = iter([(ORANGE, "s"), (AQUA, "^"), (MUTED, "D"), (INK2, "v")])
        for f in ["e5b_cpu_32c.csv", "e5b_cpu_v1_32c.csv", "e5b_cpu_64c.csv"]:
            c = load(raw, f)
            if c is None:
                continue
            c["gcells"] = c["cells"] / c["t_total"] / 1e9
            for host, ch in c.groupby("host", sort=False):
                color, marker = next(styles)
                cg = ch.groupby("nx").agg(gc=("gcells", "median")).reset_index()
                cores = int(ch["alloc_cores"].iloc[0])
                ver = ch.version.iloc[0].split("_")[0].upper()
                ax.plot(cg.nx, cg.gc, color=color, marker=marker, label=f"{ver}, {cores} CPU cores ({host})",
                        markeredgecolor="white", markeredgewidth=1.2, zorder=3)
                rows.append(cg.assign(device=f"cpu_{ver.lower()}_{cores}c_{host}"))
        write_table(pd.concat(rows), out, "e5b_gpu_vs_cpu.csv", ["device", "nx", "gc"])
        log2_axis(ax, "x", b5.nx)
        ax.set_ylim(0, None)
        ax.set(xlabel="grid size N (N × N)", ylabel="throughput (10⁹ cell updates/s)",
               title="E5b  GPU vs CPU across grid sizes")
        ax.legend(loc="best")
        save(fig, out, "e5b_gpu_vs_cpu.png")

    # E5c / E5d: 2-rank runs (breakdown) and 1 -> 2 GPU scaling
    two = d[d.exp.str.endswith("_strong")]
    if len(two):
        exp = two.exp.iloc[0]
        c5 = two.groupby("nx").agg(t=("t_total", "median"), kern=("t_compute", "median"),
                                   copy=("t_copy", "median"), mpi=("t_comm", "median")).reset_index()
        one = d[d.exp == "E5b"].groupby("nx").agg(t1=("t_total", "median")).reset_index()
        c5 = c5.merge(one, on="nx", how="left")
        c5["speedup_2"] = c5["t1"] / c5["t"]
        for k in ["kern", "copy", "mpi"]:
            c5[f"{k}_pct"] = 100 * c5[k] / c5["t"]
        write_table(c5, out, "e5cd_two_ranks.csv", ["nx", "t1", "t", "speedup_2", "kern_pct", "copy_pct", "mpi_pct"])
        fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 3.4))
        # Side by side, not stacked: each component is the maximum over ranks (possibly different ranks),
        # so the three shares need not add up to 100 %.
        w = 0.26
        for k, (col, color, lab) in enumerate([("kern_pct", BLUE, "kernel"), ("copy_pct", ORANGE, "GPU↔host halo copies"),
                                               ("mpi_pct", AQUA, "MPI (incl. waiting)")]):
            a.bar([i + (k - 1) * w for i in range(len(c5))], c5[col], width=w, color=color, label=lab,
                  edgecolor="white", linewidth=0.8)
        a.set_xticks(range(len(c5)))
        a.set_xticklabels([str(n) for n in c5.nx])
        a.grid(axis="x", visible=False)
        a.set_ylim(0, 125)  # headroom above 100 % holds the legend clear of the bars
        a.set_yticks([0, 20, 40, 60, 80, 100])
        a.set(xlabel="grid size N", ylabel="% of runtime (max over ranks)", title="(a) Where the time goes, 2 ranks")
        a.legend(loc="upper left", ncol=3, fontsize=7.0, handlelength=1.0, columnspacing=0.8)
        b.bar(range(len(c5)), c5.speedup_2, width=0.6, color=BLUE)
        b.axhline(2, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
        b.text(len(c5) - 0.5, 2, "ideal 2×", ha="right", va="bottom", fontsize=7.5, color=INK2)
        b.set_xticks(range(len(c5)))
        b.set_xticklabels([str(n) for n in c5.nx])
        b.grid(axis="x", visible=False)
        b.set(xlabel="grid size N", ylabel="speedup, 2 vs 1 rank", title="(b) Strong scaling 1 → 2")
        shared = "shared" in exp
        suptitle(fig, "E5c/d  two ranks on " + ("ONE shared GPU" if shared else "two A40 GPUs"))
        save(fig, out, "e5cd_two_ranks.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default="results_vanda/raw", help="directory with the benchmark CSVs")
    p.add_argument("--out", default="results", help="output directory (figs/ and tables/ inside)")
    a = p.parse_args()
    raw, out = pathlib.Path(a.raw), pathlib.Path(a.out)
    (out / "figs").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    setup_style()
    for step in (e1, e2, e3, e4, e5):
        step(raw, out)


if __name__ == "__main__":
    main()
