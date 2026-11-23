"""multi_convergence_v2: the Definition 2.3 runs of Table 3 on the seven networks (stored results only).

Data: results/decomp_runs.json (r = 1000, 2000) and results/table3_r4000.json (r = 4000): runs with cap = "uniform"
(Definition 2.3) at the rule band c2 of results/multi_bench.json, T = 250, batches of 25, seeds 1-2, 30% burn-in.
The pattern-dependent-cap runs stored in the same files are not read. Benchmarks: results/multi_bench_crn.json.
Excess = (T1 + T2)/HGI_E in percent = mean over the 14 pooled batches of (batches.hq - batches.hg) / HGI_E, with a
t-based 95% batch-means half-width.
Panel (a): excess against r, log-log, with a reference line of slope kappa - 1.
Panel (b): excess(r) / excess(1000) / (r/1000)^(kappa - 1); error bars = the relative half-widths of the two cells
added in quadrature; points are offset horizontally by up to 4% so that the bars can be told apart.
The script stops without drawing if a value does not round to the number printed in the report (EXPECTED below).
Writes multi_convergence_v2.png (300 dpi) and .pdf to results/final/, or to DIR with --out DIR; prints every plotted
number and the Table 3 cells to two decimals. Run from sim/:  python make_multi_fig.py"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter
from scipy.stats import t as tdist

KAPPA = 0.24
INK, INK2, SURF = "#0b0b0b", "#52514e", "#ffffff"
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "results/final"          # output folder
# Okabe-Ito colours without yellow, one marker per network
STYLE = {"A": ("#0072B2", "o"), "B": ("#E69F00", "s"), "C": ("#009E73", "^"), "D": ("#D55E00", "D"),
         "E": ("#CC79A7", "v"), "F": ("#56B4E9", "P"), "G": ("#000000", "X")}
EXPECTED = {"A": (68, "2.76", ("34.9 1.0", "21.0 0.6", "12.4 0.1")), "B": (13, "4.01", ("6.1 0.1", "3.7 0.0", "2.2 0.0")),
            "C": (68, "4.01", ("59.3 2.0", "34.1 0.8", "20.7 0.5")), "D": (41, "3.07", ("23.9 0.1", "14.3 0.1", "8.6 0.0")),
            "E": (62, "4.24", ("30.8 0.3", "18.5 0.2", "11.1 0.2")), "F": (152, "8.79", ("75.3 0.6", "44.9 0.6", "26.7 0.1")),
            "G": (660, "24.44", ("423 7", "281 8", "179 7"))}
RS = (1000, 2000, 4000)


def hw(x):
    return float(tdist.ppf(0.975, len(x) - 1) * np.std(x, ddof=1) / np.sqrt(len(x)))


def load():
    bn = json.load(open("results/multi_bench_crn.json"))["networks"]; bo = json.load(open("results/multi_bench.json"))
    dec = json.load(open("results/decomp_runs.json")); r4 = json.load(open("results/table3_r4000.json"))["runs"]
    cells = {}
    for name in bn:
        l = name[0]; H = bn[name]["HGI_E"]; c2 = bo[name]["c2"]
        for r in RS:
            src = dec if r < 4000 else r4
            runs = [o for o in src if o["net"] == name and o["cap"] == "uniform" and abs(o["c"][1] - c2) < 1e-6 and o["r"] == r and o["T"] == 250.0]
            assert sorted(o["seed"] for o in runs) == [1, 2], (name, r)
            t12 = np.concatenate([np.array(o["batches"]["hq"]) - np.array(o["batches"]["hg"]) for o in runs])
            assert len(t12) == 14
            cells[(l, r)] = dict(excess=100 * float(t12.mean()) / H, hw=100 * hw(t12) / H, c2=c2, H=H)
    return cells


def check(cells):
    bad = []
    for l, (c2, H, vals) in EXPECTED.items():
        if f"{cells[(l, 1000)]['c2']:.0f}" != str(c2) or f"{cells[(l, 1000)]['H']:.2f}" != H:
            bad.append(f"{l}: c2 or HGI_E: stored {cells[(l, 1000)]['c2']:.3f}, {cells[(l, 1000)]['H']:.4f}; report {c2}, {H}")
        for r, v in zip(RS, vals):
            c = cells[(l, r)]; s = f"{c['excess']:.0f} {c['hw']:.0f}" if l == "G" else f"{c['excess']:.1f} {c['hw']:.1f}"
            if s != v:
                bad.append(f"{l}, r = {r}: stored {c['excess']:.3f} +- {c['hw']:.3f}; report {v.replace(' ', ' +- ')}")
    return bad


def spread(logy, gap):
    """Push label positions apart (in log10 units) until neighbours are at least `gap` apart."""
    p = np.array(logy, float); order = np.argsort(p)
    for _ in range(500):
        moved = False
        for a, b in zip(order[:-1], order[1:]):
            d = p[b] - p[a]
            if d < gap:
                p[a] -= (gap - d) / 2; p[b] += (gap - d) / 2; moved = True
        if not moved:
            break
    return p


def style(ax):
    ax.set_facecolor(SURF); ax.grid(True, alpha=0.25, lw=0.5, which="both")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=7, width=0.6, length=3); ax.tick_params(which="minor", length=1.5, width=0.4)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.6)


if __name__ == "__main__":
    cells = load(); bad = check(cells)
    print(f"{'net':3s} {'c2':>8s} {'HGI_E':>8s} | " + " | ".join(f"{'r = %d (2 dec.)' % r:>18s} {'report':>11s}" for r in RS))
    for l in EXPECTED:
        print(f"{l:3s} {cells[(l, 1000)]['c2']:8.3f} {cells[(l, 1000)]['H']:8.4f} | " + " | ".join(
            f"{cells[(l, r)]['excess']:9.2f} +- {cells[(l, r)]['hw']:5.2f} {v.replace(' ', ' +- '):>11s}" for r, v in zip(RS, EXPECTED[l][2])))
    if bad:
        print("\nSTOPPED: stored values do not round to the printed numbers:\n  " + "\n  ".join(bad)); sys.exit(1)
    ratio = {}
    for l in EXPECTED:
        e0, h0 = cells[(l, 1000)]["excess"], cells[(l, 1000)]["hw"]
        for r in RS:
            c = cells[(l, r)]; v = c["excess"] / e0 / (r / 1000.0) ** (KAPPA - 1)
            ratio[(l, r)] = (v, 0.0 if r == 1000 else v * float(np.hypot(c["hw"] / c["excess"], h0 / e0)))
    print("\npanel (b): excess(r) / excess(1000) / (r/1000)^(kappa-1), +- from the two relative half-widths in quadrature")
    for l in EXPECTED:
        print(f"{l:3s} " + "   ".join(f"r = {r}: {ratio[(l, r)][0]:.3f} +- {ratio[(l, r)][1]:.3f}" for r in RS))

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 2.6), facecolor=SURF)
    for ax in (a1, a2):
        style(ax)
    rr = np.array(RS, float)
    for l, (col, mk) in STYLE.items():
        y = np.array([cells[(l, r)]["excess"] for r in RS]); e = np.array([cells[(l, r)]["hw"] for r in RS])
        a1.errorbar(rr, y, yerr=e, fmt="-", marker=mk, color=col, ms=3.6, lw=1.0, elinewidth=0.7, capsize=1.5, capthick=0.6, mec=col, zorder=3)
    ref = 160.0 * (rr / 1000.0) ** (KAPPA - 1)
    a1.plot(rr, ref, "--", color="#888888", lw=0.9, zorder=2)
    a1.text(2550, 160.0 * 2.55 ** (KAPPA - 1) * 1.16, r"slope $\kappa-1$", color=INK2, fontsize=7, ha="left", va="bottom")
    a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlim(800, 5000); a1.set_ylim(1.5, 1800)
    lab = spread([np.log10(cells[(l, 4000)]["excess"]) for l in STYLE], gap=0.15)
    for (l, (col, mk)), p in zip(STYLE.items(), lab):
        a1.text(4000 * 1.07, 10 ** p, l, color=col, fontsize=7, va="center", ha="left", fontweight="bold")
    a1.set_xticks(RS); a1.xaxis.set_major_formatter(ScalarFormatter()); a1.xaxis.set_minor_formatter(NullFormatter()); a1.minorticks_on()
    a1.xaxis.set_tick_params(which="minor", bottom=False)
    a1.set_xlabel("r", color=INK2); a1.set_ylabel(r"excess $(T_1+T_2)/\mathrm{HGI}_E$ (%)", color=INK2)
    a1.text(0.03, 0.97, "(a)", transform=a1.transAxes, ha="left", va="top", fontsize=8, color=INK)

    dodge = dict(zip(STYLE, np.linspace(-0.04, 0.04, len(STYLE))))
    for l, (col, mk) in STYLE.items():
        y = np.array([ratio[(l, r)][0] for r in RS]); e = np.array([ratio[(l, r)][1] for r in RS])
        x = rr * np.array([1.0, 1 + dodge[l], 1 + dodge[l]])
        a2.errorbar(x, y, yerr=e, fmt="-", marker=mk, color=col, ms=3.6, lw=1.0, elinewidth=0.7, capsize=1.5, capthick=0.6, mec=col, zorder=3)
    a2.axhline(1.0, color="#888888", ls="--", lw=0.9, zorder=2)
    a2.set_xscale("log"); a2.set_xlim(800, 5000)
    a2.text(4000 * 1.07, ratio[("G", 4000)][0], "G", color=STYLE["G"][0], fontsize=7, va="center", ha="left", fontweight="bold")
    a2.text(4000 * 1.07, np.mean([ratio[(l, 4000)][0] for l in "ABCDEF"]), "A–F", color=INK2, fontsize=7, va="center", ha="left")
    a2.set_xticks(RS); a2.xaxis.set_major_formatter(ScalarFormatter()); a2.xaxis.set_minor_formatter(NullFormatter())
    a2.xaxis.set_tick_params(which="minor", bottom=False)
    a2.set_xlabel("r", color=INK2); a2.set_ylabel(r"excess relative to $r^{\kappa-1}$ decay", color=INK2)
    a2.text(0.03, 0.97, "(b)", transform=a2.transAxes, ha="left", va="top", fontsize=8, color=INK)
    fig.subplots_adjust(left=0.085, right=0.965, bottom=0.165, top=0.97, wspace=0.30)
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"multi_convergence_v2.{ext}"), dpi=300, facecolor=SURF)
    print(f"\nsaved {OUT}/multi_convergence_v2.png, {OUT}/multi_convergence_v2.pdf")
