"""benchmark_sqrt_dt: the grid estimates of HGI_E on the toy against sqrt(dt), and their extrapolation (stored results only).

Data: results/benchmark_bias.json (toy, seeds 0-7, T = 20 000, one Brownian path per seed sampled at every step size):
  A_dt_<dt>.map         mean over the 8 seeds of the grid estimate at dt = 1e-2, 4e-3, 1e-3, 5e-4
  A_line_fit            least-squares line through the four means against sqrt(dt): intercept, intercept_se, slope, slope_se
and results/multi_bench_crn.json: networks["A toy (Ex. 2.2)"].HGI_E, .se, the value the report uses (extrapolated from
dt = 4e-3 and 1e-3 on paths generated at dt = 1e-3).
No error bars on the four grid points: their error (0.0136 to 0.0137) is common to all four, because the same paths
are used at every step size. The script stops without drawing if a plotted value does not round to the number printed
in the report. Writes benchmark_sqrt_dt.png (300 dpi) and .pdf to results/final/, or to DIR with --out DIR.
Run from sim/:  python make_benchmark_fig.py"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, SURF = "#0b0b0b", "#52514e", "#ffffff"
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "results/final"          # output folder
BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"          # Okabe-Ito
DTS = ("0.01", "0.004", "0.001", "0.0005")
EXPECTED = dict(grid=("2.6404", "2.6831", "2.7200", "2.7307"), intercept="2.7568", intercept_se="0.0137", slope="-1.164", slope_se="0.003",
                report="2.7628", report_se="0.0201")

if __name__ == "__main__":
    b = json.load(open("results/benchmark_bias.json")); fit = b["A_line_fit"]
    toy = json.load(open("results/multi_bench_crn.json"))["networks"]["A toy (Ex. 2.2)"]
    y = np.array([b["A_dt_" + d]["map"] for d in DTS]); se = np.array([b["A_dt_" + d]["map_se"] for d in DTS]); x = np.sqrt([float(d) for d in DTS])
    ls = np.polyfit(x, y, 1)
    assert abs(ls[1] - fit["intercept"]) < 1e-9 and abs(ls[0] - fit["slope"]) < 1e-9      # the stored fit is the line through the four means
    got = dict(grid=tuple(f"{v:.4f}" for v in y), intercept=f"{fit['intercept']:.4f}", intercept_se=f"{fit['intercept_se']:.4f}",
               slope=f"{fit['slope']:.3f}", slope_se=f"{fit['slope_se']:.3f}", report=f"{toy['HGI_E']:.4f}", report_se=f"{toy['se']:.4f}")
    print(f"{'dt':>8s} {'sqrt(dt)':>9s} {'estimate':>10s} {'report':>8s} {'SE over seeds (not drawn)':>26s}")
    for d, xx, v, g, s in zip(DTS, x, y, EXPECTED["grid"], se):
        print(f"{d:>8s} {xx:9.4f} {v:10.5f} {g:>8s} {s:26.5f}")
    print(f"fit: intercept {fit['intercept']:.5f} +- {fit['intercept_se']:.5f} (report {EXPECTED['intercept']} +- {EXPECTED['intercept_se']}); "
          f"slope {fit['slope']:.4f} +- {fit['slope_se']:.4f} (report {EXPECTED['slope']} +- {EXPECTED['slope_se']})")
    print(f"value used in the report: {toy['HGI_E']:.5f} +- {toy['se']:.5f} (report {EXPECTED['report']} +- {EXPECTED['report_se']})")
    bad = [f"{k}: stored {got[k]}, report {EXPECTED[k]}" for k in EXPECTED if got[k] != EXPECTED[k]]
    if bad:
        print("\nSTOPPED: plotted values do not round to the printed numbers:\n  " + "\n  ".join(bad)); sys.exit(1)

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.4, 2.4), facecolor=SURF)
    ax.set_facecolor(SURF); ax.grid(True, alpha=0.25, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=7, width=0.6, length=3)
    xs = np.array([0.0, 0.105])
    ax.plot(xs, fit["intercept"] + fit["slope"] * xs, "-", color=BLUE, lw=1.0, zorder=2, label="least-squares line")
    ax.plot(x, y, "o", color=BLUE, ms=4.2, zorder=3, label=r"grid estimates, $dt=10^{-2}$ to $5\cdot10^{-4}$")
    ax.errorbar([0.0], [fit["intercept"]], yerr=[fit["intercept_se"]], fmt="s", color=VERM, ms=4.2, elinewidth=0.9, capsize=2, capthick=0.7,
                zorder=4, label="intercept of the line")
    ax.errorbar([-0.004], [toy["HGI_E"]], yerr=[toy["se"]], fmt="D", color=GREEN, ms=4.0, elinewidth=0.9, capsize=2, capthick=0.7,
                zorder=4, label="value used in the report")
    ax.set_xlim(-0.009, 0.105); ax.set_xticks([0, 0.02, 0.04, 0.06, 0.08, 0.10]); ax.set_ylim(2.62, 2.80)
    ax.set_xlabel(r"$\sqrt{dt}$", color=INK2); ax.set_ylabel(r"estimate of $\mathrm{HGI}_E$", color=INK2)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right", handlelength=1.6, borderaxespad=0.2, labelspacing=0.3)
    fig.subplots_adjust(left=0.165, right=0.975, bottom=0.175, top=0.97)
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"benchmark_sqrt_dt.{ext}"), dpi=300, facecolor=SURF)
    print(f"\nsaved {OUT}/benchmark_sqrt_dt.png, {OUT}/benchmark_sqrt_dt.pdf")
