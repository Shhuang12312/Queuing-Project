#!/usr/bin/env python3
"""Collects results/*.json into one table per constant set and writes summary.csv.

    python collect.py                 # table on screen + summary.csv
    python collect.py --plot          # also excess_vs_r.png (needs matplotlib)

Per (c1, c2, kappa, r, T): the runs are pooled over seeds; +- is a 95% half-width from the pooled batch means
(batches are T/10 long unless --batch was given). "halves" is the change of T1+T2 from the first to the second half
of the post-burn window (a large change means the run is not stationary). SE(seeds) is the standard error of
T1+T2 across seeds; with short batches (T = 20) trust it more than the batch-means half-width. The rows of the report's Table 2
(r = 300 .. 10^4, c2 = 60) are printed first for comparison."""
import argparse, csv, glob, json, math, os
import numpy as np
from scipy.stats import t as tdist

HERE = os.path.dirname(os.path.abspath(__file__))
B = json.load(open(os.path.join(HERE, "benchmark.json"))); H = B["HGI_E"]


def hw(x):
    x = np.asarray(x, float); n = len(x)
    return float(tdist.ppf(0.975, n - 1) * x.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")


def summarize(runs):
    runs = sorted(runs, key=lambda o: o["seed"]); r, U = runs[0]["r"], runs[0]["U"]
    hq = np.concatenate([o["batches"]["hq"] for o in runs]); hh = np.concatenate([o["batches"]["hhat"] for o in runs])
    hg = np.concatenate([o["batches"]["hg"] for o in runs]); t12 = hq - hg
    a = np.mean([o["T12_half1"] for o in runs]); b = np.mean([o["T12_half2"] for o in runs])
    per_seed = np.array([o["T12"] for o in runs])
    return dict(c1=runs[0]["c"][0], c2=runs[0]["c"][1], kappa=runs[0]["c"][2], r=r, T=runs[0]["T"], seeds=len(runs), U=U,
                JE=float(hq.mean()), JE_hw=hw(hq), T1=float((hq - hh).mean()), T2=float((hh - hg).mean()), T2_hw=hw(hh - hg),
                excess_pct=100 * float(t12.mean()) / H, excess_hw=100 * hw(t12) / H,
                excess_se_seeds=(100 * float(per_seed.std(ddof=1) / math.sqrt(len(per_seed))) / H if len(per_seed) > 1 else float("nan")),
                T3=float(hg.mean()) - H, T3_hw=hw(hg), halves_change_pct=100 * (b - a) / a,
                phi=float((hq - hh).mean()) * r / U, T2_over_T1=float((hh - hg).mean() / (hq - hh).mean()),
                E_frac=float(np.mean([o["E_frac"] for o in runs])),
                predicted_excess_pct=float(np.mean([o["predicted"]["excess_pct_line"] for o in runs])),
                predicted_flag_drift=float(np.mean([o["predicted"]["flag_drift"] for o in runs])),
                hours_per_run=float(np.mean([o["wall_s"] for o in runs])) / 3600.0)


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--results", default=os.path.join(HERE, "results")); ap.add_argument("--plot", action="store_true")
a = ap.parse_args()
print(f"benchmark: HGI_E = {H:.4f} +- {B['se']:.4f} (SE, {B['n_seeds']} seeds; {B['source']})\n")
print("Report, Table 2 (toy, (c1, c2, kappa) = (0.1, 60, 0.24), 2 seeds; T = 250, at r = 10^4 T = 50):")
print(f"{'r':>8s} {'J_E':>14s} {'T1':>8s} {'T2':>16s} {'T1+T2 (% HGI_E)':>18s} {'T1/(U/r)':>9s}")
for x in json.load(open(os.path.join(HERE, "reference_table2.json"))):
    print(f"{x['r']:8d} {x['JE']:6.2f} +- {x['JE_hw']:4.2f} {x['T1']:8.4f} {x['T2']:7.4f} +- {x['T2_hw']:6.4f} {x['excess_pct']:9.1f} +- {x['excess_hw']:4.1f} {x['phi']:9.3f}")
runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.results, "toy_r*.json")))]
if not runs:
    print(f"\nno runs in {a.results} yet"); raise SystemExit
groups = {}
for o in runs:
    groups.setdefault((o["c"][0], o["c"][1], o["c"][2], o["r"], o["T"]), []).append(o)
rows = [summarize(g) for _, g in sorted(groups.items())]
for cs in sorted({(x["c1"], x["c2"], x["kappa"]) for x in rows}):
    sub = [x for x in rows if (x["c1"], x["c2"], x["kappa"]) == cs]
    print(f"\nYour runs, (c1, c2, kappa) = {cs}:")
    print(f"{'r':>8s} {'T':>5s} {'seeds':>5s} {'J_E':>14s} {'T1':>9s} {'T2':>18s} {'T1+T2 (% HGI_E)':>18s} {'SE(seeds)':>9s} {'line 2.83U/r':>12s} {'halves':>7s} "
          f"{'T3':>14s} {'T1/(U/r)':>9s} {'T2/T1':>6s} {'flags on':>9s} {'h/run':>6s}")
    for x in sub:
        print(f"{x['r']:8g} {x['T']:5g} {x['seeds']:5d} {x['JE']:6.2f} +- {x['JE_hw']:4.2f} {x['T1']:9.5f} {x['T2']:8.5f} +- {x['T2_hw']:7.5f} "
              f"{x['excess_pct']:9.2f} +- {x['excess_hw']:4.2f} {x['excess_se_seeds']:9.2f} {x['predicted_excess_pct']:12.2f} {x['halves_change_pct']:+6.1f}% "
              f"{x['T3']:+6.2f} +- {x['T3_hw']:4.2f} {x['phi']:9.3f} {x['T2_over_T1']:6.2f} {x['E_frac']:9.1e} {x['hours_per_run']:6.2f}")
    if len(sub) >= 2:
        lr = np.log([x["r"] for x in sub]); le = np.log([x["excess_pct"] for x in sub])
        slope = float(np.polyfit(lr, le, 1)[0])
        print(f"   slope of log(T1+T2) against log(r) over these r: {slope:+.3f}   (kappa - 1 = {cs[2] - 1:+.2f})")
keys = list(rows[0])
with open(os.path.join(HERE, "summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
print("\nwritten: summary.csv")
if a.plot:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 4.2), facecolor="#fcfcfb"); ax.set_facecolor("#fcfcfb")
    ref = json.load(open(os.path.join(HERE, "reference_table2.json")))
    ax.errorbar([x["r"] for x in ref], [x["excess_pct"] for x in ref], yerr=[x["excess_hw"] for x in ref], fmt="o-", color="#1baf7a", ms=5, capsize=2, label="report, Table 2 ($c_2=60$)")
    for cs, col in zip(sorted({(x["c1"], x["c2"], x["kappa"]) for x in rows}), ["#2a78d6", "#eb6834", "#77a9e6", "#52514e"]):
        sub = [x for x in rows if (x["c1"], x["c2"], x["kappa"]) == cs]
        ax.errorbar([x["r"] for x in sub], [x["excess_pct"] for x in sub], yerr=[x["excess_hw"] for x in sub], fmt="s-", color=col, ms=5, capsize=2, label=f"new runs, $c_2={cs[1]:g}$")
    rr = np.logspace(2.4, 5.6, 50); ax.plot(rr, 100 * 2.83 * 60 * rr ** (0.24 - 1) / H, ":", color="#52514e", lw=1.2, label=r"$2.83\,U/r$, $U=60r^{0.24}$")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("r"); ax.set_ylabel(r"$T_1+T_2$, % of $\mathrm{HGI}_E$"); ax.grid(True, alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(HERE, "excess_vs_r.png"), dpi=170, facecolor="#fcfcfb")
    print("written: excess_vs_r.png")
