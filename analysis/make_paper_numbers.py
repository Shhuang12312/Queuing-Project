"""Tables 1 to 3 of the report and the in-text numbers that depend on the benchmark, computed from stored result
files only (no simulation is run here).

    python analysis/make_paper_numbers.py              (from anywhere; about 10 s)
    python analysis/make_paper_numbers.py --out DIR    (write the two output files to DIR)

Reads (all under sim/results/):
  multi_bench_crn.json     HGI_E per network, common-path sqrt(dt) extrapolation, 8 seeds  (the benchmark of the report)
  multi_bench.json         HGI_E with independent extrapolation legs, b_max and the rule bands c2
  hgi_d_toy.json, hgi_d_toy_crn.json   HGI_D for the toy
  paper_cells.json         Table 2 runs (toy, (0.1, 60, 0.24), seeds 1-2)
  drift_check.json         seeds 3-8 at r = 1000, 2000 for the same cell; the c2 = 30, r = 400 long runs
  decomp_runs.json         Table 3 at r = 1000, 2000 (T = 250, batches of 25, seeds 1-2, both caps)
  table3_r4000.json        Table 3 at r = 4000 (same configuration); falls back to table3_r4000/<cell>.json
  production.json, table1_drift.json   Table 1
  vbcap_runs.json          the c2 scan on G under the pattern-dependent cap
  network_g_check.json     G: specification and predicted flag drift
  control_experiment.json, screen_r300.json, split_r100.json   in-text numbers of the control, the screen, the eta-split
Formula only (no data): sim/flagmodel.g_flag for the predicted flag drift, with d = 0.976896.

Writes paper_numbers.json (every number with its formatted text, source file, key and value; LaTeX rows under `latex`)
and paper_numbers.txt (the printout) next to this script. Percentages of HGI_E use the benchmark of multi_bench_crn.json.
"per_pattern" = the pattern-dependent cap, which is OUTSIDE Definition 2.3 (a labeled experiment)."""
import glob, json, math, os, sys
import numpy as np
from scipy.stats import t as tdist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM = os.path.join(ROOT, "sim"); RES = os.path.join(SIM, "results")
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIM)
from flagmodel import g_flag                      # noqa: E402  (formula (flag); no data)

KAPPA, C1, D_SERVICE = 0.24, 0.1, 0.976896
J = lambda n: json.load(open(os.path.join(RES, n)))
NUM = {}                                          # id -> dict(text=..., source=..., key=..., value=...)
LINES = []


def out(s=""):
    print(s); LINES.append(s)


def put(id_, text, source="", key="", value=None):
    """Register one number of the report: formatted text, source file, key in that file, value."""
    NUM[id_] = dict(text=text, source=source, key=key, value=value)


def hw(x, level=0.95):
    x = np.asarray(x, float); n = len(x)
    return float(tdist.ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")


def se(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")


def halves(x):
    """First-half and second-half averages of a per-batch series (odd counts split the middle batch)."""
    x = np.asarray(x, float); n = len(x); w1 = np.ones(n)
    if n % 2:
        w1[n // 2] = 0.5; w1[n // 2 + 1:] = 0
    else:
        w1[n // 2:] = 0
    w2 = 1 - w1
    return float((x * w1).sum() / w1.sum()), float((x * w2).sum() / w2.sum())


def B(o, k):
    return np.asarray(o["batches"][k], float)


def cell(runs, H, H_hw=0.0):
    """Pooled batch-means summary of one (network, cap, band, r) cell. H = benchmark; percentages divide by H."""
    runs = sorted(runs, key=lambda o: o["seed"])
    hq = np.concatenate([B(o, "hq") for o in runs]); hh = np.concatenate([B(o, "hhat") for o in runs]); hg = np.concatenate([B(o, "hg") for o in runs])
    t1, t2, t12, t3 = hq - hh, hh - hg, hq - hg, hg - H
    per_seed = [halves(B(o, "hq") - B(o, "hg")) for o in runs]
    a, b = float(np.mean([p[0] for p in per_seed])), float(np.mean([p[1] for p in per_seed]))
    r, U = runs[0]["r"], runs[0]["U"]
    return dict(seeds=[o["seed"] for o in runs], T=runs[0]["T"], n_batches=len(hq), r=r, U=U, c2=runs[0]["c"][1],
                JE=float(hq.mean()), JE_hw=hw(hq), T1=float(t1.mean()), T1_hw=hw(t1), T1_se=se([float((B(o, "hq") - B(o, "hhat")).mean()) for o in runs]),
                T2=float(t2.mean()), T2_hw=hw(t2), T12=float(t12.mean()), T12_hw=hw(t12),
                T3=float(t3.mean()), T3_hw=hw(t3), T3_hw_with_benchmark=float(math.hypot(hw(t3), H_hw)),
                excess=100 * float(t12.mean()) / H, excess_hw=100 * hw(t12) / H,
                T1_over_band=float(t1.mean() * r / U), T2_over_T1=(float(t2.mean() / t1.mean()) if t1.mean() > 1e-9 else float("nan")),
                halves_pooled=[a, b], halves_change_pct=100 * (b - a) / a,
                halves_per_seed=[dict(seed=o["seed"], first=p[0], second=p[1], change_pct=100 * (p[1] - p[0]) / p[0]) for o, p in zip(runs, per_seed)],
                E_frac=float(np.mean([o["E_frac"] for o in runs])), p_flag=np.mean([o["p_flag"] for o in runs], axis=0).tolist(),
                JE_halves=[float(np.mean([halves(B(o, "hq"))[0] for o in runs])), float(np.mean([halves(B(o, "hq"))[1] for o in runs]))])


def pm(m, h, nd=1):
    return f"${m:.{nd}f}\\pm{h:.{nd}f}$"


def lsfit(U, y):
    """y = phi_inf U - b, ordinary least squares; SEs from the residuals (nan with two points)."""
    U, y = np.asarray(U, float), np.asarray(y, float); A = np.vstack([U, -np.ones_like(U)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None); resid = y - A @ coef; dof = len(y) - 2
    cov = np.linalg.inv(A.T @ A) * (resid @ resid / dof) if dof > 0 else np.full((2, 2), np.nan)
    return dict(phi_inf=float(coef[0]), phi_inf_se=float(math.sqrt(cov[0, 0])) if dof > 0 else float("nan"), b=float(coef[1]),
                b_se=float(math.sqrt(cov[1, 1])) if dof > 0 else float("nan"), max_resid_jobs=float(np.max(np.abs(resid))), dof=dof)


# =====================================================================================================================
out("=" * 110); out("0. BENCHMARKS"); out("=" * 110)
BN = J("multi_bench_crn.json")["networks"]; BO = J("multi_bench.json")
NAMES = list(BO); L = {n: n[0] for n in NAMES}; TOY = NAMES[0]; G = NAMES[-1]
H = {n: BN[n]["HGI_E"] for n in NAMES}; HHW = {n: BN[n]["hw95"] for n in NAMES}
out(f"{'network':4s} {'HGI_E ± SE (8 seeds)':>26s} {'95% hw':>8s} {'exact':>6s}")
for n in NAMES:
    b = BN[n]
    out(f"{L[n]:4s} {b['HGI_E']:14.4f} ± {b['se']:.4f} {b['hw95']:12.4f} {('%.1f' % b['exact']) if b['exact'] is not None else '-':>6s}")
    put(f"HGI_E.{L[n]}", f"{b['HGI_E']:.2f} ± {b['se']:.2f}", "sim/results/multi_bench_crn.json", f"networks['{n}'].HGI_E, .se", [b["HGI_E"], b["se"]])
put("HGI_E.toy.hw95", f"{HHW[TOY]:.3f}", "sim/results/multi_bench_crn.json", f"networks['{TOY}'].hw95", HHW[TOY])
for n in NAMES:
    if BN[n]["exact"] is not None:
        put(f"HGI_E.{L[n]}.exact_check", f"exact {BN[n]['exact']:.0f}; estimate {BN[n]['HGI_E']:.3f} ± {BN[n]['se']:.3f} (z = {BN[n]['z_vs_exact']:+.2f})",
            "sim/results/multi_bench_crn.json", f"networks['{n}'].exact, .z_vs_exact")
        out(f"   exact check {L[n]}: " + NUM[f"HGI_E.{L[n]}.exact_check"]["text"])
SCALE = BO[TOY]["HGI_E"] / H[TOY]
put("scale.toy", f"{SCALE:.4f}", "sim/results/multi_bench_crn.json, multi_bench.json",
    "multi_bench.json['A toy (Ex. 2.2)'].HGI_E (independent extrapolation legs) / multi_bench_crn.json networks['A toy (Ex. 2.2)'].HGI_E", SCALE)
HD = J("hgi_d_toy.json")["1.0"]
out(f"HGI_D (toy, w0 = 0, varsigma = 1), stored (both legs on the same Brownian paths): {HD['HGI_D']:.4f} ± {HD['se']:.4f}")
put("HGI_D.stored", f"{HD['HGI_D']:.4f} ± {HD['se']:.4f}", "sim/results/hgi_d_toy.json", "['1.0'].HGI_D, .se", [HD["HGI_D"], HD["se"]])
if os.path.exists(os.path.join(RES, "hgi_d_toy_crn.json")):
    HC = J("hgi_d_toy_crn.json")
    for k, v in HC["runs"].items():
        out(f"   recomputed, seed {k}: {v['HGI_D']:.4f} ± {v['se']:.4f}  (legs {v['leg_dt_1e-3']:.4f}, {v['leg_dt_5e-4']:.4f}; SE of the leg difference {v['se_difference']:.1e})")
        put(f"HGI_D.recomputed.seed{k}", f"{v['HGI_D']:.4f} ± {v['se']:.4f}", "sim/results/hgi_d_toy_crn.json", f"runs['{k}'].HGI_D, .se", [v["HGI_D"], v["se"]])
    p = HC["pooled"]; out(f"   pooled over the two independent sets (20 000 paths): {p['HGI_D']:.4f} ± {p['se']:.4f}")
    put("HGI_D.pooled", f"{p['HGI_D']:.4f} ± {p['se']:.4f}", "sim/results/hgi_d_toy_crn.json", "pooled.HGI_D, .se", [p["HGI_D"], p["se"]])

# =====================================================================================================================
out(); out("=" * 110); out("1. TABLE 1 (toy at fixed constants); predicted column with d = 0.976896"); out("=" * 110)
T1D = J("table1_drift.json"); PROD = [x for x in J("production.json") if x["kind"] == "ergodic"]
assert abs(T1D["d"] - D_SERVICE) < 1e-6
TAB1 = []
out(f"{'c2':>3s} {'r':>5s} {'T':>4s} {'seeds':>6s} | {'J_E ± SE':>16s} | {'halves':>16s} | {'pred':>7s} {'flag only (local types) ± SE':>30s} {'0.5 r d E_frac':>15s} {'total idle ± SE':>17s}")
for row in T1D["rows"]:
    c1, c2, k = row["c"]; r = row["r"]
    g = [x for x in PROD if x["consts"] == row["consts"] and x["r"] == r]
    assert len(g) == len(row["seeds"]) and abs(np.mean([x["JE"] for x in g]) - row["JE"]) < 1e-12
    h1, h2 = float(np.mean([x["JE_1st"] for x in g])), float(np.mean([x["JE_2nd"] for x in g]))
    pred = 0.5 * r * D_SERVICE * g_flag(c1 * r ** k, c2 * r ** k, 1 - 0.5 / r, D_SERVICE)
    assert abs(pred - row["predicted_flag_drift"]) < 1e-3 * pred      # the stored column used d to full precision
    e = dict(c2=c2, r=r, T=row["T"], n_seeds=len(g), JE=row["JE"], JE_se=row["JE_se"], halves=[h1, h2], predicted=pred,
             flag_only=row["flag_drift_local"], flag_only_se=row["flag_drift_local_se"], flag_only_Efrac=row["flag_only_Efrac"],
             total_idle=row["total_idle_drift"], total_idle_se=row["total_idle_drift_se"], pred_over_meas=pred / row["flag_drift_local"])
    TAB1.append(e)
    out(f"{c2:3.0f} {r:5d} {row['T']:4.0f} {len(g):6d} | {row['JE']:8.2f} ± {row['JE_se']:5.2f} | {h1:7.1f} -> {h2:6.1f} | {pred:7.3f} {row['flag_drift_local']:20.3f} ± {row['flag_drift_local_se']:.3f} {row['flag_only_Efrac']:15.3f} {row['total_idle_drift']:9.3f} ± {row['total_idle_drift_se']:.3f}")
    big = row["JE"] > 100
    put(f"tab1.c2={c2:.0f}.r={r}", f"J_E {row['JE']:.{0 if big else 2}f} ± {row['JE_se']:.{0 if big else 2}f}; halves {h1:.{0 if big else 1}f} -> {h2:.{0 if big else 1}f}; flag drift {pred:.{1 if pred > 10 else 2}f} / {row['flag_drift_local']:.{1 if pred > 10 else 2}f}",
        "sim/results/table1_drift.json (rows reproduce production.json bit for bit); halves: production.json JE_1st, JE_2nd",
        f"rows[consts={row['consts']}, r={r}].JE, .JE_se, .predicted_flag_drift, .flag_drift_local", e)
out("LaTeX rows:")
LATEX1 = []
for e in TAB1:
    big = e["JE"] > 100
    je = f"${e['JE']:.0f}\\pm{e['JE_se']:.0f}$" if big else f"${e['JE']:.2f}\\pm{e['JE_se']:.{2 if e['JE'] < 10 else 1}f}$" if e["JE"] < 10 else f"${e['JE']:.1f}\\pm{e['JE_se']:.1f}$"
    hv = f"${e['halves'][0]:.0f}\\to{e['halves'][1]:.0f}$" if big else f"${e['halves'][0]:.1f}\\to{e['halves'][1]:.1f}$"
    fd = f"{e['predicted']:.1f} / {e['flag_only']:.1f}" if e["predicted"] > 10 else f"{e['predicted']:.2f} / {e['flag_only']:.2f}"
    LATEX1.append(f"{e['c2']:.0f} & {e['r']} & {e['T']:.0f} & {je} & {hv} & {fd} \\\\")
    out("  " + LATEX1[-1])
tw = [e for e in TAB1 if e["c2"] == 30 and e["r"] == 100][0]
put("text.twice_benchmark", f"J_E = {tw['JE']:.2f} is {tw['JE'] / H[TOY]:.2f} times HGI_E = {H[TOY]:.2f}", "sim/results/table1_drift.json, multi_bench_crn.json", "rows[tuned, r=100].JE / networks[A].HGI_E", tw["JE"] / H[TOY])
out(f"'about twice the benchmark' at r = 100: {tw['JE']:.2f} / {H[TOY]:.3f} = {tw['JE'] / H[TOY]:.2f}")
r400 = [e for e in TAB1 if e["c2"] == 30 and e["r"] == 400][0]
put("text.r400_times", f"{r400['JE'] / H[TOY]:.2f}", "sim/results/table1_drift.json, multi_bench_crn.json", "rows[tuned, r=400].JE / HGI_E", r400["JE"] / H[TOY])
for e in TAB1:
    put(f"text.flag_formula_error.c2={e['c2']:.0f}.r={e['r']}", f"{100 * abs(e['pred_over_meas'] - 1):.1f}%", "sim/results/table1_drift.json",
        "predicted_flag_drift / flag_drift_local - 1", e["pred_over_meas"])
out("flag formula, |predicted / measured - 1|: " + ", ".join(f"c2 = {e['c2']:.0f}, r = {e['r']}: {100 * abs(e['pred_over_meas'] - 1):.1f}%" for e in TAB1))
rs_i = np.arange(50, 30000)
dr30 = np.array([0.5 * r * D_SERVICE * g_flag(C1 * r ** KAPPA, 30 * r ** KAPPA, 1 - 0.5 / r, D_SERVICE) for r in rs_i])
rs_j = np.arange(100, 3000)
dr60 = np.array([0.5 * r * D_SERVICE * g_flag(C1 * r ** KAPPA, 60 * r ** KAPPA, 1 - 0.5 / r, D_SERVICE) for r in rs_j])
rng30 = (int(rs_i[dr30 > 1].min()), int(rs_i[dr30 > 1].max())); b01 = int(rs_j[dr60 >= 0.1].max()) + 1; b005 = int(rs_j[dr60 >= 0.05].max()) + 1
out(f"flag formula ranges (d = {D_SERVICE}): c2 = 30 above |theta| for r = {rng30[0]} .. {rng30[1]}; c2 = 60 below 0.1 from r = {b01}, below 0.05 from r = {b005}")
put("text.flag_range_c2=30", f"{rng30[0]} to {rng30[1]}", "formula (flag), sim/flagmodel.g_flag, d = 0.976896", "0.5 r d g(L, U) > 1", rng30)
put("text.flag_below_0.1_c2=60", f"r >= {b01}", "formula (flag), sim/flagmodel.g_flag, d = 0.976896", "0.5 r d g(L, U) < 0.1", b01)
put("text.flag_below_0.05_c2=60", f"r >= {b005}", "formula (flag), sim/flagmodel.g_flag, d = 0.976896", "0.5 r d g(L, U) < 0.05", b005)

# =====================================================================================================================
out(); out("=" * 110); out("2. TABLE 2 (toy, (0.1, 60, 0.24), seeds 1-2)"); out("=" * 110)
PC = [o for o in J("paper_cells.json") if o["net"] == TOY and o["cap"] == "uniform" and o["c"][1] == 60.0]
TAB2, LATEX2 = {}, []
out(f"{'r':>6s} {'T':>4s} {'nb':>3s} | {'J_E':>12s} | {'T1':>14s} | {'T2':>14s} | {'excess %':>12s} | {'T3 ± batch hw':>16s} {'(hw incl. benchmark)':>20s} | {'T1/(U/r)':>8s} {'T2/T1':>6s} | halves of T1+T2 (pooled), change")
for r in (300, 1000, 2000, 4000, 10000):
    c = cell([o for o in PC if o["r"] == r], H[TOY], HHW[TOY]); TAB2[r] = c
    out(f"{r:6d} {c['T']:4.0f} {c['n_batches']:3d} | {c['JE']:5.2f} ± {c['JE_hw']:4.2f} | {c['T1']:6.3f} ± {c['T1_hw']:5.3f} | {c['T2']:6.3f} ± {c['T2_hw']:5.3f} | {c['excess']:5.1f} ± {c['excess_hw']:4.1f} | {c['T3']:+6.2f} ± {c['T3_hw']:4.2f} {c['T3_hw_with_benchmark']:20.2f} | "
        f"{c['T1_over_band']:8.3f} {c['T2_over_T1']:6.2f} | {c['halves_pooled'][0]:.3f} -> {c['halves_pooled'][1]:.3f} ({c['halves_change_pct']:+.1f}%)")
    LATEX2.append(f"{r:<5d} & {pm(c['JE'], c['JE_hw'], 2)} & {c['T1']:.3f} & {pm(c['T2'], c['T2_hw'], 3)} & {pm(c['excess'], c['excess_hw'], 1)} & ${c['T3']:+.2f}\\pm{c['T3_hw']:.2f}$ & {c['T1_over_band']:.2f} \\\\")
    put(f"tab2.r={r}", f"{c['JE']:.2f}±{c['JE_hw']:.2f} | {c['T1']:.3f} | {c['T2']:.3f}±{c['T2_hw']:.3f} | {c['excess']:.1f}±{c['excess_hw']:.1f} | {c['T3']:+.2f}±{c['T3_hw']:.2f} | {c['T1_over_band']:.2f}",
        "sim/results/paper_cells.json; benchmark: multi_bench_crn.json", f"runs[net=A, cap=uniform, c2=60, r={r}].batches.hq/.hhat/.hg, pooled over seeds 1-2", c)
out("LaTeX rows (r & J_E & T1 & T2 & excess (%) & T3 & T1/(U/r)):")
for s in LATEX2:
    out("  " + s)
t3_in = [abs(TAB2[r]["T3"]) <= TAB2[r]["T3_hw"] for r in TAB2]; t3_in_b = [abs(TAB2[r]["T3"]) <= TAB2[r]["T3_hw_with_benchmark"] for r in TAB2]
out(f"'T3 is within its half-width of zero in every row': {sum(t3_in)} of 5 rows with the batch-means half-width; {sum(t3_in_b)} of 5 when the benchmark's own 95% half-width ({HHW[TOY]:.3f}) is added in quadrature")
out("   |T3|/hw per row: " + ", ".join(f"r = {r}: {abs(TAB2[r]['T3']) / TAB2[r]['T3_hw']:.2f}" for r in TAB2))
put("text.T3_within_hw", f"{sum(t3_in)} of 5 rows (max |T3|/hw = {max(abs(TAB2[r]['T3']) / TAB2[r]['T3_hw'] for r in TAB2):.2f})", "sim/results/paper_cells.json, multi_bench_crn.json",
    "|mean(hg) - HGI_E| <= batch-means hw of hg", [bool(x) for x in t3_in])
put("text.excess_300", f"{TAB2[300]['excess']:.0f}%", "sim/results/paper_cells.json, multi_bench_crn.json", "tab2 r=300 excess", TAB2[300]["excess"])
put("text.excess_1e4", f"{TAB2[10000]['excess']:.1f}%", "sim/results/paper_cells.json, multi_bench_crn.json", "tab2 r=10000 excess", TAB2[10000]["excess"])
put("text.T1_over_band_range", f"{min(c['T1_over_band'] for c in TAB2.values()):.2f} to {max(c['T1_over_band'] for c in TAB2.values()):.2f}", "sim/results/paper_cells.json", "T1 r / U", None)
put("text.T2_over_T1_range", f"{min(c['T2_over_T1'] for c in TAB2.values()):.2f} to {max(c['T2_over_T1'] for c in TAB2.values()):.2f}", "sim/results/paper_cells.json", "T2 / T1", None)
hc = [TAB2[r]["halves_change_pct"] for r in TAB2]
put("text.tab2_halves_range", f"{min(hc):+.1f}% to {max(hc):+.1f}%", "sim/results/paper_cells.json", "pooled halves of batches.hq - batches.hg, (second - first)/first", hc)
out(f"pooled half-to-half change of T1+T2, two seeds of the table: {min(hc):+.1f}% to {max(hc):+.1f}%")

# --- eight seeds at r = 1000, 2000 (drift_check.json part b + the two seeds of the table)
DC = J("drift_check.json")["runs"]
for r in (1000, 2000):
    runs8 = sorted([o for o in PC if o["r"] == r] + [o for o in DC if o.get("part") == "b" and o["r"] == r], key=lambda o: o["seed"])
    assert [o["seed"] for o in runs8] == list(range(1, 9))
    v = np.array([float((B(o, "hq") - B(o, "hg")).mean()) for o in runs8]); ch = np.array([100 * (halves(B(o, "hq") - B(o, "hg"))[1] - halves(B(o, "hq") - B(o, "hg"))[0]) / float((B(o, "hq") - B(o, "hg")).mean()) for o in runs8])
    out(f"8 seeds, r = {r}: excess {100 * v.mean() / H[TOY]:.2f} ± {100 * hw(v) / H[TOY]:.2f} % (95%, across seeds); half-to-half change {ch.mean():+.2f}% ± {hw(ch):.2f}%")
    put(f"text.8seed_excess.r={r}", f"{100 * v.mean() / H[TOY]:.1f} ± {100 * hw(v) / H[TOY]:.1f}", "sim/results/paper_cells.json (seeds 1-2), drift_check.json (seeds 3-8); multi_bench_crn.json",
        f"mean over 8 seeds of mean(batches.hq - batches.hg) / HGI_E, t-based 95% hw, r={r}", [float(v.mean()), hw(v)])
    put(f"text.8seed_change.r={r}", f"{ch.mean():+.1f}% ± {hw(ch):.1f}%", "sim/results/paper_cells.json, drift_check.json", "(second half - first half)/mean of T1+T2 per seed", [float(ch.mean()), hw(ch)])
pb = J("drift_check.json")["analysis"]["b_pooled"]
put("text.8seed_change.pooled", f"+{pb['mean']:.1f}% ± {pb['hw']:.1f}%", "sim/results/drift_check.json", "analysis.b_pooled.mean, .hw", pb)
out(f"pooled over the 16 runs: {pb['mean']:+.2f}% ± {pb['hw']:.2f}% (95%)   [benchmark-free]")

# --- the U/r coefficient and the 1% extrapolation
rs2 = np.array(sorted(TAB2), float); t12 = np.array([TAB2[int(r)]["T12"] for r in rs2]); t12h = np.array([TAB2[int(r)]["T12_hw"] for r in rs2]); Uor = np.array([TAB2[int(r)]["U"] / r for r in rs2])
geo = lambda m: float(np.exp(np.mean(np.log(t12[m] / Uor[m]))))
coef_all, coef_1000 = geo(rs2 > 0), geo(rs2 > 300)
out("(T1+T2)/(U/r) per r: " + ", ".join(f"{int(r)}: {a / u:.3f} ± {h / u:.3f}" for r, a, h, u in zip(rs2, t12, t12h, Uor)))
out(f"coefficient of U/r (geometric mean of the ratios): {coef_all:.3f} with r = 300, {coef_1000:.3f} for r >= 1000   [benchmark-free]")
put("text.U_over_r_coefficient", f"{coef_1000:.2f} (r >= 1000); {coef_all:.2f} with r = 300", "sim/results/paper_cells.json", "geometric mean over r of mean(batches.hq - batches.hg) r / U", [coef_1000, coef_all])
r1a = (coef_1000 * 60.0 / (0.01 * H[TOY])) ** (1 / (1 - KAPPA)); r1b = 1e4 * (TAB2[10000]["excess"] / 1.0) ** (1 / (1 - KAPPA))
out(f"'an excess of 1% needs r ~ 1e5': from the line {coef_1000:.2f} U/r = 0.01 HGI_E: r = {r1a:.3g}; from the r = 1e4 cell scaled by r^(kappa-1): r = {r1b:.3g}")
put("text.r_for_1pct", f"{r1a:.2g} (line) / {r1b:.2g} (from r = 1e4)", "sim/results/paper_cells.json, multi_bench_crn.json", "solve coef 60 r^(kappa-1) = 0.01 HGI_E", [r1a, r1b])

# =====================================================================================================================
out(); out("=" * 110); out("3. TABLE 3 (seven networks, T = 250, batches of 25, seeds 1-2, both caps)"); out("=" * 110)
DEC = J("decomp_runs.json"); VS = J("vbcap_summary.json")
if os.path.exists(os.path.join(RES, "table3_r4000.json")):
    R4 = J("table3_r4000.json")["runs"]; r4src = "sim/results/table3_r4000.json"
else:
    R4 = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "table3_r4000", "*.json")))]; r4src = "sim/results/table3_r4000/<cell>.json"
out(f"r = 4000 source: {r4src}; {len(R4)} of 26 runs available")
PP = {"A": 13.0, "B": None, "C": 13.0, "D": VS["D unequal rates"]["c2_used"], "E": VS["E unequal loads, costs"]["c2_used"], "F": VS["F 3x5 network"]["c2_used"], "G": 30.0}
TAB3 = {}


def t3cell(n, cap, c2, r):
    src = DEC if r in (1000, 2000) else R4
    runs = [o for o in src if o["net"] == n and o["cap"] == cap and abs(o["c"][1] - c2) < 1e-6 and o["r"] == r and o["T"] == 250.0]
    if len(runs) != 2:
        return None
    return cell(runs, H[n], HHW[n])


for n in NAMES:
    for cap, c2 in (("uniform", BO[n]["c2"]), ("per_pattern", PP[L[n]])):
        if c2 is None:
            continue
        for r in (1000, 2000, 4000):
            TAB3[(L[n], cap, r)] = t3cell(n, cap, c2, r)
missing = [k for k, v in TAB3.items() if v is None]
if missing:
    out(f"!! {len(missing)} cells not available yet: {missing}")
out(f"{'net':3s} {'cap':11s} {'c2':>7s} {'r':>5s} | {'excess %':>14s} | {'T1':>8s} {'T2':>8s} {'T2/T1':>6s} {'T1/(U/r)':>8s} | {'T3 ± hw':>15s} | {'E_frac':>8s} | halves of T1+T2: pooled change, per seed")
for (l, cap, r), c in TAB3.items():
    if c is None:
        continue
    out(f"{l:3s} {cap:11s} {c['c2']:7.2f} {r:5d} | {c['excess']:6.1f} ± {c['excess_hw']:5.1f} | {c['T1']:8.4f} {c['T2']:8.4f} {c['T2_over_T1']:6.2f} {c['T1_over_band']:8.3f} | {c['T3']:+6.2f} ± {c['T3_hw']:5.2f} | {c['E_frac']:8.1e} | "
        f"{c['halves_change_pct']:+5.1f}%  " + ", ".join(f"s{p['seed']}: {p['first']:.3f} -> {p['second']:.3f} ({p['change_pct']:+.1f}%)" for p in c["halves_per_seed"]))
    put(f"tab3.{l}.{cap}.r={r}", f"{c['excess']:.1f} ± {c['excess_hw']:.1f}", ("sim/results/decomp_runs.json" if r < 4000 else "sim/results/table3_r4000.json") + "; benchmark: multi_bench_crn.json",
        f"runs[net={l}, cap={cap}, c2={c['c2']:.4g}, r={r}, T=250].batches.hq - .hg, pooled over seeds 1-2, / HGI_E", c)


def e3(l, cap, r, nd=None):
    c = TAB3.get((l, cap, r))
    if c is None:
        return "n/a"
    nd = (0 if c["excess"] >= 100 else 1) if nd is None else nd
    return pm(c["excess"], c["excess_hw"], nd)


LATEX3 = []
for n in NAMES:
    l = L[n]; bm = BO[n]["b_max"]; c2u = BO[n]["c2"]
    left = " & ".join(e3(l, "uniform", r) for r in (1000, 2000, 4000))
    right = "same & same & same" if l == "B" else " & ".join(e3(l, "per_pattern", r) for r in (1000, 2000, 4000))
    c2p = "13" if l == "B" else ("$30^\\ast$" if l == "G" else f"{PP[l]:.0f}")
    LATEX3.append(f"{l} & {bm:.4f} & {H[n]:.2f} & {c2u:.0f}{'$^\\dagger$' if l == 'G' else ''} & {left} & {c2p} & {right} \\\\".replace("0.0408", "0.041").replace("0.2500", "0.250").replace("0.0801", "0.080").replace("0.0433", "0.043").replace("0.0359", "0.036"))
out("LaTeX rows (network & b_max & HGI_E & c2 & r=1e3 & 2e3 & 4e3 & c2 & r=1e3 & 2e3 & 4e3), excess in % with 95% batch-means half-widths:")
for s in LATEX3:
    out("  " + s)
out("per-doubling factors of the excess (r: 1000 -> 2000, 2000 -> 4000), and first -> second half of T1+T2 at r = 4000:")
FACT = {}
for (l, cap) in sorted({(k[0], k[1]) for k in TAB3}):
    c1_, c2_, c4_ = (TAB3.get((l, cap, r)) for r in (1000, 2000, 4000))
    if c1_ is None or c2_ is None:
        continue
    f12 = c2_["T12"] / c1_["T12"]; f24 = c4_["T12"] / c2_["T12"] if c4_ else float("nan"); FACT[(l, cap)] = (f12, f24)
    h4 = (f"pooled {c4_['halves_pooled'][0]:.4f} -> {c4_['halves_pooled'][1]:.4f} ({c4_['halves_change_pct']:+.1f}%); " + ", ".join(f"seed {p['seed']}: {p['change_pct']:+.1f}%" for p in c4_["halves_per_seed"])) if c4_ else "n/a"
    out(f"  {l} {cap:11s}: x{f12:.3f}, x{f24:.3f} | r = 4000 halves: {h4}")
    put(f"tab3.factor.{l}.{cap}", f"x{f12:.2f} (1000 to 2000), x{f24:.2f} (2000 to 4000)", "sim/results/decomp_runs.json, table3_r4000.json", "ratio of pooled mean(batches.hq - batches.hg)", [f12, f24])
    if c4_:
        put(f"tab3.halves_r4000.{l}.{cap}", f"{c4_['halves_change_pct']:+.1f}% pooled; per seed " + ", ".join(f"{p['change_pct']:+.1f}%" for p in c4_["halves_per_seed"]), "sim/results/table3_r4000.json", "halves of batches.hq - batches.hg per seed (7 post-burn batches; the middle batch is split), (second - first)/first", c4_["halves_per_seed"])
fu = [v for (l, cap), v in FACT.items() if cap == "uniform"]
allf = [x for v in fu for x in v if not math.isnan(x)]
if allf:
    out(f"Definition 2.3 cells: per-doubling factor between {min(allf):.3f} and {max(allf):.3f} (2^(kappa-1) = {2 ** (KAPPA - 1):.3f})")
    put("text.per_doubling_factor", f"{min(allf):.2f} to {max(allf):.2f}", "sim/results/decomp_runs.json, table3_r4000.json", "all Definition 2.3 cells, both doublings", allf)
for rr in ((1000, 2000), (4000,)):
    ch = {(l, r): TAB3[(l, "uniform", r)]["halves_change_pct"] for l in "ABCDEFG" for r in rr if TAB3.get((l, "uniform", r))}
    chs = {(l, r, p["seed"]): p["change_pct"] for l in "ABCDEFG" for r in rr if TAB3.get((l, "uniform", r)) for p in TAB3[(l, "uniform", r)]["halves_per_seed"]}
    if ch:
        kmax = max(ch, key=lambda k: abs(ch[k])); ksmax = max(chs, key=lambda k: abs(chs[k]))
        out(f"Definition 2.3, r in {rr}: pooled half-to-half change of the excess between {min(ch.values()):+.1f}% and {max(ch.values()):+.1f}% (largest: {kmax[0]} at r = {kmax[1]}); single seeds up to {chs[ksmax]:+.1f}% ({ksmax[0]}, r = {ksmax[1]}, seed {ksmax[2]})")
        put(f"text.halves_all_networks.r={'_'.join(map(str, rr))}", f"pooled {min(ch.values()):+.1f}% to {max(ch.values()):+.1f}%; single seeds up to {chs[ksmax]:+.1f}% ({ksmax[0]}, r = {ksmax[1]})",
            "sim/results/decomp_runs.json" if rr[0] == 1000 else "sim/results/table3_r4000.json",
            "pooled and per-seed halves of batches.hq - batches.hg, Definition 2.3 cells", dict(pooled={f"{k[0]}{k[1]}": v for k, v in ch.items()}, per_seed={f"{k[0]}{k[1]}s{k[2]}": v for k, v in chs.items()}))
t3u = [(k, c) for k, c in TAB3.items() if c is not None]
n_in = sum(abs(c["T3"]) <= c["T3_hw"] for _, c in t3u); n_in_b = sum(abs(c["T3"]) <= c["T3_hw_with_benchmark"] for _, c in t3u)
worst = max(t3u, key=lambda kc: abs(kc[1]["T3"]) / kc[1]["T3_hw"])
out(f"zero-check T3 over the {len(t3u)} cells of Table 3: {n_in} within the batch-means half-width, {n_in_b} with the benchmark's half-width added; worst {worst[0]}: {worst[1]['T3']:+.2f} ± {worst[1]['T3_hw']:.2f}")
put("text.T3_table3", f"{n_in} of {len(t3u)} cells within the batch-means half-width ({n_in_b} with the benchmark's half-width added)", "sim/results/decomp_runs.json, table3_r4000.json, multi_bench_crn.json",
    "|mean(hg) - HGI_E| <= hw", dict(n_in=n_in, n=len(t3u), n_in_with_benchmark=n_in_b))
rat = {l: [TAB3[(l, "uniform", r)]["T2_over_T1"] for r in (1000, 2000, 4000) if TAB3.get((l, "uniform", r))] for l in "ACDEFG"}
acf = [x for l in "ACDEF" for x in rat[l]]
out(f"T2/T1, Definition 2.3 cells: A and C-F {min(acf):.2f} to {max(acf):.2f}; G " + ", ".join(f"{x:.1f}" for x in rat["G"]) + "   [benchmark-free]")
ratp = [c["T2_over_T1"] for (l, cap, r), c in TAB3.items() if c is not None and cap == "per_pattern" and l in "ACDEF"]
if ratp:
    out(f"T2/T1, pattern-dependent cap cells (outside Definition 2.3), A and C-F: {min(ratp):.2f} to {max(ratp):.2f}")
    put("text.T2_over_T1_networks_cap", f"{min(ratp):.2f} to {max(ratp):.2f}", "sim/results/decomp_runs.json, table3_r4000.json", "T2/T1 per pattern-dependent-cap cell, A and C-F", ratp)
put("text.T2_over_T1_networks", f"{min(acf):.2f} to {max(acf):.2f} on A and C-F; " + " / ".join(f"{x:.1f}" for x in rat["G"]) + " on G", "sim/results/decomp_runs.json, table3_r4000.json", "T2/T1 per Definition 2.3 cell, r = 1000, 2000, 4000", rat)

# --- in-text numbers that come from Table 3's runs
def txt(id_, l, cap, r, fmt="{:.0f}%"):
    c = TAB3.get((l, cap, r))
    if c is not None:
        put(id_, fmt.format(c["excess"]), ("sim/results/decomp_runs.json" if r < 4000 else "sim/results/table3_r4000.json") + ", multi_bench_crn.json", f"tab3.{l}.{cap}.r={r}", c["excess"])
        out(f"   {id_}: {NUM[id_]['text']}")


out("in-text numbers taken from Table 3's cells:")
txt("text.toy_rule_band_1000", "A", "uniform", 1000); txt("text.toy_cap_band_1000", "A", "per_pattern", 1000, "{:.1f}%")
txt("text.G_660_1000", "G", "uniform", 1000); txt("text.G_660_4000", "G", "uniform", 4000)
txt("text.G_cap30_1000", "G", "per_pattern", 1000); txt("text.G_cap30_2000", "G", "per_pattern", 2000); txt("text.G_cap30_4000", "G", "per_pattern", 4000, "{:.1f}%")
g660pp = [o for o in DEC if o["net"] == G and o["cap"] == "per_pattern" and o["c"][1] == 660.0 and o["r"] == 1000]
c = cell(g660pp, H[G], HHW[G]); out(f"   G, pattern-dependent cap at the SAME band 660, r = 1000 (T = 250): {c['excess']:.0f} ± {c['excess_hw']:.0f} %")
put("text.G_cap660_1000", f"{c['excess']:.0f}%", "sim/results/decomp_runs.json, multi_bench_crn.json", "runs[net=G, cap=per_pattern, c2=660, r=1000, T=250]", c["excess"])
SC = [o for o in J("vbcap_runs.json") if o["net"] == G and o.get("tag") == "per_pattern_scan" and o["r"] == 1000 and o["seed"] == 1]
scan = {o["c"][1]: 100 * (o["JE"] - o["hgi_path"]) / H[G] for o in SC}
out("   G scan at r = 1000 (pattern-dependent cap, T = 40, seed 1), excess %: " + ", ".join(f"c2 = {k:.0f}: {v:.0f}" for k, v in sorted(scan.items(), reverse=True)))
put("text.G_scan_400", f"{scan[400.0]:.0f}%", "sim/results/vbcap_runs.json, multi_bench_crn.json", "runs[net=G, tag=per_pattern_scan, c2=400, r=1000, seed=1]: (JE - hgi_path)/HGI_E", scan[400.0])
put("text.G_scan_50", f"{scan[50.0]:.0f}%", "sim/results/vbcap_runs.json, multi_bench_crn.json", "runs[net=G, tag=per_pattern_scan, c2=50, r=1000, seed=1]: (JE - hgi_path)/HGI_E", scan[50.0])
DM = J("decomp_matched.json"); st = [100 * (x["T1_p"] / x["T1_u"] - 1) for x in DM if x.get("T1_u") and x["T1_u"] < 5 and x.get("T2_u", 0) < 100]
put("text.matched_T1", f"{min(st):.0f}% to {max(st):.0f}%", "sim/results/decomp_matched.json", "T1_p / T1_u - 1 over the stationary matched-band pairs", st)
out(f"   matched-band T1 increase under the pattern-dependent cap (stationary pairs): {min(st):.1f}% to {max(st):.1f}%   [benchmark-free]")

# --- G: measured flag drift at the band 660
GS = J("network_g_check.json"); Kg = np.array(GS["spec"]["K"], float); thg = np.abs(Kg @ np.array(GS["spec"]["eta"], float))        # theta = K eta
gfd = {}
for r in (1000, 2000, 4000):
    c = TAB3.get(("G", "uniform", r))
    if c is not None:
        gfd[r] = (r * Kg @ np.array(c["p_flag"]) / thg).tolist()
pred = {row["r"]: row["pred_floor"] for row in GS["rows"]}
out("   G at c2 = 660, flag drift relative to |theta_i| (upper bound r sum_j K_ij P(E_j = 1)/|theta_i|): " + "; ".join(f"r = {r}: measured {np.round(v, 2).tolist()}, predicted {pred.get(r, float('nan')):.2f}" for r, v in gfd.items()))
if gfd:
    put("text.G_flag_drift_measured_max", f"{max(max(v) for v in gfd.values()):.2f}", "sim/results/decomp_runs.json, table3_r4000.json (p_flag); network_g_check.json (K, theta)",
        "max over r and i of r sum_j K_ij p_flag_j / |theta_i|", gfd)
put("text.G_flag_drift_predicted", f"{min(pred.values()):.2f} to {max(pred.values()):.2f}", "sim/results/network_g_check.json", "rows[].pred_floor", pred)

# =====================================================================================================================
out(); out("=" * 110); out("4. phi: T1 = (phi_inf U - b)/r, with the r = 4000 points of the T = 250 runs   [benchmark-free]"); out("=" * 110)
PHI = {}
runs60 = [o for o in PC] + [o for o in DC if o.get("part") == "b"]
pts = {}
for o in runs60:
    pts.setdefault((o["r"], o["U"]), []).append(float((B(o, "hq") - B(o, "hhat")).mean()))
ks = sorted(pts); f = lsfit([k[1] for k in ks], [np.mean(pts[k]) * k[0] for k in ks]); f["phi"] = {k[0]: float(np.mean(pts[k]) * k[0] / k[1]) for k in ks}; PHI["A, c2 = 60"] = f
mu = {r: math.log((1 - 0.5 / r) / D_SERVICE) for r in (300, 1000, 10000)}
for l in "ACDEFG":
    cs = [(r, TAB3.get((l, "uniform", r))) for r in (1000, 2000, 4000)]
    cs = [(r, c) for r, c in cs if c is not None]
    if len(cs) >= 2:
        f = lsfit([c["U"] for _, c in cs], [c["T1"] * r for r, c in cs]); f["phi"] = {r: c["T1_over_band"] for r, c in cs}; f["phi_hw"] = {r: c["T1_hw"] * r / c["U"] for r, c in cs}; PHI[f"{l}, c2 = {cs[0][1]['c2']:.0f}"] = f
    cs = [(r, TAB3.get((l, "per_pattern", r))) for r in (1000, 2000, 4000)]
    cs = [(r, c) for r, c in cs if c is not None]
    if len(cs) >= 2:
        f = lsfit([c["U"] for _, c in cs], [c["T1"] * r for r, c in cs]); f["phi"] = {r: c["T1_over_band"] for r, c in cs}; f["phi_hw"] = {r: c["T1_hw"] * r / c["U"] for r, c in cs}; PHI[f"{l}, pattern-dependent cap (outside Def. 2.3), c2 = {cs[0][1]['c2']:.0f}"] = f
for k, f in PHI.items():
    out(f"  {k:58s} phi(r): " + ", ".join(f"{r}: {v:.3f}" + (f" ± {f['phi_hw'][r]:.3f}" if "phi_hw" in f else "") for r, v in f["phi"].items()) + f"  ->  phi_inf = {f['phi_inf']:.3f} ± {f['phi_inf_se']:.3f}, b = {f['b']:.1f} ± {f['b_se']:.1f} jobs (max residual {f['max_resid_jobs']:.2f} jobs, {f['dof']} dof)")
    put(f"phi.{k}", "phi(r) = " + " / ".join(f"{v:.3f}" + (f" ± {f['phi_hw'][r]:.3f}" if "phi_hw" in f else "") for r, v in f["phi"].items()) + f" at r = {' / '.join(str(r) for r in f['phi'])}; fit phi_inf = {f['phi_inf']:.3f} ± {f['phi_inf_se']:.3f}, b = {f['b']:.1f} ± {f['b_se']:.1f} jobs", "sim/results/paper_cells.json, drift_check.json" if "60" in k and k.startswith("A, c2 = 60") else "sim/results/decomp_runs.json, table3_r4000.json",
        "least squares of mean(batches.hq - batches.hhat) r on U over the available r", f)
out("  (± on phi(r) = 95% batch-means half-width of T1 r/U; ± on the fit = SE from the residuals, 1 dof with three points)")
out(f"  toy mechanism: 1/mu = 1/log(a/d) = {1 / mu[300]:.1f} (r = 300), {1 / mu[1000]:.1f} (r = 1000), {1 / mu[10000]:.1f} (r = 1e4) jobs")
put("text.one_over_mu", f"{1 / mu[300]:.1f} to {1 / mu[10000]:.1f} jobs", "formula, d = 0.976896", "1/log((1 - 0.5/r)/d), r = 300 .. 1e4", [1 / mu[300], 1 / mu[10000]])

# =====================================================================================================================
out(); out("=" * 110); out("5. OTHER IN-TEXT NUMBERS"); out("=" * 110)
CE = J("control_experiment.json")["summary"]
for c2 in (30.0, 60.0, 10.0):
    for vb in (0.25, None):
        rows = sorted([x for x in CE if x["vb"] == vb and x["c2"] == c2], key=lambda x: x["r"])
        s = " / ".join(f"{x['JE']:.2f} ± {x['JE_se']:.2f}" for x in rows)
        out(f"  control, flags off, |v^b| = {'0.25' if vb else 'b_max'}, c2 = {c2:.0f}: J_E = {s} at r = 30 / 100 / 300 / 1000; T1+T2 = " + " / ".join(f"{x['T12']:.3f}" for x in rows))
        put(f"control.vb={'0.25' if vb else 'bmax'}.c2={c2:.0f}", s, "sim/results/control_experiment.json", f"summary[vb={vb}, c2={c2}].JE, .JE_se (r = 30, 100, 300, 1000)", [x["JE"] for x in rows])
        if vb:
            fr = [rows[i + 1]["T12"] / rows[i]["T12"] for i in range(3)]
            put(f"control.T12_factor.c2={c2:.0f}", " / ".join(f"{x:.2f}" for x in fr), "sim/results/control_experiment.json", "ratio of summary.T12 between consecutive r", fr)
allfr = [float(x) for k, v in NUM.items() if k.startswith("control.T12_factor") for x in v["value"]]
out(f"  control: T1+T2 falls by a factor {min(allfr):.2f} to {max(allfr):.2f} at each step of r (|v^b| = 0.25, three bands)")
put("text.control_factor", f"{min(allfr):.2f} to {max(allfr):.2f}", "sim/results/control_experiment.json", "ratios of summary.T12", allfr)
SR = J("screen_r300.json")
scr = {tuple(x["c"]): x for x in SR["summary"]}; worst = max(scr.values(), key=lambda x: x["JE"])
out("  screen at r = 300 (T = 40, 8 seeds), J_E ± SE: " + "; ".join(f"{tuple(x['c'])}: {x['JE']:.2f} ± {x['JE_se']:.2f}" for x in SR["summary"]))
out(f"     selected (0.1, 30, 0.24): {scr[(0.1, 30.0, 0.24)]['JE']:.1f}; worst {tuple(worst['c'])}: {worst['JE']:.0f}; (1, 30, 0.24): {scr[(1.0, 30.0, 0.24)]['JE']:.1f}; paired difference {SR['tie']['diff']:.2f} ± {SR['tie']['hw95']:.2f} (95%)")
put("text.screen", f"{scr[(0.1, 30.0, 0.24)]['JE']:.1f} against {worst['JE']:.0f} for the worst cell; (1, 30, 0.24): {scr[(1.0, 30.0, 0.24)]['JE']:.1f}; paired difference {SR['tie']['diff']:.2f} ± {SR['tie']['hw95']:.2f}",
    "sim/results/screen_r300.json", "summary[c].JE; tie.diff, tie.hw95", None)
SP = J("split_r100.json")
zmax = max(abs(v["JE"]) for v in SP["pairwise_z"].values())
out("  eta-split at r = 100 (c2 = 60, T = 150, 8 seeds per split, different seeds): J_E = " + ", ".join(f"{k}: {v['JE'][0]:.2f} ± {v['JE'][1]:.2f}" for k, v in SP["stats"].items()) + f"; largest pairwise |z| on J_E {zmax:.2f}")
put("text.split_r100", ", ".join(f"{v['JE'][0]:.2f}" for v in SP["stats"].values()) + f" (SE {min(v['JE'][1] for v in SP['stats'].values()):.2f} to {max(v['JE'][1] for v in SP['stats'].values()):.2f}); largest pairwise difference {zmax:.1f} SE",
    "sim/results/split_r100.json", "stats[split].JE = [mean, SE, hw]; pairwise_z[pair].JE", None)
A4 = J("drift_check.json")["analysis"]
out(f"  c2 = 30, r = 400, T = 320, 8 seeds: J_E = {A4['a_JE']['mean']:.1f} ± {A4['a_JE']['hw']:.1f} (95%), i.e. {A4['a_JE']['mean'] / H[TOY]:.1f} times HGI_E; measured flag drift {A4['a_flag_only']['mean']:.2f}")
put("text.r400_T320", f"{A4['a_JE']['mean']:.1f} ± {A4['a_JE']['hw']:.1f}", "sim/results/drift_check.json", "analysis.a_JE.mean, .hw", A4["a_JE"])
put("text.r400_flag_drift", f"{A4['a_flag_only']['mean']:.2f}", "sim/results/drift_check.json", "analysis.a_flag_only.mean", A4["a_flag_only"])
JD = J("jd_seeds.json"); jd = {}
for o in JD:
    jd.setdefault(o["r"], []).append(o["JD"])
out("  J_D deep band (jd_seeds.json): " + ", ".join(f"r = {r}: {np.mean(v):.2f} ± {se(v):.2f} ({len(v)} seeds)" for r, v in sorted(jd.items())) + f"  against HGI_D = {HD['HGI_D']:.3f}")
for r, v in sorted(jd.items()):
    put(f"JD.deep.r={r}", f"{np.mean(v):.2f} ± {se(v):.2f} ({len(v)} seeds)", "sim/results/jd_seeds.json", f"mean and SE of JD over seeds, r={r}", [float(np.mean(v)), se(v)])
jdt = {}
for x in J("production.json"):
    if x["kind"] == "discounted" and x["consts"] == "tuned":
        jdt.setdefault(x["r"], []).append(x["JD"])
out("  J_D tuned set (production.json): " + ", ".join(f"r = {r}: {np.mean(v):.2f} ± {se(v):.2f} ({len(v)} seeds)" for r, v in sorted(jdt.items())))
for r, v in sorted(jdt.items()):
    put(f"JD.tuned.r={r}", f"{np.mean(v):.2f} ± {se(v):.2f} ({len(v)} seeds)", "sim/results/production.json", f"kind=discounted, consts=tuned, r={r}: mean and SE of JD", [float(np.mean(v)), se(v)])

# =====================================================================================================================
json.dump(dict(note="Every number printed by analysis/make_paper_numbers.py: formatted text, source file, key in that file and value.",
               benchmark=dict(HGI_E=H, hw95=HHW), numbers=NUM, latex=dict(table1=LATEX1, table2=LATEX2, table3=LATEX3),
               table3_r4000_runs_available=len(R4)),
          open(os.path.join(OUT, "paper_numbers.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
open(os.path.join(OUT, "paper_numbers.txt"), "w").write("\n".join(LINES) + "\n")
print(f"\nwrote {OUT}/paper_numbers.json ({len(NUM)} numbers) and {OUT}/paper_numbers.txt")
