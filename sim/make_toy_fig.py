"""Figure 1 of the report (toy_convergence_v5): the toy network under Definition 2.3 at three hysteresis bands.

(a) J_E against r for (c1, c2, kappa) = (0.5, 10, 0.22), (0.1, 30, 0.24), (0.1, 60, 0.24), mean over seeds with the
    standard error across seeds; markers are hollow where the flag formula predicts a drift above |theta|. The dashed
    series is c2 = 30 with the flags switched off (OUTSIDE Definition 2.3).
(b) Flag drift 0.5 r d g(L, U): formula (lines) and simulation (dots, 0.5 r d sum_j p_flag_j or 0.5 r d E_frac),
    d = 0.976896.
(c) c2 = 60: excess (T1 + T2)/HGI_E and its part T1/HGI_E in percent, with the line fitted for r >= 1000.
Inputs (stored results only): results/production.json (c2 = 10, 30), results/paper_cells.json (c2 = 60, r <= 1e4:
fields JE, p_flag, batches), results/large_r/*.json (c2 = 60, r = 2e4, 4e4), results/control_experiment.json (flags
off), results/multi_bench_crn.json (benchmark).
Every plotted number that the report prints is checked first; the script stops without drawing if one does not round
to the printed value.
Writes toy_convergence_v5.png (300 dpi) and .pdf to results/final/, or to DIR with --out DIR.
Run from sim/:  python make_toy_fig.py"""
import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flagmodel import g_flag
from common import mean_hw

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#ffffff"
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "results/final"          # output folder
_B = json.load(open("results/multi_bench_crn.json"))["networks"]["A toy (Ex. 2.2)"]
HGI, HGI_SE, D = _B["HGI_E"], _B["se"], 0.976896
HGI_E_TOY = HGI
SETS = {"tuned": ((0.1, 30, 0.24), BLUE, r"$c_2=30$ $(0.1,30,0.24)$"),
        "reference": ((0.5, 10, 0.22), ORANGE, r"$c_2=10$ $(0.5,10,0.22)$"),
        "deep": ((0.1, 60, 0.24), AQUA, r"$c_2=60$ $(0.1,60,0.24)$")}

# ------------------------------------------------------------------------------------------ data and checks before drawing
from scipy.stats import t as _tdist
_STOP = []
# control: flags off at the definition's own b_max, (c2, kappa) = (30, 0.24)  (OUTSIDE Definition 2.3)
CONTROL_EXPECTED = {30: ("6.03", "0.06"), 100: ("4.47", "0.09"), 300: ("3.79", "0.33"), 1000: ("3.50", "0.18")}
CTRL = sorted([x for x in json.load(open("results/control_experiment.json"))["summary"] if x["vb"] is None and x["c2"] == 30.0 and x["kappa"] == 0.24],
              key=lambda x: x["r"])
assert [x["r"] for x in CTRL] == [30, 100, 300, 1000] and all(x["n"] == 4 for x in CTRL)
print("control (flags off), (c2, kappa) = (30, 0.24), push 0.98 b_max, 4 seeds:")
for x in CTRL:
    g = CONTROL_EXPECTED[x["r"]]; ok = (f"{x['JE']:.2f}", f"{x['JE_se']:.2f}") == g
    print(f"  r = {x['r']:5d}: stored J_E = {x['JE']:.4f} +- {x['JE_se']:.4f} (SE)   report {g[0]} +- {g[1]}   {'ok' if ok else 'DOES NOT ROUND TO THE PRINTED NUMBER'}")
    if not ok:
        _STOP.append(f"control at r = {x['r']}: stored {x['JE']:.4f} +- {x['JE_se']:.4f}, report {g[0]} +- {g[1]}")
# the runs at r = 20 000 and 40 000 (longleaf/run_toy.py)
LARGE_DIR = "results/large_r"
LARGE_EXPECTED = {20000: dict(JE="4.00", T1="0.030285", T12="0.090043", excess="3.26", T1_pct="1.096", JE_hw_seeds="2.60", excess_hw_seeds="0.05",
                              JE_hw_batch="1.29", excess_hw_batch="0.02", JE_se="0.82", predicted="9.68e-04"),
                  40000: dict(JE="2.93", T1="0.018068", T12="0.053652", excess="1.94", T1_pct="0.654", JE_hw_seeds="1.00", excess_hw_seeds="0.03",
                              JE_hw_batch="0.70", excess_hw_batch="0.01", JE_se="0.32", predicted="1.48e-04")}
_hw = lambda v: float(_tdist.ppf(0.975, len(v) - 1) * np.std(v, ddof=1) / math.sqrt(len(v)))
LARGE, LARGE_STAT = [], {}
print("toy, (0.1, 60, 0.24), r = 20000 and 40000, T = 50, seeds 1-4:")
for _r in (20000, 40000):
    _files = [os.path.join(LARGE_DIR, f"toy_r{_r}_c260_k0.24_T50_s{_s}.json") for _s in (1, 2, 3, 4)]
    _missing = [f for f in _files if not os.path.exists(f)]
    if _missing:
        _STOP.append(f"run files not found: {', '.join(_missing)}"); continue
    _runs = [json.load(open(f)) for f in _files]
    assert all(o["r"] == _r and o["c"] == [0.1, 60.0, 0.24] and o["T"] == 50.0 and o["burn"] == 0.3 and o["batches"]["len"] == 5.0 and o["seed"] == k + 1
               for k, o in enumerate(_runs))
    _je = np.array([o["JE"] for o in _runs]); _t1 = np.array([o["T1"] for o in _runs]); _t12 = np.array([o["T12"] for o in _runs])
    _ex = 100 * _t12 / HGI
    _hq = np.concatenate([o["batches"]["hq"] for o in _runs]); _hg = np.concatenate([o["batches"]["hg"] for o in _runs])
    _pf = np.array([o["p_flag"][:2] for o in _runs]); _U = _runs[0]["U"]
    _epi = float(np.sum(_pf) * 0.7 * 50.0 * _r * _r / _U)               # flagged time / (U time units per episode)
    st = dict(JE=float(_je.mean()), JE_se=float(np.std(_je, ddof=1) / 2), T1=float(_t1.mean()), T12=float(_t12.mean()), excess=float(_ex.mean()),
              T1_pct=100 * float(_t1.mean()) / HGI, JE_hw_seeds=_hw(_je), excess_hw_seeds=_hw(_ex), JE_hw_batch=_hw(_hq), excess_hw_batch=100 * _hw(_hq - _hg) / HGI,
              T1_pct_hw_seeds=100 * _hw(_t1) / HGI, predicted=float(_runs[0]["predicted"]["flag_drift"]), measured_flag=float(_r * D * _pf.mean()), episodes=_epi, U=_U)
    LARGE_STAT[_r] = st; LARGE += [dict(r=_r, JE=o["JE"]) for o in _runs]
    got = dict(JE=f"{st['JE']:.2f}", T1=f"{st['T1']:.6f}", T12=f"{st['T12']:.6f}", excess=f"{st['excess']:.2f}", T1_pct=f"{st['T1_pct']:.3f}",
               JE_hw_seeds=f"{st['JE_hw_seeds']:.2f}", excess_hw_seeds=f"{st['excess_hw_seeds']:.2f}", JE_hw_batch=f"{st['JE_hw_batch']:.2f}",
               excess_hw_batch=f"{st['excess_hw_batch']:.2f}", JE_se=f"{st['JE_se']:.2f}", predicted=f"{st['predicted']:.2e}")
    print(f"  r = {_r}: J_E {st['JE']:.4f} (SE {st['JE_se']:.3f}; t-hw {st['JE_hw_seeds']:.3f}; batch hw {st['JE_hw_batch']:.3f})  T1 {st['T1']:.6f}  T1+T2 {st['T12']:.6f}  "
          f"excess {st['excess']:.4f} % (t-hw {st['excess_hw_seeds']:.4f}; batch hw {st['excess_hw_batch']:.4f})  T1 {st['T1_pct']:.4f} %")
    for k, v in LARGE_EXPECTED[_r].items():
        if got[k] != v:
            _STOP.append(f"r = {_r}, {k}: stored {got[k]}, report {v}")
    print(f"       predicted flag drift {st['predicted']:.3e}; measured {st['measured_flag']:.2e} from about {st['episodes']:.1f} flag episodes (not drawn)")
if _STOP:
    print("STOPPED, no figure was written:\n  " + "\n  ".join(_STOP)); sys.exit(1)
# ---------------------------------------------------------------------------------------------------------------------------

prod = [x for x in json.load(open("results/production.json")) if x["kind"] == "ergodic"]
# c2 = 60, r <= 1e4: the Table 2 runs (seeds 1, 2), in every panel
PC_DEEP = [o for o in json.load(open("results/paper_cells.json")) if o["net"].startswith("A") and o["cap"] == "uniform" and o["c"][1] == 60.0]
assert sorted((o["r"], o["seed"]) for o in PC_DEEP) == [(r, s) for r in (300, 1000, 2000, 4000, 10000) for s in (1, 2)] and all("p_flag" in o for o in PC_DEEP)
runs = {"tuned": [x for x in prod if x["consts"] == "tuned"],
        "reference": [x for x in prod if x["consts"] == "reference"],
        "deep": PC_DEEP}


def floor_model(r, c):
    c1, c2, k = c
    return 0.5 * r * D * g_flag(c1 * r ** k, c2 * r ** k, 1 - 0.5 / r, D)


def agg(rows, key):
    out = {}
    for x in rows:
        out.setdefault(x["r"], []).append(key(x))
    rs = np.array(sorted(out))
    m = np.array([np.mean(out[r]) for r in rs])
    se = np.array([np.std(out[r], ddof=1) / np.sqrt(len(out[r])) if len(out[r]) > 1 else 0 for r in rs])
    return rs, m, se


def style(ax):
    ax.set_facecolor(SURF); ax.grid(True, alpha=0.25, lw=0.6, which="both")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8.5)


# ------------------------------------------------------------------------------------------ checks against the printed values
def _fmt(v, nd):
    return f"{v:.{nd}f}"


_bad = []
_rs60, _m60, _se60 = agg(runs["deep"] + LARGE, lambda x: x["JE"])
EXP60 = {300: "4.66", 1000: "3.60", 2000: "2.82", 4000: "3.14", 10000: "2.53", 20000: "4.00", 40000: "2.93"}
print("panel (a), c2 = 60:  r, J_E (plotted), SE across seeds (plotted), report")
for _r, _m, _s in zip(_rs60, _m60, _se60):
    print(f"  {int(_r):6d}  {_m:.4f}  {_s:.4f}  {EXP60[int(_r)]}")
    if _fmt(_m, 2) != EXP60[int(_r)]:
        _bad.append(f"c2 = 60, r = {int(_r)}: J_E {_m:.4f}, report {EXP60[int(_r)]}")
_rs30, _m30, _ = agg(runs["tuned"], lambda x: x["JE"]); _rs10, _m10, _ = agg(runs["reference"], lambda x: x["JE"])
for _r, _e, _nd in ((100, "5.45", 2), (400, "7.56", 2), (1000, "32.8", 1), (4000, "18.5", 1)):
    _v = float(_m30[list(_rs30).index(_r)])
    if _fmt(_v, _nd) != _e:
        _bad.append(f"c2 = 30, r = {_r}: J_E {_v:.4f}, report {_e}")
_v = float(_m10[list(_rs10).index(1000)])
if _fmt(_v, 0) != "777":
    _bad.append(f"c2 = 10, r = 1000: J_E {_v:.3f}, report 777")
for _x, _e in zip(CTRL, ("6.03", "4.47", "3.79", "3.50")):
    if _fmt(_x["JE"], 2) != _e:
        _bad.append(f"control, r = {_x['r']}: J_E {_x['JE']:.4f}, report {_e}")
_ex = {}
for _r in (300, 1000, 2000, 4000, 10000):
    _g = [o for o in PC_DEEP if o["r"] == _r]
    _ex[_r] = float(np.mean(np.concatenate([np.array(o["batches"]["hq"]) - np.array(o["batches"]["hg"]) for o in _g])))
_coef = float(np.exp(np.mean([np.log(_ex[_r] / (60.0 * _r ** 0.24 / _r)) for _r in (1000, 2000, 4000, 10000)])))
_pct = {**{_r: 100 * _v / HGI for _r, _v in _ex.items()}, **{_r: LARGE_STAT[_r]["excess"] for _r in LARGE_STAT}}
for _r, _e, _nd in ((300, "71.2", 1), (1000, "32.1", 1), (2000, "19.4", 1), (4000, "11.2", 1), (10000, "5.6", 1), (20000, "3.26", 2), (40000, "1.94", 2)):
    if _fmt(_pct[_r], _nd) != _e:
        _bad.append(f"panel (c), r = {_r}: excess {_pct[_r]:.4f} %, report {_e}")
if _fmt(_coef, 4) != "2.8329":
    _bad.append(f"panel (c): line coefficient {_coef:.5f}, report 2.8329")
print("panel (c): excess % =", {k: round(v, 3) for k, v in _pct.items()}, " line coefficient %.4f" % _coef)
if _bad:
    print("STOPPED, no figure was written:\n  " + "\n  ".join(_bad)); sys.exit(1)
# ---------------------------------------------------------------------------------------------------------------------------

fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14.5, 4.1), facecolor=SURF)
for ax in (a1, a2, a3):
    style(ax)

# (a) J_E vs r; hollow = predicted flag drift above |theta| (the window average keeps growing)
for key in ("reference", "tuned", "deep"):
    c, col, lab = SETS[key]
    rs, m, se = agg(runs[key] + (LARGE if key == "deep" else []), lambda x: x["JE"])
    a1.plot(rs, m, "-", color=col, lw=1.8, label=lab, zorder=2)
    for r, mm, ss in zip(rs, m, se):
        unstable = floor_model(r, c) > 1
        a1.errorbar([r], [mm], yerr=[ss], fmt="o", ms=6, color=col, mfc=SURF if unstable else col,
                    mew=1.6, elinewidth=1, capsize=2, zorder=3)
a1.errorbar([x["r"] for x in CTRL], [x["JE"] for x in CTRL], yerr=[x["JE_se"] for x in CTRL], fmt="--D", color=SETS["tuned"][1],
            mfc=SETS["tuned"][1], ms=6, lw=1.6, elinewidth=1, capsize=2, zorder=3, label=r"$c_2=30$, flags off")
a1.axhspan(HGI - HGI_SE, HGI + HGI_SE, color=INK2, alpha=0.15, lw=0)
a1.axhline(HGI, color=INK2, ls="--", lw=1.2)
a1.text(105, HGI * 0.80, rf"$\mathrm{{HGI}}_E={HGI:.2f}$", color=INK2, fontsize=8.5)
a1.set_xscale("log"); a1.set_yscale("log"); a1.set_ylim(1.8, 1500)
a1.set_xlabel("r", color=INK2); a1.set_ylabel(r"$J_E$ (window average)", color=INK2)
a1.set_title(r"(a) Ergodic cost vs r  (hollow = predicted flag drift above $|\theta|$)", color=INK, fontsize=10, loc="left")
a1.legend(fontsize=7.8, frameon=False, loc="upper left")

# (b) flag drift: formula curves and measured markers
rg = np.logspace(1.5, 10, 600)
for key in ("reference", "tuned", "deep"):
    c, col, lab = SETS[key]
    a2.loglog(rg, [floor_model(r, c) for r in rg], "-", color=col, lw=1.8)
    rows = runs[key]
    meas = [(x["r"], 0.5 * x["r"] * D * (sum(x["p_flag"]) if "p_flag" in x else x["E_frac"])) for x in rows]
    rs = sorted({m[0] for m in meas})
    mv = [np.mean([m[1] for m in meas if m[0] == r]) for r in rs]
    ok = [(r, v) for r, v in zip(rs, mv) if v > 0]
    if ok:
        a2.loglog(*zip(*ok), "o", color=col, ms=6, mfc=col, zorder=3)
a2.axhline(1.0, color=INK2, ls="--", lw=1.2)
a2.text(2e5, 1.4, r"above $|\theta|=1$: workload drifts up", color=INK2, fontsize=8)
a2.axhline(0.05, color=INK2, ls=":", lw=1)
a2.text(3e7, 0.065, r"5% of $|\theta|$", color=INK2, fontsize=8)
a2.set_ylim(1e-4, 3e3); a2.set_xlim(30, 1e10)
for _r, _st in LARGE_STAT.items():
    # the line uses d = 0.976896, the run files d = 0.97689637: the two predictions agree to 0.03% and round alike
    assert f"{floor_model(_r, SETS['deep'][0]):.2e}" == LARGE_EXPECTED[_r]["predicted"] and abs(floor_model(_r, SETS["deep"][0]) / _st["predicted"] - 1) < 1e-3
a2.set_xlabel("r", color=INK2); a2.set_ylabel("hysteresis idle drift per resource", color=INK2)
a2.set_title("(b) Flag model (lines) vs simulation (dots)", color=INK, fontsize=10, loc="left")

# (c) c2 = 60 from the Table 2 runs (paper_cells.json): T1 + T2 and T1, batch-means half-widths
PC = [o for o in json.load(open("results/paper_cells.json")) if o["net"].startswith("A") and o["cap"] == "uniform" and o["c"][1] == 60.0]
rs_c = np.array(sorted({o["r"] for o in PC}), float)
t12, t12h, t1, t1h = [], [], [], []
for r in rs_c:
    g = [o for o in PC if o["r"] == r]
    hq = np.concatenate([o["batches"]["hq"] for o in g]); hh = np.concatenate([o["batches"]["hhat"] for o in g]); hg = np.concatenate([o["batches"]["hg"] for o in g])
    m, h_ = mean_hw(hq - hg); t12.append(m); t12h.append(h_)
    m, h_ = mean_hw(hq - hh); t1.append(m); t1h.append(h_)
t12, t12h, t1, t1h = map(np.array, (t12, t12h, t1, t1h))
U_over_r = 60.0 * rs_c ** 0.24 / rs_c


def geo_fit(mask):
    return float(np.exp(np.mean(np.log(t12[mask] / U_over_r[mask]))))


def ls_fit(mask):                      # least squares through the origin, weights 1/hw^2
    w = 1.0 / t12h[mask] ** 2
    c = float(np.sum(w * t12[mask] * U_over_r[mask]) / np.sum(w * U_over_r[mask] ** 2))
    return c, float(1.0 / np.sqrt(np.sum(w * U_over_r[mask] ** 2)))


all_r, no300 = np.ones(len(rs_c), bool), rs_c > 300
fit = dict(per_r={int(r): dict(T12=float(a), T12_hw=float(b), T1=float(c_), U_over_r=float(u), ratio=float(a / u), ratio_hw=float(b / u))
                  for r, a, b, c_, u in zip(rs_c, t12, t12h, t1, U_over_r)},
           geometric_mean=dict(all_r=geo_fit(all_r), without_300=geo_fit(no300)),
           weighted_least_squares=dict(all_r=ls_fit(all_r), without_300=ls_fit(no300)),
           HGI_E=HGI, HGI_E_se=HGI_SE, d=D, excess_pct={int(r): float(100 * a / HGI) for r, a in zip(rs_c, t12)})
cfit = fit["geometric_mean"]["without_300"]
a3.errorbar(rs_c, t12 / HGI_E_TOY * 100, yerr=t12h / HGI_E_TOY * 100, fmt="-o", color=AQUA, lw=2, ms=6, capsize=2, label=r"excess $T_1+T_2$ (Table 2 runs)")
a3.errorbar(rs_c, t1 / HGI_E_TOY * 100, yerr=t1h / HGI_E_TOY * 100, fmt="--s", color=AQUA, alpha=0.6, ms=5, capsize=2, label=r"its part $T_1$")
# r = 2e4, 4e4 continue both series; error bar of the excess = across-seed 95% half-width
_rn = np.array(sorted(LARGE_STAT), float)
a3.plot(np.append(rs_c[-1], _rn), np.append(t12[-1] / HGI * 100, [LARGE_STAT[int(r)]["excess"] for r in _rn]), "-", color=AQUA, lw=2)
a3.errorbar(_rn, [LARGE_STAT[int(r)]["excess"] for r in _rn], yerr=[LARGE_STAT[int(r)]["excess_hw_seeds"] for r in _rn], fmt="o", color=AQUA, ms=6, capsize=2)
a3.plot(np.append(rs_c[-1], _rn), np.append(t1[-1] / HGI * 100, [LARGE_STAT[int(r)]["T1_pct"] for r in _rn]), "--", color=AQUA, alpha=0.6)
a3.errorbar(_rn, [LARGE_STAT[int(r)]["T1_pct"] for r in _rn], yerr=[LARGE_STAT[int(r)]["T1_pct_hw_seeds"] for r in _rn], fmt="s", color=AQUA, alpha=0.6, ms=5, capsize=2)
rr = np.logspace(np.log10(250), 5, 100)
a3.plot(rr, cfit * 60 * rr ** -0.76 / HGI_E_TOY * 100, ":", color=INK2, lw=1.2, label=rf"${cfit:.2f}\,U/r$ (fit, $r\geq1000$), $U=60r^{{0.24}}$")
a3.axhline(5, color=INK2, ls="--", lw=1); a3.text(260, 5.6, "5%", color=INK2, fontsize=8)
a3.set_xscale("log"); a3.set_yscale("log"); a3.set_ylim(0.5, 200)
a3.set_xlabel("r", color=INK2); a3.set_ylabel("% of $\\mathrm{HGI}_E$", color=INK2)
a3.set_title("(c) $c_2=60$: excess $T_1+T_2$ and its part $T_1$", color=INK, fontsize=10, loc="left")
a3.legend(fontsize=7.8, frameon=False, loc="upper right")
fig.tight_layout()
assert abs(cfit - _coef) < 1e-12 and all(abs(100 * a / HGI - _pct[int(r)]) < 1e-9 for r, a in zip(rs_c, t12))        # drawn = checked
_texts = [t.get_text() for ax in (a1, a2, a3) for t in ([ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label] + list(ax.texts)
                                                          + (list(ax.get_legend().get_texts()) if ax.get_legend() else []))]
for _w in ("reference", "tuned", "deep band", "transient", "ssc"):
    assert not any(_w in t.lower() for t in _texts), (_w, [t for t in _texts if _w in t.lower()])
os.makedirs(OUT, exist_ok=True)
for _ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"toy_convergence_v5.{_ext}"), dpi=300, bbox_inches="tight", facecolor=SURF)
print("per r: (T1+T2)/(U/r) =", {k: round(v["ratio"], 3) for k, v in fit["per_r"].items()}, "± hw", {k: round(v["ratio_hw"], 3) for k, v in fit["per_r"].items()})
print("geometric-mean fit: all r %.3f, without r = 300 %.3f" % (fit["geometric_mean"]["all_r"], fit["geometric_mean"]["without_300"]))
print("weighted LS fit:    all r %.3f ± %.3f, without r = 300 %.3f ± %.3f" % (*fit["weighted_least_squares"]["all_r"], *fit["weighted_least_squares"]["without_300"]))
print("line %.4f U/r at r = 2e4, 4e4: %s %% of HGI_E" % (cfit, ", ".join("%.3f" % (cfit * 60 * r ** -0.76 / HGI * 100) for r in _rn)))
_dots = {}
for _x in PC_DEEP:
    _dots.setdefault(_x["r"], []).append(0.5 * _x["r"] * D * sum(_x["p_flag"]))
print("panel (b), c2 = 60 dots (Table 2 runs, 0.5 r d sum p_flag, mean of 2 seeds):", {r: float("%.3g" % np.mean(v)) for r, v in sorted(_dots.items())})
print(f"saved {OUT}/toy_convergence_v5.png, {OUT}/toy_convergence_v5.pdf")
