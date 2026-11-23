"""Stationarity of two toy configurations by the halves of the observation window.

(a) Toy, Definition 2.3 with (c1, c2, kappa) = (0.1, 30, 0.24) at r = 400 (Table 1 row):
    T = 320, 8 seeds (1-8), batches of 16 (14 post-burn batches), storing h.Q, h^(W), h^(G) per batch, the flag
    fractions and the total idle drift.
(b) Toy, (0.1, 60, 0.24) at r = 1000 and 2000 (Table 2): seeds 3-8, T = 250, batches of 25, to be pooled with
    seeds 1-2 of results/paper_cells.json.
Writes results/drift_check.json. Analysis: python drift_check.py --analyze (prints estimates with 95% CIs)."""
import json, sys, time, warnings
from multiprocessing import Pool
import numpy as np
import networks as N
from common import log_run, halves, mean_hw, HGI_E_TOY

warnings.filterwarnings("ignore")
TOY = "A toy (Ex. 2.2)"
JOBS = [("a", 400, 320.0, 16.0, 30.0, s) for s in range(1, 9)] + [("b", r, 250.0, 25.0, 60.0, s) for r in (2000, 1000) for s in range(3, 9)]
_nw = None


def go(job):
    global _nw
    part, r, T, batch, c2, seed = job
    if _nw is None:
        _nw = N.build(TOY)
    t0 = time.time()
    o = _nw.run(r, T, 0.1, c2, 0.24, seed, burn=0.3, batch=batch)
    o["part"] = part; o["wall_s"] = time.time() - t0; o["cap"] = "uniform"; o["label"] = _nw.tab.label
    return o


def analyze(res):
    PC = json.load(open("results/paper_cells.json"))
    out = {}
    print("(a) toy, c2 = 30, r = 400, T = 320, 8 seeds, 14 post-burn batches of 16")
    g = sorted([o for o in res if o["part"] == "a"], key=lambda o: o["seed"])
    d = 0.976896
    rows = []
    for o in g:
        x = np.array(o["batches"]["hq"]) - np.array(o["batches"]["hg"]); j = np.array(o["batches"]["hq"])
        a, b = halves(x); ja, jb = halves(j)
        tt = np.arange(len(x)); slope = np.polyfit(tt, x, 1)[0] / 16.0          # per diffusion time unit
        rows.append(dict(seed=o["seed"], JE=o["JE"], T12=float(x.mean()), T12_h1=a, T12_h2=b, JE_h1=ja, JE_h2=jb, slope=float(slope),
                         idle=float(np.mean(o["idle_drift"])), flag_only=float(0.5 * 400 * d * o["E_frac"]), flag_per_type=[float(400 * d * p) for p in o["p_flag"]]))
        print(f"   seed {o['seed']}: J_E {o['JE']:6.2f} (halves {ja:6.2f} -> {jb:6.2f})  T1+T2 {x.mean():6.3f} (halves {a:6.3f} -> {b:6.3f})  total idle drift {rows[-1]['idle']:.3f}  flag-only {rows[-1]['flag_only']:.3f}")
    for key, lab in (("JE", "J_E"), ("T12", "T1+T2")):
        dif = np.array([r_[key + "_h2"] - r_[key + "_h1"] for r_ in rows]); m, hw = mean_hw(dif)
        lev = np.array([r_[key] for r_ in rows]); ml, hwl = mean_hw(lev)
        out[f"a_{key}"] = dict(mean=ml, hw=hwl, half_change=m, half_change_hw=hw, rel=m / ml, rel_hw=hw / ml)
        print(f"   {lab}: mean {ml:.3f} ± {hwl:.3f};  second half − first half = {m:+.3f} ± {hw:.3f} (95% CI, 8 seeds) = {100*m/ml:+.1f}% ± {100*hw/ml:.1f}% of the mean")
    sl = np.array([r_["slope"] for r_ in rows]); m, hw = mean_hw(sl)
    out["a_slope"] = dict(mean=m, hw=hw)
    print(f"   least-squares slope of T1+T2 per diffusion time unit: {m:+.5f} ± {hw:.5f}  (window 224 units: {224*m:+.3f} ± {224*hw:.3f})")
    for key, lab in (("idle", "total idle drift / |theta|"), ("flag_only", "flag-only drift 0.5 r d E_frac")):
        m, hw = mean_hw([r_[key] for r_ in rows]); out["a_" + key] = dict(mean=m, hw=hw)
        print(f"   {lab}: {m:.3f} ± {hw:.3f}")
    out["a_rows"] = rows
    print("(b) toy, c2 = 60, T = 250, batches of 25: seeds 1-2 (paper_cells.json) + 3-8 (new)")
    for r in (1000, 2000):
        g = sorted([o for o in PC if o["net"] == TOY and o["r"] == r and o["c"][1] == 60.0] + [o for o in res if o["part"] == "b" and o["r"] == r], key=lambda o: o["seed"])
        rel, lev, t3 = [], [], []
        for o in g:
            x = np.array(o["batches"]["hq"]) - np.array(o["batches"]["hg"]); a, b = halves(x)
            rel.append(100 * (b - a) / x.mean()); lev.append(100 * x.mean() / HGI_E_TOY); t3.append(np.mean(o["batches"]["hg"]) - HGI_E_TOY)
        m, hw = mean_hw(rel); ml, hwl = mean_hw(lev); m3, hw3 = mean_hw(t3)
        tstat = m / (hw / 2.3646) if len(rel) == 8 else float("nan")
        out[f"b_{r}"] = dict(seeds=[o["seed"] for o in g], rel_change_pct=[float(v) for v in rel], mean=m, hw=hw, excess_pct=ml, excess_hw=hwl, T3=m3, T3_hw=hw3)
        print(f"   r = {r}: per-seed change of T1+T2, second vs first half (% of the window mean): {np.round(rel, 1).tolist()}")
        print(f"            mean {m:+.2f}% ± {hw:.2f}% (95% CI, {len(rel)} seeds; t = {tstat:.2f});  excess {ml:.2f} ± {hwl:.2f}% of HGI_E;  T3 = {m3:+.3f} ± {hw3:.3f}")
    pooled = out["b_1000"]["rel_change_pct"] + out["b_2000"]["rel_change_pct"]; m, hw = mean_hw(pooled)
    out["b_pooled"] = dict(mean=m, hw=hw, n=len(pooled))
    print(f"   r = 1000 and 2000 pooled (16 runs): {m:+.2f}% ± {hw:.2f}%")
    return out


if __name__ == "__main__":
    if "--analyze" in sys.argv:
        d = json.load(open("results/drift_check.json")); a = analyze(d["runs"])
        d["analysis"] = a; json.dump(d, open("results/drift_check.json", "w"))
        sys.exit(0)
    t0 = time.time()
    with Pool(4) as p:
        res = p.map(go, JOBS, chunksize=1)
    for o in res:
        if "ygrid" in o: del o["ygrid"]
    json.dump(dict(note="Definition 2.3 verbatim, toy. part a: (0.1, 30, 0.24), r = 400, T = 320, batches of 16, seeds 1-8. part b: (0.1, 60, 0.24), r = 1000, 2000, T = 250, batches of 25, seeds 3-8.",
                   runs=res), open("results/drift_check.json", "w"))
    dt = time.time() - t0
    log_run("python drift_check.py", "a: 1-8; b: 3-8 at r = 1000 and r = 2000", dt, "sim/results/drift_check.json, drift_check.log",
            f"20 runs on 4 cores; longest single run {max(o['wall_s'] for o in res):.0f} s")
    print(f"{dt:.0f}s; longest single run {max(o['wall_s'] for o in res):.0f}s", flush=True)
