"""Table 1: the rows of results/production.json with the flag probabilities stored per job type.

The production.json rows have the total idle drift and the any-flag fraction E_frac, but not P(E_j = 1) per
type. The toy kernel is deterministic given the seed, so the same rows are run here (same constants, T, seeds) and
must reproduce J_E bit for bit; the output adds, per row and per seed:
  total idle drift          r (C_i - used_i)/|theta_i|
  flag-only drift (E_frac)  0.5 r d E_frac                                  (the Figure 1(b) quantity)
  flag drift per resource   r d P(E_i = 1) + r x_3 P(E_3 = 1), i = 1, 2      (exact, from the per-type flag times)
Writes results/table1_drift.json (production.json is only read)."""
import json, time, warnings
from multiprocessing import Pool
import numpy as np
import toysim3
from flagmodel import g_flag
from common import log_run, mean_hw, se

warnings.filterwarnings("ignore")
ROWS = [("reference", 1000, 80, (1, 2)), ("tuned", 100, 160, (1, 2, 3, 4)), ("tuned", 400, 160, (1, 2, 3, 4)), ("tuned", 1000, 160, (1, 2, 3, 4)), ("tuned", 4000, 40, (1, 2, 3, 4))]
D = float(toysim3.TAB.x((1, 0, 0))[0])                       # 0.976896: service rate of a local queue below U
X3 = float(toysim3.TAB.x((0, 0, 1))[2])                      # service rate of the shared type when only it is below U


def go(job):
    cs, r, T, s = job
    t0 = time.time(); o = toysim3.run(r, T, cs, s, burn=0.3, n_grid=400); o["wall_s"] = time.time() - t0
    del o["grid"]
    return o


if __name__ == "__main__":
    t0 = time.time()
    jobs = sorted([(cs, r, T, s) for cs, r, T, seeds in ROWS for s in seeds], key=lambda j: -j[1] ** 2 * j[2])
    with Pool(4) as p:
        res = p.map(go, jobs, chunksize=1)
    prod = [x for x in json.load(open("results/production.json")) if x["kind"] == "ergodic"]
    out = []
    print(f"d = {D:.6f}; x_3 below U = {X3:.4f}")
    for cs, r, T, seeds in ROWS:
        g = sorted([o for o in res if o["consts"] == cs and o["r"] == r], key=lambda o: o["seed"])
        same = all(any(x["consts"] == cs and x["r"] == r and x["seed"] == o["seed"] and x["JE"] == o["JE"] and x["E_frac"] == o["E_frac"] for x in prod) for o in g)
        c1, c2, k = toysim3.CONSTS[cs]
        pred = 0.5 * r * D * g_flag(c1 * r ** k, c2 * r ** k, 1 - 0.5 / r, D)
        idle = np.array([np.mean(o["idle_drift"]) for o in g]); fo = np.array([0.5 * r * D * o["E_frac"] for o in g])
        fr = np.array([np.mean([r * D * o["p_flag"][0] + r * X3 * (o["E_frac"] - o["p_flag"][0] - o["p_flag"][1]), r * D * o["p_flag"][1] + r * X3 * (o["E_frac"] - o["p_flag"][0] - o["p_flag"][1])]) for o in g])
        fl = np.array([np.mean([r * D * o["p_flag"][0], r * D * o["p_flag"][1]]) for o in g]); fb = np.array([np.mean(o["f_below"]) for o in g])
        row = dict(consts=cs, c=(c1, c2, k), r=r, T=T, seeds=list(seeds), reproduces_production_bit_for_bit=bool(same), JE=float(np.mean([o["JE"] for o in g])), JE_se=se([o["JE"] for o in g]),
                   predicted_flag_drift=float(pred), total_idle_drift=float(idle.mean()), total_idle_drift_se=se(idle), flag_only_Efrac=float(fo.mean()), flag_only_Efrac_se=se(fo),
                   flag_drift_local=float(fl.mean()), flag_drift_local_se=se(fl), f_below=float(fb.mean()), per_seed=[dict(seed=o["seed"], JE=o["JE"], idle_drift=o["idle_drift"], E_frac=o["E_frac"], p_flag=o["p_flag"], f_below=o["f_below"]) for o in g])
        out.append(row)
        print(f"c2 = {c2:.0f}, r = {r}, T = {T}: reproduces production.json: {same};  predicted {pred:.3f} | total idle {idle.mean():.3f} ± {se(idle):.3f} | flag-only 0.5 r d E_frac {fo.mean():.3f} ± {se(fo):.3f} | "
              f"r d P(E_j=1), local types {fl.mean():.3f} ± {se(fl):.3f} | f_below {fb.mean():.3f}  -> predicted/flag-only = {pred/fo.mean():.3f}")
    json.dump(dict(note="Table 1 rows rerun with toykern3 (same seeds); adds flag-only drift next to the total idle drift. ± = SE across seeds.", d=D, rows=out), open("results/table1_drift.json", "w"), indent=1)
    dt = time.time() - t0
    log_run("python table1_rerun.py", "c2 = 10 row: 1, 2; c2 = 30 rows: 1-4", dt, "sim/results/table1_drift.json, table1_drift.log", f"18 runs on 4 cores; longest single run {max(o['wall_s'] for o in res):.0f} s")
    print(f"{dt:.0f}s", flush=True)
