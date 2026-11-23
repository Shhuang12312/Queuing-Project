"""The screen of Section 5.1: toy, Definition 2.3, r = 300, T = 40, 30% burn-in,
c1 in {0.1, 1}, c2 in {10, 30}, kappa in {0.15, 0.24}, 8 seeds (1-8), batches of 4 (7 post-burn batches).
Writes results/screen_r300.json."""
import json, time, warnings
from multiprocessing import Pool
import numpy as np
import networks as N
from common import log_run, halves, mean_hw, se

warnings.filterwarnings("ignore")
CELLS = [(c1, c2, k) for c1 in (0.1, 1.0) for c2 in (10.0, 30.0) for k in (0.15, 0.24)]
SEEDS = tuple(range(1, 9)); _nw = None


def go(job):
    global _nw
    c1, c2, k, s = job
    if _nw is None:
        _nw = N.build("A toy (Ex. 2.2)")
    o = _nw.run(300, 40.0, c1, c2, k, s, burn=0.3, batch=4.0)
    return dict(c=(c1, c2, k), seed=s, JE=o["JE"], JE_1st=o["JE_1st"], JE_2nd=o["JE_2nd"], hgi_path=o["hgi_path"], E_frac=o["E_frac"], hq=o["batches"]["hq"], hg=o["batches"]["hg"])


if __name__ == "__main__":
    t0 = time.time()
    with Pool(4) as p:
        res = p.map(go, [c + (s,) for c in CELLS for s in SEEDS], chunksize=4)
    summ = []
    print(f"{'(c1, c2, kappa)':18s} {'J_E, 8 seeds':>18s} {'seeds 1-2':>10s} {'halves (8 seeds)':>20s} {'T1+T2':>16s}")
    for c in CELLS:
        g = sorted([o for o in res if tuple(o["c"]) == c], key=lambda o: o["seed"]); je = np.array([o["JE"] for o in g]); t12 = np.array([o["JE"] - o["hgi_path"] for o in g])
        m, hw = mean_hw(je)
        row = dict(c=c, JE=m, JE_se=se(je), JE_hw95=hw, JE_seeds_1_2=float(je[:2].mean()), half1=float(np.mean([o["JE_1st"] for o in g])), half2=float(np.mean([o["JE_2nd"] for o in g])),
                   T12=float(t12.mean()), T12_se=se(t12), per_seed=je.tolist())
        summ.append(row)
        print(f"{str(c):18s} {m:8.2f} ± {se(je):6.2f} {row['JE_seeds_1_2']:10.2f} {row['half1']:9.2f} -> {row['half2']:8.2f} {row['T12']:8.2f} ± {row['T12_se']:5.2f}")
    a = np.array(next(r_ for r_ in summ if r_["c"] == (0.1, 30.0, 0.24))["per_seed"]); b = np.array(next(r_ for r_ in summ if r_["c"] == (1.0, 30.0, 0.24))["per_seed"])
    d = b - a; m, hw = mean_hw(d)
    print(f"tie: J_E(1, 30, 0.24) − J_E(0.1, 30, 0.24), paired by seed: {m:+.3f} ± {hw:.3f} (95% CI, 8 seeds); per seed {np.round(d, 2).tolist()}")
    json.dump(dict(note="Definition 2.3, toy, r = 300, T = 40, burn 0.3, seeds 1-8", summary=summ, tie=dict(diff=m, hw95=hw, per_seed=d.tolist()), runs=res), open("results/screen_r300.json", "w"))
    dt = time.time() - t0
    log_run("python screen_r300.py", "1-8 per cell (8 cells)", dt, "sim/results/screen_r300.json, screen_r300.log", "64 runs on 4 cores")
    print(f"{dt:.0f}s", flush=True)
