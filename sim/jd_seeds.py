"""Discounted cost J_D from an empty start (discount 1 per diffusion unit) for the toy with
(0.1, 60, 0.24): many seeds per r so the SE is small enough to test Theorem 2.5 (HGI_D = 1.336 ± 0.006, results/hgi_d_toy.json).
Horizon T = 12 (e^-12 ~ 6e-6 truncation). toysim3 kernel. Writes results/jd_seeds.json."""
import json, sys, time
from multiprocessing import Pool
import numpy as np
import toysim3 as t

PLAN = [(1000, 32), (2000, 32), (4000, 32), (10000, 8)]
CONSTS = (0.1, 60.0, 0.24)


def go(job):
    r, seed = job
    o = t.run(r, 12.0, CONSTS, seed=seed, burn=0.3)
    return dict(r=r, seed=seed, JD=o["JD"], JD_level=o["JD_level"], E_frac=o["E_frac"])


if __name__ == "__main__":
    t0 = time.time()
    jobs = [(r, s) for r, n in PLAN for s in range(101, 101 + n)]
    jobs.sort(key=lambda j: -j[0])
    res = []
    with Pool(4) as p:
        for o in p.imap_unordered(go, jobs):
            res.append(o); json.dump(res, open("results/jd_seeds.json", "w"))
            if len(res) % 8 == 0:
                print(f"{len(res)}/{len(jobs)} done ({time.time()-t0:.0f}s)", flush=True)
    for r, n in PLAN:
        v = np.array([o["JD"] for o in res if o["r"] == r])
        print(f"r={r:6d} n={len(v)} JD={v.mean():.3f} ± {v.std(ddof=1)/np.sqrt(len(v)):.3f} (SD {v.std(ddof=1):.3f})  vs HGI_D = 1.336", flush=True)
    print("ALL DONE", flush=True)
