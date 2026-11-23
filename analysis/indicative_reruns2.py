"""Second batch of short runs (see indicative_reruns.py): the no-flag ("control") configuration.
  5. eta-split at r = 100 with flags off and |v^b| = 0.24, for several (c2, kappa), windows and seed sets;
  6. control experiment with long windows (so the window average is close to the stationary mean).
Run from anywhere:  python analysis/indicative_reruns2.py  -> analysis/indicative_reruns2.json"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indicative_reruns import job, ROOT

if __name__ == "__main__":
    t0 = time.time(); jobs = []
    splits = (("split all-alpha", [-0.5] * 3, [0.0] * 3), ("split all-beta", [0.0] * 3, [0.5] * 3), ("split mixed", [-0.25] * 3, [0.25] * 3))
    for name, ab, bb in splits:
        for (c2, kap) in ((10.0, 0.22), (30.0, 0.24)):
            for T in (20.0, 40.0, 80.0, 160.0):
                jobs += [(name + " flags off vb=0.24", 0.24, ab, bb, 100, T, -1.0, c2, kap, s) for s in (0, 1, 2, 3, 4)]
    for (c2, kap) in ((10.0, 0.22), (30.0, 0.24)):
        for r, T in ((30, 4000.0), (100, 2000.0), (300, 400.0), (1000, 100.0)):
            jobs += [("control long vb=0.24", 0.24, None, None, r, T, -1.0, c2, kap, s) for s in (1, 2)]
    jobs.sort(key=lambda a: -(a[4] ** 2) * a[5])
    with Pool(4) as p:
        res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(os.path.join(ROOT, "analysis", "indicative_reruns2.json"), "w"), indent=0)
    print("5. eta-split at r = 100, flags off, |v^b| = 0.24: J_E mean ± SE over the seed set")
    for (c2, kap) in ((10.0, 0.22), (30.0, 0.24)):
        for T in (20.0, 40.0, 80.0, 160.0):
            for seeds in ((0, 1, 2, 3), (1, 2, 3, 4), (0, 1, 2), (1, 2)):
                out = []
                for name, _, _ in splits:
                    v = np.array([o["JE"] for o in res if o["tag"].startswith(name) and o["c"][1] == c2 and o["T"] == T and o["seed"] in seeds])
                    out.append(f"{v.mean():.3f} ± {v.std(ddof=1)/np.sqrt(len(v)):.3f}")
                print(f"  c2={c2:.0f} kappa={kap} T={T:.0f} seeds={seeds}: " + " / ".join(out))
    print("6. control experiment, long windows (2 seeds):")
    for (c2, kap) in ((10.0, 0.22), (30.0, 0.24)):
        for r in (30, 100, 300, 1000):
            g = [o for o in res if o["tag"].startswith("control long") and o["c"][1] == c2 and o["r"] == r]
            print(f"  c2={c2:.0f} kappa={kap} r={r:5d} T={g[0]['T']:.0f}: J_E = {[round(o['JE'], 3) for o in g]} mean {np.mean([o['JE'] for o in g]):.3f}; "
                  f"h^(G) = {[round(o['hgi_path'], 3) for o in g]}; halves {[(round(o['JE_1st'], 2), round(o['JE_2nd'], 2)) for o in g]}")
    print(f"{time.time()-t0:.0f}s")
