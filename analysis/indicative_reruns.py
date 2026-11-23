"""Short runs (each well under two minutes) of four configurations of the report, with short windows and few seeds.
The numbers printed in the report come from the longer stored runs of sim/control_experiment.py, sim/screen_r300.py
and sim/split_r100.py.
Run from anywhere:  python analysis/indicative_reruns.py   -> analysis/indicative_reruns.json
  1. control experiment (flags off, |v^b| raised): J_E at r = 30, 100, 300, 1000;
  2. flags off at the definition's own b_max;
  3. the Section 5.1 screen at r = 300 over c1 in {0.1, 1}, c2 in {10, 30}, kappa in {0.15, 0.24};
  4. the eta-split check at r = 100.
Flags are switched off by passing c1 = -1 (L < 0, so no queue is ever below L); Z still uses U = c2 r^kappa."""
import json, os, sys, time, warnings
from multiprocessing import Pool
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim")); sys.path.insert(0, os.path.join(ROOT, "rsn_flat"))
os.chdir(os.path.join(ROOT, "sim"))
warnings.filterwarnings("ignore")
import gensim
from rsn import RSN
from offline import OfflineTables

K2 = [[1, 0, 1], [0, 1, 1]]
_c = {}


def net(vb_norm=None, alpha_bar=None, beta_bar=None):
    key = (vb_norm, None if alpha_bar is None else tuple(alpha_bar), None if beta_bar is None else tuple(beta_bar))
    if key not in _c:
        kw = dict(eta=[-0.5] * 3) if alpha_bar is None else dict(alpha_bar=alpha_bar, beta_bar=beta_bar)
        tab = None
        if vb_norm is not None:
            tab = OfflineTables(RSN(K2, [1, 1, 1], [1, 1, 1], [1, 1, 1], **kw), gamma_c=0.98, gamma_b=0.98, vb_norm=vb_norm)
        _c[key] = gensim.Net("toy", K2, [1, 1, 1], [1, 1, 1], [1, 1, 1], tables=tab, **kw)
    return _c[key]


def job(a):
    tag, vb, ab, bb, r, T, c1, c2, kap, seed = a
    o = net(vb, ab, bb).run(r, T, c1, c2, kap, seed)
    return dict(tag=tag, vb_norm=vb, r=r, T=T, c=(c1, c2, kap), seed=seed, JE=o["JE"], JE_1st=o["JE_1st"], JE_2nd=o["JE_2nd"],
                hgi_path=o["hgi_path"], E_frac=o["E_frac"])


if __name__ == "__main__":
    t0 = time.time(); jobs = []
    for vb, tag in ((0.24, "control vb=0.24"), (0.25, "control vb=0.25"), (None, "flags off at b_max")):
        for (c2, kap) in ((30.0, 0.24), (10.0, 0.22)):
            for r in (30, 100, 300, 1000):
                jobs += [(tag, vb, None, None, r, 40.0, -1.0, c2, kap, s) for s in (1, 2, 3, 4)]
    for c1 in (0.1, 1.0):
        for c2 in (10.0, 30.0):
            for kap in (0.15, 0.24):
                jobs += [("screen r=300", None, None, None, 300, T, c1, c2, kap, s) for T in (20.0, 40.0) for s in (1, 2)]
    for name, ab, bb in (("split all-alpha", [-0.5] * 3, [0.0] * 3), ("split all-beta", [0.0] * 3, [0.5] * 3), ("split mixed", [-0.25] * 3, [0.25] * 3)):
        for (c1, c2, kap) in ((0.1, 30.0, 0.24), (0.1, 60.0, 0.24), (0.5, 10.0, 0.22)):
            jobs += [(name, None, ab, bb, 100, 40.0, c1, c2, kap, s) for s in (1, 2, 3, 4)]
        jobs += [(name + " flags off vb=0.24", 0.24, ab, bb, 100, 40.0, -1.0, 30.0, 0.24, s) for s in (1, 2, 3, 4)]
    with Pool(4) as p:
        res = p.map(job, jobs, chunksize=4)
    json.dump(res, open(os.path.join(ROOT, "analysis", "indicative_reruns.json"), "w"), indent=0)
    def show(filt, keyf):
        groups = {}
        for o in res:
            if filt(o): groups.setdefault(keyf(o), []).append(o)
        for k, g in groups.items():
            je = np.array([o["JE"] for o in g]); se = je.std(ddof=1) / np.sqrt(len(je)) if len(je) > 1 else float("nan")
            print(f"  {str(k):70s} n={len(g)} J_E={je.mean():9.3f} ± {se:7.3f}  halves {np.mean([o['JE_1st'] for o in g]):8.2f} -> {np.mean([o['JE_2nd'] for o in g]):8.2f}  h^(G)={np.mean([o['hgi_path'] for o in g]):.3f}")
    print("1-2. flags off (T = 40, 4 seeds; SE across seeds):"); show(lambda o: o["tag"].startswith(("control", "flags off")), lambda o: (o["tag"], o["c"][1], o["c"][2], o["r"]))
    print("3. screen at r = 300:"); show(lambda o: o["tag"].startswith("screen"), lambda o: (o["c"], o["T"]))
    print("4. eta-split at r = 100 (T = 40, seeds 1-4, same seeds across splits):"); show(lambda o: o["tag"].startswith("split"), lambda o: (o["tag"], o["c"]))
    print(f"{time.time()-t0:.0f}s")
