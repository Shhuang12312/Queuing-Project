"""The eta-split check at r = 100 as a statistical test: toy, Definition 2.3, (0.1, 60, 0.24),
T = 150, 30% burn-in, batches of 15; splits (alpha_bar, beta_bar) = (-0.5, 0), (0, +0.5), (-0.25, +0.25) (all eta = -1/2);
8 seeds per split, different across splits (1-8, 11-18, 21-28). Writes results/split_r100.json."""
import json, time, warnings
from multiprocessing import Pool
import numpy as np
import gensim
from common import log_run, mean_hw, se

warnings.filterwarnings("ignore")
K2 = [[1, 0, 1], [0, 1, 1]]
SPLITS = {"all-alpha": ([-0.5] * 3, [0.0] * 3), "all-beta": ([0.0] * 3, [0.5] * 3), "mixed": ([-0.25] * 3, [0.25] * 3)}
SEEDS = {"all-alpha": tuple(range(1, 9)), "all-beta": tuple(range(11, 19)), "mixed": tuple(range(21, 29))}
_c = {}


def go(job):
    name, s = job
    if name not in _c:
        _c[name] = gensim.Net(name, K2, [1, 1, 1], [1, 1, 1], [1, 1, 1], alpha_bar=SPLITS[name][0], beta_bar=SPLITS[name][1])
    o = _c[name].run(100, 150.0, 0.1, 60.0, 0.24, s, burn=0.3, batch=15.0)
    return dict(split=name, seed=s, JE=o["JE"], level=o["level"], hgi_path=o["hgi_path"], E_frac=o["E_frac"], hq=o["batches"]["hq"], hhat=o["batches"]["hhat"], hg=o["batches"]["hg"])


if __name__ == "__main__":
    t0 = time.time()
    with Pool(4) as p:
        res = p.map(go, [(n, s) for n in SPLITS for s in SEEDS[n]], chunksize=2)
    st = {}
    for n in SPLITS:
        g = [o for o in res if o["split"] == n]
        st[n] = {k: (float(np.mean(v)), se(v), mean_hw(v)[1]) for k, v in (("JE", [o["JE"] for o in g]), ("T1", [o["JE"] - o["level"] for o in g]), ("T2", [o["level"] - o["hgi_path"] for o in g]),
                                                                       ("T12", [o["JE"] - o["hgi_path"] for o in g]), ("hG", [o["hgi_path"] for o in g]))}
        print(f"{n:10s} seeds {SEEDS[n][0]}-{SEEDS[n][-1]}: " + "  ".join(f"{k} = {v[0]:.3f} ± {v[1]:.3f}" for k, v in st[n].items()) + "   (± = SE across 8 seeds)")
    z = {}
    for a, b in (("all-alpha", "all-beta"), ("all-alpha", "mixed"), ("all-beta", "mixed")):
        z[f"{a} vs {b}"] = {k: float((st[a][k][0] - st[b][k][0]) / np.hypot(st[a][k][1], st[b][k][1])) for k in ("JE", "T1", "T2", "T12")}
        print(f"   {a} vs {b}: " + "  ".join(f"z({k}) = {v:+.2f}" for k, v in z[f"{a} vs {b}"].items()))
    json.dump(dict(note="Definition 2.3, toy, r = 100, (0.1, 60, 0.24), T = 150, seeds differ across splits", stats=st, pairwise_z=z, runs=res), open("results/split_r100.json", "w"))
    dt = time.time() - t0
    log_run("python split_r100.py", "all-alpha 1-8; all-beta 11-18; mixed 21-28", dt, "sim/results/split_r100.json, split_r100.log", "24 runs on 4 cores")
    print(f"{dt:.0f}s", flush=True)
