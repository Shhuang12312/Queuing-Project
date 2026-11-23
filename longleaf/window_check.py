#!/usr/bin/env python3
"""Does a short window bias T1+T2?  Toy, Definition 2.3, (0.1, 60, 0.24), r = 1000, seeds 1-16, T = 20, 50, 100
(30% burn-in, batches of T/10), against the report's eight-seed value at T = 250 (sim/results/drift_check.json).
Uses the repository's sim/ (build the kernels with sim/build.sh first).
Writes sim/results/longleaf_window_check.json. About 5 minutes on 4 cores."""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "sim"))
H = json.load(open(os.path.join(ROOT, "sim", "results", "multi_bench_crn.json")))["networks"]["A toy (Ex. 2.2)"]["HGI_E"]
R, SEEDS, TS = 1000, range(1, 17), (20.0, 50.0, 100.0)


def one(job):
    T, seed = job
    import gensim
    nw = gensim.Net("A toy (Ex. 2.2)", [[1, 0, 1], [0, 1, 1]], [1, 1, 1], [1, 1, 1], [1, 1, 1], eta=[-0.5] * 3)
    o = nw.run(R, T, 0.1, 60.0, 0.24, seed, burn=0.3, batch=T / 10)
    hq, hh, hg = (np.array(o["batches"][k]) for k in ("hq", "hhat", "hg"))
    return dict(T=T, seed=seed, T1=float((hq - hh).mean()), T2=float((hh - hg).mean()), T12=float((hq - hg).mean()), T12_batches=(hq - hg).tolist(), JE=o["JE"])


if __name__ == "__main__":
    t0 = time.time()
    with Pool(4) as p:
        res = p.map(one, [(T, s) for T in TS for s in SEEDS], chunksize=1)
    out = dict(r=R, c=[0.1, 60.0, 0.24], HGI_E=H, runs=res, summary=[])
    for T in TS:
        v = np.array([o["T12"] for o in res if o["T"] == T]); b = np.array([o["T12_batches"] for o in res if o["T"] == T])
        s = dict(T=T, n=len(v), excess_pct=100 * v.mean() / H, se_pct=100 * v.std(ddof=1) / np.sqrt(len(v)) / H, sd_per_seed_pct=100 * v.std(ddof=1) / H,
                 by_batch_pct=(100 * b.mean(axis=0) / H).tolist(), T1=float(np.mean([o["T1"] for o in res if o["T"] == T])))
        out["summary"].append(s)
        print(f"T = {T:5g}: T1+T2 = {s['excess_pct']:.2f} +- {s['se_pct']:.2f} % of HGI_E (SE, 16 seeds); SD per seed {s['sd_per_seed_pct']:.2f}; T1 = {s['T1']:.5f}; by batch: " + " ".join(f"{x:.1f}" for x in s["by_batch_pct"]))
    out["wall_s"] = time.time() - t0
    json.dump(out, open(os.path.join(ROOT, "sim", "results", "longleaf_window_check.json"), "w"), indent=1)
    print(f"{out['wall_s']:.0f}s")
