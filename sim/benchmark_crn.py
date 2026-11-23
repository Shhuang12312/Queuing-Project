"""HGI_E for all seven networks with both grid legs on the same Brownian path, paths generated at dt = 5e-4.

Each seed draws one path at dt = 5e-4 (T = 20000, first 200 time units dropped), coarsens it to dt = 2e-3, and
extrapolates in sqrt(dt) per seed, e2 + (e2 - e1) s2/(s1 - s2); 8 seeds (0-7). Exact per-coordinate reflection map,
exact dual-vertex h^. The values of results/multi_bench.json (independent legs) are printed next to the result.
Writes results/benchmark_crn.json."""
import json, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, "../rsn_flat")
from rsn import RSN
from simulation import make_hhat
import networks as N
from common import log_run, mean_hw, se

T, T0, DTF, KC = 20000.0, 200.0, 5e-4, 4
CHUNK_T = 1000.0


def one(job):
    name, seed = job
    net = RSN(*N.SPECS[name], name=name); hh = make_hhat(net); I = net.I
    L = np.linalg.cholesky(net.Sigma); rng = np.random.default_rng(seed)
    nf = int(round(CHUNK_T / DTF)); st = [dict(A=np.zeros(I), mn=np.zeros(I), acc=0.0, n=0) for _ in range(2)]; t_done = 0.0
    for _ in range(int(round(T / CHUNK_T))):
        dXf = rng.standard_normal((nf, I)) @ L.T * np.sqrt(DTF) + net.theta * DTF
        for lev, (k, dt) in enumerate(((1, DTF), (KC, DTF * KC))):
            dX = dXf if k == 1 else dXf.reshape(nf // k, k, I).sum(axis=1); s = st[lev]; m = len(dX)
            A = s["A"] + np.cumsum(dX, axis=0); Mn = np.minimum(np.minimum.accumulate(A, axis=0), s["mn"]); W = A - Mn
            skip = min(int(round(max(0.0, T0 - t_done) / dt)), m)
            s["acc"] += float(hh(W[skip:]).sum()); s["n"] += m - skip; s["A"], s["mn"] = A[-1], Mn[-1]
        t_done += CHUNK_T
    e2, e1 = st[0]["acc"] / st[0]["n"], st[1]["acc"] / st[1]["n"]; s1, s2 = np.sqrt(DTF * KC), np.sqrt(DTF)
    return dict(net=name, seed=seed, fine=e2, coarse=e1, extrap=e2 + (e2 - e1) * s2 / (s1 - s2))


if __name__ == "__main__":
    t0 = time.time(); B = json.load(open("results/multi_bench.json"))
    with Pool(4) as p:
        res = p.map(one, [(n, s) for n in N.SPECS for s in range(8)], chunksize=1)
    out = {}
    print(f"{'network':28s} {'common-path HGI_E (8 seeds)':>30s} {'95% hw':>8s} {'multi_bench.json':>18s} {'difference':>12s}")
    for n in N.SPECS:
        v = np.array([o["extrap"] for o in res if o["net"] == n]); m, hw = mean_hw(v); old = B[n]["HGI_E"]
        out[n] = dict(HGI_E=m, se=se(v), hw95=hw, per_seed=v.tolist(), fine_dt_mean=float(np.mean([o["fine"] for o in res if o["net"] == n])), multi_bench=old, rel_diff_pct=100 * (old / m - 1))
        print(f"{n[:28]:28s} {m:14.4f} ± {se(v):.4f} (SE) {hw:8.4f} {old:18.4f} {100*(old/m-1):+11.2f}%")
    json.dump(dict(config=dict(T=T, T0=T0, dt=(DTF * KC, DTF), seeds=list(range(8))), networks=out, runs=res), open("results/benchmark_crn.json", "w"), indent=1)
    dt = time.time() - t0
    log_run("python benchmark_crn.py", "0-7 per network (default_rng(seed))", dt, "sim/results/benchmark_crn.json, benchmark_crn.log", "56 paths on 4 cores")
    print(f"{dt:.0f}s", flush=True)
