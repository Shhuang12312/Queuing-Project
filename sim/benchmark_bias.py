"""Grid bias of the HGI_E benchmark on the toy (theta = (-1,-1), Sigma = [[4,2],[2,4]]).

Part A (grid schemes, common Brownian paths). For each seed (0-7) one Brownian path of length T = 20000 is drawn at
dt = 5e-4 and coarsened to dt = 1e-3, 4e-3, 1e-2 by summing increments, so every dt sees the same path. On each grid:
  * reflection map: W = X - min(0, running min X) per coordinate (the scheme of benchmark_hgi_e);
  * reflected Euler: w <- max(w + dX, 0) per step, as an explicit loop on the SAME increments.
The time average of h^(W) = max(W1, W2) over [200, T] is recorded for both. sqrt(dt) extrapolations: the pair
(4e-3, 1e-3) used for the reported HGI_E, the pair (1e-3, 5e-4), and a least-squares line in sqrt(dt) over all four.
Part B (independent check without time discretization). A lattice free process with the same drift and (to O(1/r))
covariance: independent Poisson arrivals at rates alpha^r_j = 1 - 0.5/r and Poisson service completions at rate 1 per
type, X = K(A - S)/r, reflected exactly per coordinate; r = 25, 50, 100, T = 20000, seeds 0-7. Its bias is O(1/r)
(per-coordinate stationary mean 2 - 1/r), so a line in 1/r gives a second estimate of HGI_E.
Part C. The discounted benchmark at dt = 1e-2 (one leg).
Writes results/benchmark_bias.json."""
import json, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, "../rsn_flat")
from rsn import make_2lln
from simulation import benchmark_hgi_d
from common import log_run, mean_hw, se

T, T0, DTF = 20000.0, 200.0, 5e-4
LEVELS = {5e-4: 1, 1e-3: 2, 4e-3: 8, 1e-2: 20}
SEEDS = tuple(range(8))
LCH = np.linalg.cholesky(np.array([[4.0, 2.0], [2.0, 4.0]])); THETA = np.array([-1.0, -1.0])
CHUNK_T = 2000.0                                     # time units per chunk (4e6 fine steps)


def euler_loop(dx1, dx2, w1, w2, skip):
    """Explicit reflected Euler on lists of increments; returns (sum of max(w1,w2) after `skip` steps, w1, w2)."""
    acc = 0.0
    for k in range(len(dx1)):
        w1 += dx1[k]; w2 += dx2[k]
        if w1 < 0.0: w1 = 0.0
        if w2 < 0.0: w2 = 0.0
        if k >= skip:
            acc += w1 if w1 > w2 else w2
    return acc, w1, w2


def grid_seed(seed):
    rng = np.random.default_rng(seed)
    st = {dt: dict(A=np.zeros(2), mn=np.zeros(2), acc=0.0, n=0, w=(0.0, 0.0), eacc=0.0) for dt in LEVELS}
    nfine = int(round(CHUNK_T / DTF)); nchunks = int(round(T / CHUNK_T)); t_done = 0.0
    for c in range(nchunks):
        dXf = rng.standard_normal((nfine, 2)) @ LCH.T * np.sqrt(DTF) + THETA * DTF
        for dt, k in LEVELS.items():
            dX = dXf if k == 1 else dXf.reshape(nfine // k, k, 2).sum(axis=1)
            s = st[dt]; m = len(dX)
            A = s["A"] + np.cumsum(dX, axis=0); Mn = np.minimum(np.minimum.accumulate(A, axis=0), s["mn"]); W = A - Mn
            skip = int(round(max(0.0, T0 - t_done) / dt)); skip = min(skip, m)
            s["acc"] += float(np.maximum(W[skip:, 0], W[skip:, 1]).sum()); s["n"] += m - skip
            s["A"], s["mn"] = A[-1], Mn[-1]
            ea, w1, w2 = euler_loop(dX[:, 0].tolist(), dX[:, 1].tolist(), s["w"][0], s["w"][1], skip)
            s["eacc"] += ea; s["w"] = (w1, w2)
        t_done += CHUNK_T
    return dict(seed=seed, map={str(dt): st[dt]["acc"] / st[dt]["n"] for dt in LEVELS}, euler={str(dt): st[dt]["eacc"] / st[dt]["n"] for dt in LEVELS})


JUMP = np.array([[1, 0], [0, 1], [1, 1], [-1, 0], [0, -1], [-1, -1]], dtype=np.int64)


def lattice(job):
    r, seed = job
    rng = np.random.default_rng(1000 + seed)
    a = 1.0 - 0.5 / r; rates = np.array([a, a, a, 1.0, 1.0, 1.0]); lam = rates.sum(); cdf = np.cumsum(rates) / lam
    n_total = int(round(lam * r * r * T)); n_skip = int(round(lam * r * r * T0))
    A = np.zeros(2, dtype=np.int64); mn = np.zeros(2, dtype=np.int64); acc = 0; n = 0; done = 0
    while done < n_total:
        m = min(8_000_000, n_total - done)
        typ = np.searchsorted(cdf, rng.random(m)); J = JUMP[np.minimum(typ, 5)]
        P = A + np.cumsum(J, axis=0); Mn = np.minimum(np.minimum.accumulate(P, axis=0), mn); W = P - Mn
        skip = min(max(0, n_skip - done), m)
        acc += int(np.maximum(W[skip:, 0], W[skip:, 1]).sum()); n += m - skip
        A, mn = P[-1], Mn[-1]; done += m
    return dict(r=r, seed=seed, value=acc / n / r, events=n_total)


if __name__ == "__main__":
    t0 = time.time()
    with Pool(4) as p:
        grid = p.map(grid_seed, SEEDS, chunksize=1)
        t_grid = time.time() - t0
        lat = p.map(lattice, sorted([(r, s) for r in (25, 50, 100) for s in SEEDS], key=lambda j: -j[0]), chunksize=1)
    t_lat = time.time() - t0 - t_grid
    dts = sorted(LEVELS)
    M = {dt: np.array([g["map"][str(dt)] for g in grid]) for dt in dts}; E = {dt: np.array([g["euler"][str(dt)] for g in grid]) for dt in dts}
    out = dict(config=dict(T=T, T0=T0, seeds=list(SEEDS), dt=dts, note="common Brownian path per seed across dt; averages over [200, 20000]"), grid=grid, lattice=lat)
    print("Part A: time average of h^(W), 8 seeds, mean ± SE (same Brownian paths at every dt)")
    for dt in dts[::-1]:
        d = np.max(np.abs(M[dt] - E[dt]))
        print(f"   dt = {dt:7.4f}: reflection map {M[dt].mean():.4f} ± {se(M[dt]):.4f}   reflected Euler {E[dt].mean():.4f} ± {se(E[dt]):.4f}   max |map − Euler| over seeds {d:.1e}")
        out[f"A_dt_{dt}"] = dict(map=float(M[dt].mean()), map_se=se(M[dt]), euler=float(E[dt].mean()), euler_se=se(E[dt]), max_abs_diff=float(d))
    def extrap(d1, d2):
        s1, s2 = np.sqrt(d1), np.sqrt(d2); v = M[d2] + (M[d2] - M[d1]) * s2 / (s1 - s2); return v
    for d1, d2 in ((4e-3, 1e-3), (1e-3, 5e-4), (1e-2, 4e-3), (1e-2, 5e-4)):
        v = extrap(d1, d2); m, hw = mean_hw(v)
        out[f"A_extrap_{d1}_{d2}"] = dict(mean=m, se=se(v), hw95=hw)
        print(f"   sqrt(dt) extrapolation from dt = ({d1}, {d2}): {m:.4f} ± {se(v):.4f} (SE), 95% CI ± {hw:.4f}")
    sq = np.sqrt(np.array(dts)); fits = np.array([np.polyfit(sq, [g["map"][str(dt)] for dt in dts], 1) for g in grid])
    m0, hw0 = mean_hw(fits[:, 1]); ms, hws = mean_hw(fits[:, 0])
    out["A_line_fit"] = dict(intercept=m0, intercept_se=se(fits[:, 1]), intercept_hw95=hw0, slope=ms, slope_se=se(fits[:, 0]), slope_hw95=hws)
    print(f"   least-squares line in sqrt(dt) over the four grids, per seed: intercept {m0:.4f} ± {se(fits[:,1]):.4f} (SE), slope {ms:.3f} ± {se(fits[:,0]):.3f}")
    for dt in dts[::-1]:
        b = 100 * (M[dt] / fits[:, 1] - 1); print(f"   bias of the grid value at dt = {dt}: {b.mean():+.2f}% ± {se(b):.2f}% of the fitted intercept")
        out[f"A_bias_pct_{dt}"] = dict(mean=float(b.mean()), se=se(b))
    print("Part B: lattice free process (no time discretization), 8 seeds, T = 20000")
    Lr = {}
    for r in (25, 50, 100):
        v = np.array([o["value"] for o in lat if o["r"] == r]); Lr[r] = v
        print(f"   r = {r:4d}: {v.mean():.4f} ± {se(v):.4f}  ({[o['events'] for o in lat if o['r'] == r][0]:.2e} events per seed)")
        out[f"B_r_{r}"] = dict(mean=float(v.mean()), se=se(v))
    inv = np.array([1 / 25, 1 / 50, 1 / 100]); fl = np.array([np.polyfit(inv, [Lr[r][i] for r in (25, 50, 100)], 1) for i in range(len(SEEDS))])
    mb, hwb = mean_hw(fl[:, 1]); two = 2 * Lr[100] - Lr[50]; m2, hw2 = mean_hw(two)
    out["B_line_fit"] = dict(intercept=mb, intercept_se=se(fl[:, 1]), intercept_hw95=hwb, slope=float(fl[:, 0].mean()), two_point_50_100=m2, two_point_se=se(two), two_point_hw95=hw2)
    print(f"   line in 1/r over r = 25, 50, 100 (per seed; seeds are independent across r): intercept {mb:.4f} ± {se(fl[:,1]):.4f} (SE), 95% CI ± {hwb:.4f}; slope {fl[:,0].mean():.2f}")
    print(f"   two-point Richardson from r = 50, 100: {m2:.4f} ± {se(two):.4f} (SE), 95% CI ± {hw2:.4f}")
    net = make_2lln(); t1 = time.time()
    md, sed = benchmark_hgi_d(net, dt=1e-2, n_paths=10000); md2, sed2 = benchmark_hgi_d(net, dt=4e-3, n_paths=10000)
    out["C_hgi_d"] = dict(dt_1e_2=(md, sed), dt_4e_3=(md2, sed2), reference=1.3363)
    print(f"Part C: HGI_D single leg at dt = 1e-2: {md:.4f} ± {sed:.4f} ({100*(md/1.3363-1):+.1f}% vs 1.3363); dt = 4e-3: {md2:.4f} ± {sed2:.4f} ({100*(md2/1.3363-1):+.1f}%)")
    json.dump(out, open("results/benchmark_bias.json", "w"), indent=1)
    dt_all = time.time() - t0
    log_run("python benchmark_bias.py", "grid: 0-7 (default_rng(seed)); lattice: default_rng(1000 + seed), seeds 0-7 per r; HGI_D: 12345", dt_all,
            "sim/results/benchmark_bias.json, benchmark_bias.log", f"grid part {t_grid:.0f} s, lattice part {t_lat:.0f} s on 4 cores")
    print(f"{dt_all:.0f}s", flush=True)
