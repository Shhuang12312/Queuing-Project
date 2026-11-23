"""Free-process increment test. Default: toy, r = 2000, (0.1, 60, 0.24), T = 500, burn 30%.
Usage: python increment_test.py [network name] [c2] [output json]   e.g.  "D unequal rates" 41 results/increment_test_D.json
sample X^(t+1) - X^(t) = Y(t+1)/r - Y(t)/r on the unit grid of the post-burn window (Y = W - idleness, sampled
exactly inside holding intervals by the kernel), 2 seeds. Reports the sample mean vs theta = (-1, -1) and the
sample covariance vs Sigma = [[4, 2], [2, 4]] with standard errors (batch-means SE for the mean; delta-method /
moment SE for the covariance entries), plus the lag-1 autocorrelation of the increments (should be ~0: the
pre-limit free process is a compensated Poisson martingale plus drift theta t)."""
import json, sys, time
from multiprocessing import Pool
import numpy as np
import networks as N

NAME = sys.argv[1] if len(sys.argv) > 1 else "A toy (Ex. 2.2)"
C2 = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
OUT = sys.argv[3] if len(sys.argv) > 3 else "results/increment_test.json"
_nw = None


def run(seed):
    global _nw
    if _nw is None:
        _nw = N.build(NAME)
    o = _nw.run(2000, 500.0, 0.1, C2, 0.24, seed, burn=0.3, ygrid=1.0)
    return seed, o["ygrid"]["Y"], o["E_frac"], o["JE"], o["hgi_path"]


if __name__ == "__main__":
    t0 = time.time()
    with Pool(2) as p:
        res = p.map(run, [1, 2])
    n = N.build(NAME).net
    theta, Sigma = n.theta, n.Sigma
    dX_all = []
    for seed, Y, ef, je, hg in res:
        dX = np.diff(Y, axis=0); dX_all.append(dX)
        print(f"seed {seed}: {len(dX)} unit increments; mean {dX.mean(axis=0).round(4).tolist()}; "
              f"cov {np.cov(dX.T).round(3).tolist()}; lag-1 autocorr "
              f"{[round(float(np.corrcoef(dX[:-1, i], dX[1:, i])[0, 1]), 3) for i in range(n.I)]}; "
              f"E_frac {ef:.1e}; JE {je:.3f}; h^(G) {hg:.3f}  ({time.time()-t0:.0f}s)")
    dX = np.vstack(dX_all); N_ = len(dX)
    mean = dX.mean(axis=0)
    se_mean = dX.std(axis=0, ddof=1) / np.sqrt(N_)
    # batch-means SE of the mean (batches of 25 increments) as a check on the iid SE
    nb = N_ // 25
    bm = dX[:nb * 25].reshape(nb, 25, 2).mean(axis=1)
    se_bm = bm.std(axis=0, ddof=1) / np.sqrt(nb)
    cov = np.cov(dX.T)
    c = dX - mean
    pairs = [(i, j) for i in range(n.I) for j in range(i, n.I)]
    prods = np.stack([c[:, i] * c[:, j] for i, j in pairs], axis=1)
    se_cov = prods.std(axis=0, ddof=1) / np.sqrt(N_)
    z_mean = (mean - theta) / se_mean
    z_cov = (np.array([cov[i, j] for i, j in pairs]) - np.array([Sigma[i, j] for i, j in pairs])) / se_cov
    print(f"\nPOOLED {N_} unit increments (2 seeds), network {NAME!r}, c2 = {C2}:")
    print(f"  mean      = {mean.round(4).tolist()}  SE (iid) {se_mean.round(4).tolist()}  SE (batch means, {nb} x 25) {se_bm.round(4).tolist()}"
          f"  vs theta = {theta.round(4).tolist()}  z = {z_mean.round(2).tolist()}")
    print(f"  cov       = {np.round(cov, 3).tolist()}  SE (upper triangle {pairs}) {se_cov.round(3).tolist()}"
          f"  vs Sigma = {np.round(Sigma, 4).tolist()}  z = {z_cov.round(2).tolist()}")
    verdict = "PASS" if (np.all(np.abs(z_mean) < 3) and np.all(np.abs(z_cov) < 3)) else "FAIL"
    print(f"  -> {verdict} (all |z| < 3)")
    json.dump(dict(network=NAME, c2=C2, r=2000, T=500.0, N=int(N_), cov_pairs=pairs, mean=mean.tolist(), se_mean=se_mean.tolist(), se_mean_bm=se_bm.tolist(), theta=theta.tolist(),
                   cov=cov.tolist(), se_cov=se_cov.tolist(), Sigma=Sigma.tolist(), z_mean=z_mean.tolist(), z_cov=z_cov.tolist(),
                   verdict=verdict, per_seed=[dict(seed=s_, n=len(d), mean=d.mean(axis=0).tolist(), cov=np.cov(d.T).tolist())
                                               for (s_, *_), d in zip(res, dX_all)]),
              open(OUT, "w"), indent=1)
