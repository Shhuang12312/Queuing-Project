"""A J = 18 network end to end (6 resources, 6 local + 12 shared types, heterogeneous
primitives, deterministic from seed 2026). Offline pass with the full 2^18 v^b table (every
entry asserted), b_max from the true lambda~, then two short Definition-2.3 runs at r = 300 with
the compiled engine (needs genkern.c built with MAXJ >= 18). Writes results/j18_e2e.json.
~4 min. The network is a mechanics test: its eq. (6) cap is b_max ~ 2.5e-5, so at r = 300 the
flags make it transient, which the run reports (E_frac, growing halves)."""
import json, sys, time, warnings
import numpy as np
sys.path.insert(0, "../rsn_flat")
from rsn import RSN                      # noqa: E402
from offline import OfflineTables        # noqa: E402
from gensim import Net                   # noqa: E402

warnings.filterwarnings("ignore")


def make_j18(seed=2026, I=6, J=18):
    rng = np.random.default_rng(seed)
    cols = [np.eye(I)[:, i] for i in range(I)]
    seen = set()
    while len(cols) < J:
        k = int(rng.integers(2, 4))
        c = np.zeros(I); c[rng.choice(I, k, replace=False)] = 1.0
        if tuple(c) not in seen:
            seen.add(tuple(c)); cols.append(c)
    K = np.stack(cols, axis=1)
    alpha = np.round(rng.uniform(0.7, 1.3, J), 2); beta = np.round(rng.uniform(0.8, 1.4, J), 2)
    h = np.round(rng.uniform(0.8, 2.0, J), 2); eta = -np.round(rng.uniform(0.3, 0.7, J), 2)
    return RSN(K, alpha, beta, h, eta=eta, name="J18 (6x18)")


if __name__ == "__main__":
    net = make_j18()
    print(net)
    t0 = time.time(); tab = OfflineTables(net, gamma_c=0.98, gamma_b=0.98); t_lpc = time.time() - t0
    t0 = time.time(); tab.enumerate_vb(); t_lpb = time.time() - t0
    print(tab.summary())
    print(f"offline: LP-c/M/lambda~ {t_lpc:.0f}s + full v^b table {t_lpb:.0f}s = {t_lpc + t_lpb:.0f}s", flush=True)
    t0 = time.time(); nw = Net("J18", net.K, net.alpha, net.beta, net.h, net.eta, tables=tab)
    print(f"xtab {nw.xtab.shape}, {len(nw.V)} dual vertices (vertex-LP err {nw.vertex_err:.1e}): {time.time()-t0:.0f}s")
    print("rho*/4 =", net.rho_star / 4, " r_min =", net.r_min_heavy_traffic())
    res = []
    for seed in (1, 2):
        t0 = time.time(); o = nw.run(300, 10, 0.1, 60, 0.24, seed); o["seconds"] = time.time() - t0; res.append(o)
        print(f"r=300 T=10 c=(0.1,60,0.24) seed={seed}: {o['seconds']:.1f}s {o['events']:.2e} events JE={o['JE']:.3f} "
              f"hgi_path={o['hgi_path']:.3f} E_frac={o['E_frac']:.3f} halves {o['JE_1st']:.2f}->{o['JE_2nd']:.2f} "
              f"max idle_drift/|theta|={max(o['idle_drift']):.2f}", flush=True)
    json.dump(dict(spec=dict(K=net.K.astype(int).tolist(), alpha=net.alpha.tolist(), beta=net.beta.tolist(),
                             h=net.h.tolist(), eta=net.eta.tolist()),
                   offline_seconds=dict(lp_c=t_lpc, lp_b=t_lpb), b_max=tab.b_max, lam_tilde=tab.lam_tilde,
                   n_M=len(tab.M_set), n_lp_c=tab.n_lp_c, n_vertices=len(nw.V), runs=res),
              open("results/j18_e2e.json", "w"), indent=1)
    print("saved results/j18_e2e.json")
