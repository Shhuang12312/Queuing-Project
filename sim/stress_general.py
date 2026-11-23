"""Stress test of rsn_flat on general networks."""
import itertools, json, sys, time
import numpy as np
import os as _os
FLAT = sys.argv[1] if len(sys.argv) > 1 else _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "rsn_flat")
sys.path.insert(0, FLAT)            # path to the rsn_flat copy
from rsn import RSN, make_2lln, make_3res_5type
from offline import OfflineTables
from policy import GeneralThresholdPolicy
from simulation import gillespie_ergodic, hhat_batch, benchmark_hgi_e
from je import run_JE

def header(s): print(f"\n=== {s} ===")

# A. toy objects
header("A. toy (Example 2.2), gamma_c=gamma_b=0.98")
tab = OfflineTables(make_2lln(), gamma_c=0.98, gamma_b=0.98)
print(tab.summary())
print("v^c(0,0,0) =", np.round(tab.vc((0, 0, 0)), 4))
print("max ||v^b||_2 =", round(max(np.linalg.norm(tab.vb(z)) for z in itertools.product((0, 1), repeat=3)), 4))

# B. trivial case
header("B. Example 2.1 (h=(1,1,2)) trivial case")
t21 = OfflineTables(make_2lln(h=(1, 1, 2)), gamma_c=0.98, gamma_b=0.98)
print("trivial:", t21.trivial, " M:", t21.M_set, " b_max:", t21._get_b_max())

# C. 3-resource / 5-type
header("C. 3res5type (dim ker K = 2, first case with C^h_K nonzero)")
n35 = make_3res_5type()
t35 = OfflineTables(n35, gamma_c=0.98, gamma_b=0.98)
print(t35.summary())
print("|M| =", len(t35.M_set), "of", 2 ** 5, "; Sigma =\n", n35.Sigma)

# D. random heterogeneous networks
header("D. random heterogeneous networks (offline pass + admissibility)")
def random_network(rng, I, extra):
    cols = [np.eye(I)[:, i] for i in range(I)]
    for _ in range(extra):
        while True:
            c = (rng.random(I) < 0.5).astype(float)
            if c.sum() >= 1: break
        cols.append(c)
    K = np.stack(cols, axis=1); J = K.shape[1]
    return RSN(K, rng.uniform(.5, 2, J), rng.uniform(.5, 2, J),
               rng.uniform(.5, 3, J), eta=-rng.uniform(.2, 1, J), name="rand")
fails = 0
for seed in range(20):
    rng = np.random.default_rng(seed)
    net = random_network(rng, int(rng.integers(2, 5)), int(rng.integers(1, 4)))
    try:
        t = OfflineTables(net, gamma_c=0.98, gamma_b=0.98)
        bm = t._get_b_max()
        ok = all(np.linalg.norm(t.vb(z)) <= bm + 1e-12 for z in itertools.product((0, 1), repeat=net.J))
        print(f"seed {seed:2d}: I={net.I} J={net.J} trivial={t.trivial} |M|={len(t.M_set):3d} "
              f"b_max={bm:.4f} eq6_ok={ok}")
    except AssertionError as e:
        fails += 1; print(f"seed {seed:2d}: ASSERTION {e}")
print("assertion failures:", fails)

# E. lazy mode (what happens for J > max_enumerate, default 16)
header("E. large-J path: same toy with max_enumerate=2 -- b_max must equal the enumerated one")
lazy = OfflineTables(make_2lln(), gamma_c=0.98, gamma_b=0.98, max_enumerate=2)
vb = lazy.vb((1, 1, 1))
print(f"lazy b_max = {lazy._get_b_max():.4f}  (enumerated: {tab._get_b_max():.4f})")
print(f"lazy ||v^b(1,1,1)|| = {np.linalg.norm(vb):.4f} -> within eq.(6) cap 0.0408: "
      f"{np.linalg.norm(vb) <= tab._get_b_max() + 1e-12}")

# F. exact reproduction of a sweep entry
header("F. reproduce convergence_results.json entry faithful|r=50|seed=1")
store = json.load(open(FLAT + "/convergence_results.json"))
ref = store["faithful|r=50|seed=1"]
HH = lambda W: np.maximum(W[:, 0], W[:, 1])
t0 = time.time()
net = make_2lln()
out = run_JE(net, r=50.0, T=20.0, c1=0.5, c2=10.0, kappa=0.22, seed=1,
             n_snapshots=1200, hhat_fn=HH)
level = float(HH((out["snapshots"] @ (net.K @ net.M).T) / 50.0).mean())
for k, mine in [("JE", out["JE"]), ("gap", out["ssc_gap"]), ("E_frac", out["E_frac"]), ("level", level)]:
    print(f"{k:6s} stored={ref[k]!r:24} rerun={mine!r:24} exact={ref[k] == mine}")
print(f"({time.time() - t0:.0f}s)")

# G. dual-vertex h_hat vs LP on 3res5type
header("G. exact vectorized h_hat via dual vertices (3res5type)")
def hhat_vertices(net):
    A = (net.K @ net.M).T                  # J x I ; dual: max y.w s.t. A y <= h
    V = []
    for rows in itertools.combinations(range(net.J), net.I):
        B = A[list(rows)]
        if abs(np.linalg.det(B)) < 1e-12: continue
        y = np.linalg.solve(B, net.h[list(rows)])
        if np.all(A @ y <= net.h + 1e-10): V.append(y)
    V = np.unique(np.round(np.array(V), 12), axis=0)
    return V, (lambda W: (np.atleast_2d(W) @ V.T).max(axis=1))
V, hv = hhat_vertices(n35)
W = np.random.default_rng(0).uniform(0, 10, (300, 3))
err = np.max(np.abs(hv(W) - hhat_batch(W, n35)))
print(f"{len(V)} dual vertices; max |vertex - LP| over 300 points = {err:.2e}")

# H. end-to-end run on 3res5type with the general h_hat
header("H. short faithful run on 3res5type (r=30, T=5)")
t0 = time.time()
o = run_JE(n35, r=30, T=5, c1=0.1, c2=30, kappa=0.24, seed=1, n_snapshots=400,
           hhat_fn=hv, tables=t35)
print(f"J_E={o['JE']:.3f}±{o['JE_ci']:.3f} gap={o['ssc_gap']:.4f} E_frac={o['E_frac']:.4f} "
      f"({time.time() - t0:.0f}s)")
m, se = benchmark_hgi_e(n35, hhat_fn=hv, dt=2e-3, T=2000.0, seeds=(0, 1))
print(f"HGI_E(3res5type) ~ {m:.3f} ± {se:.3f} (fixed dt, unextrapolated)")
