"""Test networks and the rule that picks c2 from the primitives (no simulation tuning)."""
import math
import numpy as np
from gensim import Net
from flagmodel import g_flag

K2 = [[1, 0, 1], [0, 1, 1]]
K35 = [[1, 0, 0, 1, 0], [0, 1, 0, 1, 1], [0, 0, 1, 0, 1]]
# Network G: Example 3 of Budhiraja and Johnson (2020), Math. Oper. Res. 45(3), p. 812 (Example 5.6 of arXiv:1710.09042).
# (The dictionary key below is the label by which every stored result file identifies this network; cite the
# example as above.) I = 3, J = 6,
# job types chi_1, chi_2, chi_3, chi_{1,2}, chi_{2,3}, chi_{1,2,3} (chi_S uses the resources in S),
# mu = 1 for all types, h = (5, 5, 5, 7, 8, 11). The 2020 paper's viable-ranking Condition 4.4 fails
# (the set O is empty for chi_{1,2}, chi_{2,3}, chi_{1,2,3}), so its policy does not apply; the 2024
# construction needs only Conditions 1-3. Not given in the paper: arrival rates and eta; we take
# alpha = 1 (rho = 1, C = (3, 4, 3)) and eta = -1/2 as in the toy, so theta = K eta = (-1.5, -2, -1.5).
K36 = [[1, 0, 0, 1, 0, 1], [0, 1, 0, 1, 1, 1], [0, 0, 1, 0, 1, 1]]
SPECS = {
    "A toy (Ex. 2.2)":         (K2, [1, 1, 1], [1, 1, 1], [1, 1, 1], [-0.5] * 3),
    "B cost-trivial (Ex. 2.1)": (K2, [1, 1, 1], [1, 1, 1], [1, 1, 2], [-0.5] * 3),
    "C shared type costly":    (K2, [1, 1, 1], [1, 1, 1], [1, 1, 5], [-0.5] * 3),
    "D unequal rates":         (K2, [0.7, 1.0, 1.2], [1.0, 1.3, 1.0], [1.5, 1.0, 1.2], [-0.4, -0.5, -0.6]),
    "E unequal loads, costs":  (K2, [1.0, 0.5, 0.8], [1, 1, 1], [2.0, 1.0, 1.5], [-0.3, -0.6, -0.5]),
    "F 3x5 network":           (K35, [1] * 5, [1] * 5, [3, 1, 3, 1, 1], [-0.5] * 5),
    "G ranking-map fails (BJ20 Ex. 6.6)": (K36, [1] * 6, [1] * 6, [5, 5, 5, 7, 8, 11], [-0.5] * 6),
}
# a 2x3 network whose push b_max is small (not simulated; prediction only)
SLOW = {"D0 unequal rates, small b_max": (K2, [0.6, 1.2, 0.9], [1.5, 1.0, 2.0], [1.0, 1.5, 1.2], [-0.4, -0.6, -0.5])}
C1, KAPPA, R0, TARGET, F_OCC = 0.1, 0.24, 1000.0, 0.05, 0.5


def build(name, vb_cap="uniform", gamma=0.98):
    """vb_cap="per_pattern" builds the LABELED EXPERIMENT tables (outside eq. (6));
    every output derived from such a Net must carry nw.tab.label."""
    spec = SPECS.get(name) or SLOW[name]
    if vb_cap == "uniform":
        return Net(name, *spec, gamma=gamma)
    from offline import OfflineTables
    from rsn import RSN
    net = RSN(*spec, name=name)
    tab = OfflineTables(net, gamma_c=gamma, gamma_b=gamma, vb_cap=vb_cap)
    return Net(name, *spec, gamma=gamma, tables=tab)


def floor_drift(nw, r, U, f=1.0):
    """Predicted hysteresis idle drift / |theta_i| (max over resources), flag model with
    the worst-case service rate x_bar_j over patterns with z_j = 1 (conservative).
    f: near-empty occupancy P(Q_j < U, E_j = 0), a scalar or one value per type."""
    n = nw.net
    L = C1 * n.beta.min() * r ** KAPPA
    a = n.alphas_r(r)
    f = np.broadcast_to(np.asarray(f, float), (n.J,))
    per_j = np.zeros(n.J)
    for j in range(n.J):
        xb = max(nw.xtab[i, j] for i in range(2 ** n.J) if (i >> j) & 1)
        d = n.beta[j] * xb
        per_j[j] = f[j] * r * xb * g_flag(L, U, a[j], d) if d < a[j] else np.inf
    return float(np.max((n.K @ per_j) / np.abs(n.theta)))


def rule_c2(nw, f=F_OCC, target=TARGET, r0=R0):
    """Smallest c2 whose predicted floor drift at r0 = 1000 is <= target (5%) of |theta|
    (flag model; occupancy f = 0.5 as measured on the toy, or per-type values, e.g.
    ssc_occupancy(nw))."""
    bm = nw.net.beta.min()
    for U in np.arange(5.0, 20000.0, 1.0):
        if floor_drift(nw, r0, U, f) <= target:
            return U / (bm * r0 ** KAPPA)
    return np.nan


def ssc_occupancy(nw, n_samples=3000, T=20000.0, dt=1e-2, seed=0):
    """
    f_j = P(q*_j(W) = 0) under the stationary limit RBM, where q*(W) = argmin{h.q : K M q = W,
    q >= 0} is the cost-minimizing configuration. State-space collapse keeps Q^r/r near q*(W), so
    f_j is the fraction of time type j is near empty -- the occupancy that multiplies the flag
    formula. Toy: q* = (w1 - m, w2 - m, m), m = min(w1, w2), so f = (1/2, 1/2, 0) exactly, which is
    the 0.5 measured on the toy; on other networks it can be far smaller (D: local types rarely
    empty), which is why the f = 0.5 rule is conservative there. Monte Carlo: exact Skorokhod map
    on a dt-grid, 10% burn-in, n_samples LPs.
    """
    from scipy.optimize import linprog
    n = nw.net
    Lc = np.linalg.cholesky(n.Sigma)
    rng = np.random.default_rng(seed)
    steps = int(T / dt)
    dX = rng.standard_normal((steps, n.I)) @ Lc.T * np.sqrt(dt) + n.theta * dt
    A = np.cumsum(dX, axis=0)
    W = A - np.minimum(np.minimum.accumulate(A, axis=0), 0.0)
    burn = steps // 10
    idx = np.linspace(burn, steps - 1, n_samples).astype(int)
    KM = n.K @ n.M
    zero = np.zeros(n.J)
    for w in W[idx]:
        res = linprog(c=n.h, A_eq=KM, b_eq=w, bounds=[(0, None)] * n.J, method="highs")
        zero += res.x < 1e-9
    return zero / len(idx)
