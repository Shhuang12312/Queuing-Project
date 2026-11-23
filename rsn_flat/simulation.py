# Standalone module with plain imports; pure Python. The compiled kernels are in sim/.

"""
simulation.py — exact CTMC simulation, HGI benchmark, and estimators.

Scaling conventions:
  * the rth system runs with O(1) rates alpha^r = alpha + beta*eta/r —
    rates are NEVER multiplied by r;
  * the diffusion clock is t -> r^2 t: a horizon of T diffusion units is
    r^2*T physical time units;
  * read-outs are Q/r (queues) and W = K M Q / r (workloads).

The model under the policy is a CTMC (between events the allocation is
constant), so Gillespie simulation is exact — no Euler/tau-leap error.

h_hat(w) = min{ h.q : q >= 0, K M q = w } is evaluated EXACTLY and vectorized
through the dual: h_hat(w) = max_k V_k . w over the vertices V of the dual
polyhedron {y : (KM)^T y <= h} (the primal is feasible for every w >= 0 by
Condition 3, so strong duality applies and the maximum is attained at a
vertex). The vertices are enumerated once per network (C(J, I) candidate
bases); when that count exceeds `MAX_VERTEX_COMBINATIONS` the code falls back
to one LP per snapshot (never per event). Toy: V = {(1,0), (0,1)}, so
h_hat(w) = max(w_1, w_2); Example 2.1 (h = (1,1,2)): V = {(1,1)}, h_hat = w_1 + w_2.
"""

import math
import itertools
import numpy as np
from scipy.optimize import linprog

MAX_VERTEX_COMBINATIONS = 200_000


# ---------------------------------------------------------------------------
# HGI ideal cost  h_hat(w)
# ---------------------------------------------------------------------------
def hhat_lp(w, rsn):
    """LP: min h.q s.t. K M q = w, q >= 0 (reference implementation, one LP)."""
    res = linprog(c=rsn.h, A_eq=rsn.K @ rsn.M, b_eq=np.asarray(w, float),
                  bounds=[(0, None)] * rsn.J, method="highs")
    if res.success:
        return float(res.fun)
    # infeasible only from numerical round-off of w; clip and retry once
    res = linprog(c=rsn.h, A_eq=rsn.K @ rsn.M,
                  b_eq=np.maximum(np.asarray(w, float), 0.0),
                  bounds=[(0, None)] * rsn.J, method="highs")
    return float(res.fun) if res.success else float("nan")


def hhat_batch_lp(W, rsn):
    """LP h_hat for each row of W (n x I)."""
    return np.array([hhat_lp(w, rsn) for w in np.atleast_2d(W)])


def dual_vertices(rsn, max_combinations=MAX_VERTEX_COMBINATIONS):
    """
    Vertices of the dual polyhedron {y in R^I : (KM)^T y <= h}, as an (n_v x I)
    array, or None when C(J, I) > max_combinations. For w >= 0,
    h_hat(w) = max_k V_k . w  (exact).
    """
    if math.comb(rsn.J, rsn.I) > max_combinations:
        return None
    A = (rsn.K @ rsn.M).T                    # J x I: dual constraints A y <= h
    V = []
    for rows in itertools.combinations(range(rsn.J), rsn.I):
        B = A[list(rows)]
        if abs(np.linalg.det(B)) < 1e-12:
            continue
        y = np.linalg.solve(B, rsn.h[list(rows)])
        if np.all(A @ y <= rsn.h + 1e-10):
            V.append(y)
    return np.unique(np.round(np.array(V), 12), axis=0)


def make_hhat(rsn, max_combinations=MAX_VERTEX_COMBINATIONS):
    """
    Vectorized ideal cost W (n x I) -> (n,). Exact dual-vertex formula when the
    vertex enumeration is affordable, else the LP fallback. The result is cached
    on the network object (`rsn._hhat_fn`); `fn.method` is "vertices" or "lp".
    """
    fn = getattr(rsn, "_hhat_fn", None)
    if fn is not None:
        return fn
    V = dual_vertices(rsn, max_combinations)
    if V is None:
        def fn(W):
            return hhat_batch_lp(W, rsn)
        fn.method, fn.vertices = "lp", None
    else:
        VT = np.ascontiguousarray(V.T)

        def fn(W):
            return (np.atleast_2d(np.asarray(W, float)) @ VT).max(axis=1)
        fn.method, fn.vertices = "vertices", V
    rsn._hhat_fn = fn
    return fn


def hhat(w, rsn):
    """h_hat(w) for one workload vector (exact; dual vertices, LP fallback)."""
    return float(make_hhat(rsn)(np.asarray(w, float).reshape(1, -1))[0])


def hhat_batch(W, rsn):
    """h_hat for each row of W (n x I) — exact dual-vertex formula by default."""
    return np.asarray(make_hhat(rsn)(W), float)


# ---------------------------------------------------------------------------
# Exact Gillespie simulation with time-weighted ergodic accumulators
# ---------------------------------------------------------------------------
def gillespie_ergodic(rsn, pol, r, T_diff=20.0, burn_frac=0.3,
                      n_snapshots=2000, q0=None, rng=None):
    """
    Simulate the rth CTMC under `pol` to physical time r^2*T_diff.

    Returns dict with:
      JE        : time average of h.Q/r over the post-burn-in window (exact,
                  time-weighted — zero sampling variance given the path)
      HGI_path  : mean of h_hat(K M Q/r) over n_snapshots uniform times
                  (same-path ideal cost; JE - HGI_path is the SSC gap)
      ssc_gap   : JE_snap - HGI_path computed on the SAME snapshots (paired)
      E_frac    : fraction of (post-burn-in) time some E_j = 1  (idling diag.)
      util      : time-average of K b / C (per-resource utilization diag.)
      snapshots : (n_snapshots x J) queue array, for any further analysis
    """
    rng = np.random.default_rng() if rng is None else rng
    rsn.check_r(r)                                   # |rho^r - rho| < rho*/4, raises
    if hasattr(pol, "check_r"):
        pol.check_r(r)                               # (c2~ - c1~) r^kappa > 1, raises
    alpha_r, beta_r = rsn.alphas_r(r), rsn.betas_r(r)
    J = rsn.J
    T_end  = (r ** 2) * float(T_diff)
    T_burn = float(burn_frac) * T_end
    snap_times = np.linspace(T_burn, T_end, n_snapshots, endpoint=False)

    q = np.zeros(J) if q0 is None else np.asarray(q0, float).copy()
    if hasattr(pol, "reset_E"):
        pol.reset_E(0.0)

    t = 0.0
    int_cost = 0.0
    int_Eon  = 0.0
    int_util = np.zeros(rsn.I)
    snaps = np.empty((n_snapshots, J))
    next_snap = 0
    h = rsn.h

    while t < T_end:
        b, Z, E = pol.allocation(q, r)
        dep = beta_r * b * (q > 0)
        rates = np.concatenate([alpha_r, dep])
        total = rates.sum()
        t_next = t + rng.exponential(1.0 / total)

        # snapshots falling in the holding interval [t, t_next)
        while next_snap < n_snapshots and snap_times[next_snap] < t_next:
            snaps[next_snap] = q
            next_snap += 1

        # time-weighted accumulators over the post-burn-in overlap
        overlap = min(t_next, T_end) - max(t, T_burn)
        if overlap > 0:
            int_cost += overlap * float(h @ q)
            int_util += overlap * (rsn.K @ b)
            if np.any(E > 0):
                int_Eon += overlap

        t = t_next
        if t >= T_end:
            break
        # inverse-CDF event draw (faster than rng.choice with p=)
        k = int(np.searchsorted(np.cumsum(rates), rng.random() * total))
        if k < J:
            q[k] += 1.0
        else:
            k -= J
            if q[k] > 0:
                q[k] -= 1.0

    while next_snap < n_snapshots:          # horizon edge
        snaps[next_snap] = q
        next_snap += 1

    window = T_end - T_burn
    W_hat = (snaps @ (rsn.K @ rsn.M).T) / r
    hgi_path = hhat_batch(W_hat, rsn)
    je_snap  = (snaps @ h) / r
    return dict(
        JE=int_cost / window / r,
        HGI_path=float(hgi_path.mean()),
        ssc_gap=float((je_snap - hgi_path).mean()),
        E_frac=int_Eon / window,
        util=int_util / window / rsn.C,
        snapshots=snaps,
        n_violations=getattr(pol, "n_violations", 0),
    )


# ---------------------------------------------------------------------------
# RBM benchmark (limit process)
# ---------------------------------------------------------------------------
def rbm_ergodic_hgi(rsn, T=3000.0, dt=0.01, burn_frac=0.1, thin=None, rng=None):
    """
    Long-run average of h_hat(W) for dW = theta dt + Sigma^{1/2} dB reflected
    componentwise in R^I_+ by REFLECTED EULER (clip at 0 each step). Kept for
    reference only. On a grid it is the same scheme as the Skorokhod map of the
    discretized path, and it is biased low by O(sqrt(dt)) (-4.2% at dt = 0.01 on
    the toy, sim/results/benchmark_bias.json). Use `benchmark_hgi_e` (sqrt(dt)-extrapolated, both legs on one
    Brownian path) for any quoted number.
    """
    rng = np.random.default_rng() if rng is None else rng
    L = np.linalg.cholesky(rsn.Sigma)
    steps = int(T / dt)
    burn = int(burn_frac * steps)
    thin = max(1, (steps - burn) // 5000) if thin is None else thin

    w = np.zeros(rsn.I)
    sq = np.sqrt(dt)
    kept = []
    for k in range(steps):
        w = np.maximum(w + rsn.theta * dt + L @ (sq * rng.standard_normal(rsn.I)), 0.0)
        if k >= burn and (k - burn) % thin == 0:
            kept.append(w.copy())
    vals = hhat_batch(np.asarray(kept), rsn)
    return float(vals.mean()), np.asarray(kept)


# ---------------------------------------------------------------------------
# Uniform-snapshot Monte Carlo estimator of J_E (multi-path version)
# ---------------------------------------------------------------------------
def estimate_JE_uniform(rsn, pol_factory, r, tau=20.0, n_paths=4,
                        n_snap_per_path=500, burn_frac=0.3, rng=None):
    """
    J_E estimate from uniform time snapshots on n_paths INDEPENDENT paths
    (fixes the correlated-single-path SE problem: the SE is computed across
    path means, which are independent).
    """
    rng = np.random.default_rng() if rng is None else rng
    means = []
    for _ in range(n_paths):
        out = gillespie_ergodic(rsn, pol_factory(), r, T_diff=tau,
                                burn_frac=burn_frac,
                                n_snapshots=n_snap_per_path, rng=rng)
        means.append(out["JE"])
    means = np.asarray(means)
    se = means.std(ddof=1) / np.sqrt(len(means)) if len(means) > 1 else float("nan")
    return float(means.mean()), float(se), means


# ---------------------------------------------------------------------------
# High-precision ergodic HGI benchmark (exact componentwise reflection map)
# ---------------------------------------------------------------------------
def _rbm_mean_hhat(rsn, hhat_fn, dt, T, seed, n_lp=5000, chunk=10_000_000):
    """Time average of hhat_fn(W) for one RBM path at step dt (exact Skorokhod map)."""
    L = np.linalg.cholesky(rsn.Sigma)
    rng = np.random.default_rng(seed)
    steps = int(T / dt)
    thin = 1 if getattr(hhat_fn, "method", "vertices") != "lp" else max(1, steps // n_lp)
    A_level = np.zeros(rsn.I)
    run_min = np.zeros(rsn.I)
    acc, n, done = 0.0, 0, 0
    kept = []
    while done < steps:
        m = min(chunk, steps - done)
        dX = rng.standard_normal((m, rsn.I)) @ L.T * np.sqrt(dt) + rsn.theta * dt
        A = A_level + np.cumsum(dX, axis=0)
        Mn = np.minimum(np.minimum.accumulate(A, axis=0), run_min)
        W = A - Mn
        if thin == 1:
            acc += float(np.sum(hhat_fn(W))); n += m
        else:
            idx = np.arange((-done) % thin, m, thin)
            kept.append(W[idx])
        A_level, run_min = A[-1], Mn[-1]
        done += m
    if thin == 1:
        return acc / n
    return float(np.mean(hhat_fn(np.vstack(kept))))


def _rbm_mean_hhat_pair(rsn, hhat_fn, dt, k, T, seed, n_lp=5000, chunk=10_000_000):
    """Time averages of hhat_fn(W) for ONE Brownian path sampled at step dt and at step k*dt (the coarse
    increments are the fine ones summed in blocks of k), each reflected by the per-coordinate Skorokhod map
    of its own grid. Returns (fine, coarse). The random numbers are consumed exactly as in
    `_rbm_mean_hhat(rsn, hhat_fn, dt, T, seed)`, so `fine` equals that function's value bit for bit."""
    L = np.linalg.cholesky(rsn.Sigma)
    rng = np.random.default_rng(seed)
    steps = int(T / dt)
    lp = getattr(hhat_fn, "method", "vertices") == "lp"
    thin = (max(1, steps // n_lp), max(1, steps // k // n_lp)) if lp else (1, 1)
    chunk -= chunk % k
    st = [dict(A=np.zeros(rsn.I), mn=np.zeros(rsn.I), acc=0.0, n=0, kept=[], done=0) for _ in range(2)]
    done = 0
    while done < steps:
        m = min(chunk, steps - done)
        dXf = rng.standard_normal((m, rsn.I)) @ L.T * np.sqrt(dt) + rsn.theta * dt
        mc = m // k                                           # a ragged tail (< k fine steps) is dropped from the coarse leg
        for lev, dX in enumerate((dXf, dXf[:mc * k].reshape(mc, k, rsn.I).sum(axis=1))):
            if len(dX) == 0:
                continue
            q = st[lev]
            A = q["A"] + np.cumsum(dX, axis=0)
            Mn = np.minimum(np.minimum.accumulate(A, axis=0), q["mn"])
            W = A - Mn
            if thin[lev] == 1:
                q["acc"] += float(np.sum(hhat_fn(W))); q["n"] += len(dX)
            else:
                q["kept"].append(W[np.arange((-q["done"]) % thin[lev], len(dX), thin[lev])])
            q["A"], q["mn"] = A[-1], Mn[-1]
            q["done"] += len(dX)
        done += m
    if not lp:
        return st[0]["acc"] / st[0]["n"], st[1]["acc"] / st[1]["n"]
    return tuple(float(np.mean(hhat_fn(np.vstack(q["kept"])))) for q in st)


def benchmark_hgi_e(rsn, hhat_fn=None, dt=(4e-3, 1e-3), T=20000.0, seeds=(0, 1, 2, 3, 4, 5, 6, 7),
                    n_lp=5000, return_details=False):
    """
    HGI_E = E_pi[hhat(W)] for the limit RBM dW = theta dt + Sigma^{1/2} dB reflected
    componentwise in R^I_+, by the per-coordinate Skorokhod map on a discretized
    Brownian path (vectorized running minima). On a grid this map is the same scheme
    as reflected Euler (the two agree to rounding error, sim/results/benchmark_bias.json);
    it misses the minima between grid points, so the time average is biased low by
    O(sqrt(dt)): -4.2% at dt = 1e-2, -1.3% at dt = 1e-3 on the toy.

    With `dt` a pair (dt_1 > dt_2, dt_1/dt_2 an integer) the estimate is
    sqrt(dt)-extrapolated seed by seed,
        e = e_2 + (e_2 - e_1) sqrt(dt_2)/(sqrt(dt_1) - sqrt(dt_2)),
    with BOTH legs evaluated on the SAME Brownian path (one path per seed at step dt_2;
    the dt_1 leg sums its increments in blocks), so the path noise cancels in e_2 - e_1
    and the SE of e is that of a single leg. (mean, SE) are taken across seeds. A scalar
    `dt` gives the plain fixed-step estimate. The variant that draws
    the two legs from different paths is `benchmark_hgi_e_independent_legs`.

    hhat_fn: vectorized ideal cost (n x I -> n); default is the exact
    dual-vertex formula of `make_hhat` (LP fallback thins to n_lp points).
    Toy (theta = -1, Sigma = [[4,2],[2,4]]), defaults (8 seeds): see
    sim/results/multi_bench_crn.json. Check: networks B and C have hhat(w) = w_1 + w_2,
    so HGI_E = sum_i Sigma_ii / (2 |theta_i|) = 4 exactly.
    """
    hhat_fn = make_hhat(rsn) if hhat_fn is None else hhat_fn
    if np.isscalar(dt):
        vals = np.array([_rbm_mean_hhat(rsn, hhat_fn, float(dt), T, s, n_lp) for s in seeds])
        per_dt = {float(dt): vals}
    else:
        dt1, dt2 = (float(d) for d in dt)
        if not dt1 > dt2 > 0:
            raise ValueError("dt must be a pair (dt_1 > dt_2 > 0) or a scalar")
        ratio = dt1 / dt2
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("dt_1 / dt_2 must be an integer (both legs use the same Brownian path)")
        legs = np.array([_rbm_mean_hhat_pair(rsn, hhat_fn, dt2, int(round(ratio)), T, s, n_lp) for s in seeds])
        e2, e1 = legs[:, 0], legs[:, 1]
        s1, s2 = math.sqrt(dt1), math.sqrt(dt2)
        vals = e2 + (e2 - e1) * s2 / (s1 - s2)
        per_dt = {dt1: e1, dt2: e2}
    mean = float(vals.mean())
    se = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else float("nan")
    if return_details:
        return mean, se, dict(per_seed=vals, per_dt=per_dt, seeds=tuple(seeds), T=T)
    return mean, se


def benchmark_hgi_e_independent_legs(rsn, hhat_fn=None, dt=(4e-3, 1e-3), T=20000.0, seeds=(0, 1, 2, 3),
                    n_lp=5000, return_details=False):
    """
    Benchmark with independent extrapolation legs (it produced the HGI_E values of
    sim/results/multi_bench.json, toy 2.81 +- 0.02). Use `benchmark_hgi_e` for any quoted number.
    Its two legs are drawn from DIFFERENT Brownian paths (the dt_1 leg and the dt_2 leg consume the seed's
    stream at different step sizes), so the extrapolation 2 e_2 - e_1 carries the noise of both legs: on
    networks B and C, where HGI_E = 4 exactly, it returned 4.069 +- 0.026.

    HGI_E = E_pi[hhat(W)] for the limit RBM dW = theta dt + Sigma^{1/2} dB reflected
    componentwise in R^I_+, by the per-coordinate Skorokhod map on a
    discretized Brownian path (vectorized running minima).

    The remaining discretization error of the time average is O(sqrt(dt)), so
    with `dt` a pair (dt_1 > dt_2) the estimate is sqrt(dt)-extrapolated
    seed by seed:  e = e_2 + (e_2 - e_1) sqrt(dt_2)/(sqrt(dt_1) - sqrt(dt_2)),
    and (mean, SE) are taken across seeds. A scalar `dt` gives the plain
    fixed-step estimate (biased low by ~2% at dt = 1e-3 on the toy).

    hhat_fn: vectorized ideal cost (n x I -> n); default is the exact
    dual-vertex formula of `make_hhat` (LP fallback thins to n_lp points).
    Toy (theta = -1, Sigma = [[4,2],[2,4]]): 2.81 +- 0.02 with the defaults.
    """
    hhat_fn = make_hhat(rsn) if hhat_fn is None else hhat_fn
    if np.isscalar(dt):
        vals = np.array([_rbm_mean_hhat(rsn, hhat_fn, float(dt), T, s, n_lp) for s in seeds])
        per_dt = {float(dt): vals}
    else:
        dt1, dt2 = (float(d) for d in dt)
        if not dt1 > dt2 > 0:
            raise ValueError("dt must be a pair (dt_1 > dt_2 > 0) or a scalar")
        e1 = np.array([_rbm_mean_hhat(rsn, hhat_fn, dt1, T, s, n_lp) for s in seeds])
        e2 = np.array([_rbm_mean_hhat(rsn, hhat_fn, dt2, T, s, n_lp) for s in seeds])
        s1, s2 = math.sqrt(dt1), math.sqrt(dt2)
        vals = e2 + (e2 - e1) * s2 / (s1 - s2)
        per_dt = {dt1: e1, dt2: e2}
    mean = float(vals.mean())
    se = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else float("nan")
    if return_details:
        return mean, se, dict(per_seed=vals, per_dt=per_dt, seeds=tuple(seeds), T=T)
    return mean, se


# ---------------------------------------------------------------------------
# Discounted HGI benchmark (limit process, from a given initial workload)
# ---------------------------------------------------------------------------
def _rbm_discounted_hhat(rsn, hhat_fn, w0, varsigma, dt, T, n_paths, seed, chunk=2000, coarse=None):
    """Mean over paths of int_0^T e^{-varsigma t} h_hat(W(t)) dt (trapezoid on the dt-grid) for the RBM
    dW = theta dt + Sigma^{1/2} dB started at w0, reflected componentwise by the exact 1-d Skorokhod map
    of the discretized path. Returns (mean, SE over paths); with `coarse` = k (integer) it also evaluates the
    same Brownian paths on the k-times coarser grid (common random numbers for the sqrt(dt) extrapolation) and
    returns (mean_fine, se_fine, mean_coarse, se_coarse, se_of_difference)."""
    L = np.linalg.cholesky(rsn.Sigma)
    n = int(round(T / dt))
    if coarse is not None and n % coarse:
        raise ValueError("T/dt must be a multiple of the coarsening factor")

    def weights(step, npts):
        tg = np.arange(npts + 1) * step
        w = np.exp(-varsigma * tg) * step
        w[0] *= 0.5; w[-1] *= 0.5
        return w

    def reflect_cost(dX, k):
        X = w0 + np.concatenate([np.zeros((k, 1, rsn.I)), np.cumsum(dX, axis=1)], axis=1)
        W = X - np.minimum(np.minimum.accumulate(X, axis=1), 0.0)
        return hhat_fn(W.reshape(-1, rsn.I)).reshape(k, -1)

    wf = weights(dt, n)
    wc = weights(dt * coarse, n // coarse) if coarse else None
    rng = np.random.default_rng(seed)
    fine, crs = [], []
    for c0 in range(0, n_paths, chunk):
        k = min(chunk, n_paths - c0)
        dX = rng.standard_normal((k, n, rsn.I)) @ L.T * math.sqrt(dt) + rsn.theta * dt
        fine.append(reflect_cost(dX, k) @ wf)
        if coarse:
            crs.append(reflect_cost(dX.reshape(k, n // coarse, coarse, rsn.I).sum(axis=2), k) @ wc)
    f = np.concatenate(fine)
    if not coarse:
        return float(f.mean()), float(f.std(ddof=1) / math.sqrt(len(f)))
    c = np.concatenate(crs); d = f - c
    return (float(f.mean()), float(f.std(ddof=1) / math.sqrt(len(f))), float(c.mean()),
            float(c.std(ddof=1) / math.sqrt(len(c))), float(d.std(ddof=1) / math.sqrt(len(d))))


def benchmark_hgi_d(rsn, w0=None, varsigma=1.0, hhat_fn=None, dt=(1e-3, 5e-4), T=20.0, n_paths=10000,
                    seed=12345, return_details=False):
    """
    HGI_D(w0) = int_0^oo e^{-varsigma t} E h_hat(W(t)) dt  (paper eq. (11), unnormalized: no factor varsigma)
    for the limit RBM started at w0 (default 0, the empty system), by the exact per-coordinate Skorokhod map
    on a discretized Brownian path, truncated at T (tail e^{-varsigma T} E h_hat ~ 1e-8 at the defaults).

    As for the ergodic benchmark the fixed-dt estimate is biased low by O(sqrt(dt)); with `dt` a pair
    (dt_1 > dt_2, dt_1/dt_2 an integer) the result is sqrt(dt)-extrapolated,
    e = e_2 + (e_2 - e_1) sqrt(dt_2)/(sqrt(dt_1) - sqrt(dt_2)), with BOTH legs evaluated on the same Brownian
    paths (the coarse path is the fine one summed in blocks), so the extrapolation term's noise is the SE of the
    paired difference, not 2.4x the leg SE. A scalar `dt` gives the plain estimate.
    Toy, w0 = 0, varsigma = 1, defaults: see sim/results/hgi_d_toy.json (recorded value 1.33).
    Note: every queueing-side J_D in this project uses varsigma = 1 (toykern3.c hard-codes it); this routine is
    the only place varsigma != 1 can be evaluated.
    """
    hhat_fn = make_hhat(rsn) if hhat_fn is None else hhat_fn
    w0 = np.zeros(rsn.I) if w0 is None else np.asarray(w0, float).reshape(rsn.I)
    if np.any(w0 < 0) or varsigma <= 0:
        raise ValueError("w0 must be >= 0 and varsigma > 0")
    if np.isscalar(dt):
        m, se = _rbm_discounted_hhat(rsn, hhat_fn, w0, varsigma, float(dt), T, n_paths, seed)
        det = {float(dt): (m, se)}
    else:
        dt1, dt2 = (float(d) for d in dt)
        if not dt1 > dt2 > 0:
            raise ValueError("dt must be a pair (dt_1 > dt_2 > 0) or a scalar")
        ratio = dt1 / dt2
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("dt_1 / dt_2 must be an integer (common random numbers across the two legs)")
        e2, s2e, e1, s1e, sde = _rbm_discounted_hhat(rsn, hhat_fn, w0, varsigma, dt2, T, n_paths, seed,
                                                     coarse=int(round(ratio)))
        s1, s2 = math.sqrt(dt1), math.sqrt(dt2)
        f = s2 / (s1 - s2)
        m = e2 + (e2 - e1) * f
        se = math.sqrt(s2e ** 2 + (f * sde) ** 2)
        det = {dt1: (e1, s1e), dt2: (e2, s2e), "se_difference": sde}
    if return_details:
        return m, se, dict(per_dt=det, varsigma=varsigma, w0=w0.tolist(), T=T, n_paths=n_paths)
    return m, se
