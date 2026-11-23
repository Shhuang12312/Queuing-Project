"""Exact dual-vertex h_hat as the default; sqrt(dt)-extrapolated benchmark with both extrapolation legs on the
same Brownian path."""
import math

import numpy as np
import pytest

from rsn import make_2lln, make_3res_5type
from simulation import (make_hhat, hhat, hhat_batch, hhat_batch_lp, dual_vertices,
                        benchmark_hgi_e, benchmark_hgi_e_independent_legs)
from nets import six_networks, twenty_random


@pytest.mark.parametrize("name,net", six_networks() + [("3res5type", make_3res_5type())]
                         + [(n.name + str(i), n) for i, n in enumerate(twenty_random()[:8])])
def test_vertex_hhat_equals_lp(name, net):
    fn = make_hhat(net)
    assert fn.method == "vertices" and len(fn.vertices) >= 1
    W = np.random.default_rng(0).uniform(0, 8, (400, net.I))
    assert np.max(np.abs(fn(W) - hhat_batch_lp(W, net))) < 1e-10
    assert np.array_equal(hhat_batch(W, net), fn(W))
    assert hhat(W[0], net) == fn(W[:1])[0]
    # every vertex is dual feasible: (KM)^T y <= h
    assert np.all((net.K @ net.M).T @ fn.vertices.T <= net.h[:, None] + 1e-9)


def test_paper_examples_analytic():
    toy = make_2lln()
    assert sorted(map(tuple, dual_vertices(toy).tolist())) == [(0.0, 1.0), (1.0, 0.0)]
    W = np.random.default_rng(1).uniform(0, 5, (100, 2))
    assert np.allclose(hhat_batch(W, toy), W.max(axis=1), atol=1e-12)     # h_hat = max(w1, w2)
    ex21 = make_2lln(h=(1, 1, 2))
    assert dual_vertices(ex21).tolist() == [[1.0, 1.0]]
    assert np.allclose(hhat_batch(W, ex21), W.sum(axis=1), atol=1e-12)    # h_hat = w1 + w2
    c = make_2lln(h=(1, 1, 5))                                             # shared type too costly
    assert np.allclose(hhat_batch(W, c), W.sum(axis=1), atol=1e-12)


def test_lp_fallback_when_enumeration_too_large():
    net = make_3res_5type(h=(1, 2, 1, 3, 2))
    net._hhat_fn = None
    fn_lp = make_hhat(net, max_combinations=1)
    assert fn_lp.method == "lp"
    net._hhat_fn = None
    fn_v = make_hhat(net)
    W = np.random.default_rng(2).uniform(0, 8, (50, 3))
    assert np.max(np.abs(fn_lp(W) - fn_v(W))) < 1e-9


def test_toy_benchmark_2_76():
    """Default: dt = (4e-3, 1e-3), T = 20000, seeds 0-7,
    exact vertex h_hat, BOTH legs on the same Brownian path per seed (~25 s): 2.763 +- 0.020."""
    m, se, d = benchmark_hgi_e(make_2lln(), return_details=True)
    assert abs(m - 2.7628) < 2e-3 and 0.01 < se < 0.03 and len(d["per_seed"]) == 8
    e1, e2 = d["per_dt"][4e-3], d["per_dt"][1e-3]
    s1, s2 = math.sqrt(4e-3), math.sqrt(1e-3)
    assert abs(m - (e2.mean() + (e2.mean() - e1.mean()) * s2 / (s1 - s2))) < 1e-12
    assert np.all(e1 < e2) and np.all(e2 < d["per_seed"])   # every seed: grid averages are biased low, O(sqrt dt)
    assert abs(e2.mean() / m - 0.987) < 0.003          # 1.3% low at dt = 1e-3 on the toy
    assert abs(e1.mean() / m - 0.973) < 0.003          # 2.7% low at dt = 4e-3
    # common path: the leg difference is nearly noise-free (its spread across seeds is far below a leg's)
    assert (e2 - e1).std(ddof=1) < 0.05 * e2.std(ddof=1)
    # scalar dt = the plain fixed-step path, same RNG consumption as the fine leg above
    m_fixed, _ = benchmark_hgi_e(make_2lln(), dt=1e-3, T=20000.0, seeds=(0,))
    assert m_fixed == e2[0]
    with pytest.raises(ValueError):
        benchmark_hgi_e(make_2lln(), dt=(1e-3, 4e-3), T=100.0, seeds=(0,))
    with pytest.raises(ValueError):
        benchmark_hgi_e(make_2lln(), dt=(3e-3, 2e-3), T=100.0, seeds=(0,))      # ratio must be an integer


@pytest.mark.parametrize("h", [(1, 1, 2), (1, 1, 5)], ids=["B", "C"])
def test_benchmark_is_exact_on_networks_B_and_C(h):
    """Networks B and C: h_hat(w) = w_1 + w_2, and each coordinate of the RBM is a one-dimensional RBM with
    exponential stationary law of mean Sigma_ii / (2 |theta_i|) = 4 / 2, so HGI_E = 4 exactly.
    The common-path benchmark must agree within 2 SE (the independent-leg routine gives 4.069 +- 0.026)."""
    net = make_2lln(h=h)
    assert np.allclose(net.theta, [-1, -1]) and np.allclose(np.diag(net.Sigma), [4, 4])
    m, se = benchmark_hgi_e(net)
    assert abs(m - 4.0) < 2 * se and se < 0.04


def test_independent_leg_benchmark():
    """The routine behind multi_bench.json reproduces its toy value (4 seeds)."""
    m, se = benchmark_hgi_e_independent_legs(make_2lln())
    assert abs(m - 2.8094) < 1e-3 and abs(se - 0.0243) < 1e-3


def _hhat_G_closed_form(W):
    """Workload cost of network G (beta = 1), w = (w1, w2, w3) in the row order of K; returns (value, case)."""
    w1, w2, w3 = W.T
    c1 = w2 >= w1 + w3
    c2 = ~c1 & (w2 >= np.maximum(w1, w3))
    v3 = 5 * (w1 + w2 + w3) + np.minimum(np.minimum(w1, w2), w3) - 3 * np.minimum(w1, w2) - 2 * np.minimum(w2, w3)
    val = np.where(c1, 5 * w2 + 2 * w1 + 3 * w3, np.where(c2, 3 * w1 + 4 * w2 + 4 * w3, v3))
    return val, np.where(c1, 1, np.where(c2, 2, 3))


def test_network_G_hhat_closed_form():
    """Network G (Example 3 of Budhiraja and Johnson (2020), Math. Oper. Res. 45(3), p. 812): the code's h_hat equals
        5 w2 + 2 w1 + 3 w3                                             if w2 >= w1 + w3
        3 w1 + 4 w2 + 4 w3                                             if w1 + w3 > w2 >= max(w1, w3)
        5 (w1 + w2 + w3) + min(w1, w2, w3) - 3 min(w1, w2) - 2 min(w2, w3)   otherwise
    to 1e-9 at 1000 random w >= 0 (uniform on [0, 10]^3, seed 20261120), each case hit at least 50 times
    (counts with this seed: 159 / 160 / 681). Checked for the exact dual-vertex function (the one the compiled
    kernel's table is built from) and for the LP."""
    from rsn import RSN
    from nets import SEVEN
    K, alpha, beta, h, eta = SEVEN[[k for k in SEVEN if k.startswith("G ")][0]]
    assert beta == [1] * 6 and h == [5, 5, 5, 7, 8, 11]
    assert K == [[1, 0, 0, 1, 0, 1], [0, 1, 0, 1, 1, 1], [0, 0, 1, 0, 1, 1]]
    net = RSN(K, alpha, beta, h, eta=eta)
    W = np.random.default_rng(20261120).uniform(0, 10, (1000, 3))
    closed, case = _hhat_G_closed_form(W)
    counts = [int((case == c).sum()) for c in (1, 2, 3)]
    assert counts == [159, 160, 681] and min(counts) >= 50, counts
    fn = make_hhat(net)
    assert fn.method == "vertices"
    for name, val in (("dual vertices", fn(W)), ("LP", hhat_batch_lp(W, net))):
        err = np.abs(val - closed)
        i = int(np.argmax(err > 1e-9)) if np.any(err > 1e-9) else -1
        assert i < 0, f"{name}: first failing w = {W[i].tolist()} (case {case[i]}): code {val[i]!r}, closed form {closed[i]!r}"
    # the five affine pieces of the closed form are the five dual vertices
    assert sorted(map(tuple, np.round(fn.vertices, 12).tolist())) == [(2, 5, 3), (3, 3, 5), (3, 4, 4), (5, 1, 5), (5, 2, 4)]
