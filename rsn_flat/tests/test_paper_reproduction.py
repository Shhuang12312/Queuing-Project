"""Paper reproduction: Example 2.2 (the 2LLN toy) and Example 2.1 (cost-trivial case),
plus the bit-for-bit reproduction of one entry of convergence_results.json."""
import json
import os

import numpy as np
import pytest

from rsn import make_2lln
from offline import OfflineTables
from simulation import hhat_batch, dual_vertices
from je import run_JE

FLAT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_example_2_2_network_objects():
    net = make_2lln()                                  # alpha = beta = h = 1, eta = -1/2
    assert net.K.tolist() == [[1, 0, 1], [0, 1, 1]]
    assert np.array_equal(net.rho, np.ones(3)) and np.array_equal(net.C, [2.0, 2.0])
    assert net.rho_star == 1.0
    assert np.array_equal(net.theta, [-1.0, -1.0])     # theta = K eta
    assert np.array_equal(net.Sigma, [[4.0, 2.0], [2.0, 4.0]])   # 2 K K^T
    assert net.local_types == {0: 0, 1: 1}
    assert net.r_min_heavy_traffic() == pytest.approx(2.0)


def test_example_2_2_offline_objects():
    tab = OfflineTables(make_2lln(), gamma_c=0.98, gamma_b=0.98)
    assert not tab.trivial
    assert np.allclose(tab.u, np.array([-1, -1, 1]) / np.sqrt(3))            # u_1
    assert tab.lam == pytest.approx(-1 / np.sqrt(3))                           # lambda = h beta . u_1
    assert tab.M_set == [(0, 0, 0), (0, 0, 1)]                                 # M
    for z in tab.M_set:                                                        # v^c = 0.49 (-1,-1,1)
        assert np.allclose(tab.vc(z), 0.49 * np.array([-1, -1, 1]))
        assert tab.lam_c(z) == pytest.approx(-0.49)
    assert tab.lam_tilde == pytest.approx(-0.49)                               # eq. (5)
    # eq. (6), Euclidean reading: b_max = min(1/4, 0.49 / (4 sqrt3 sqrt3)) = 0.49/12
    assert tab.b_max == pytest.approx(0.49 / 12) == pytest.approx(0.040833, abs=1e-6)
    for z in tab._vb_cache:
        vb = tab.vb(z)
        if any(z):
            assert np.linalg.norm(vb) == pytest.approx(0.98 * tab.b_max)
            A_z = [i for i in range(2) if any(tab.rsn.K[i, j] == 1 and not z[j] for j in range(3))]
            assert np.allclose((tab.rsn.K @ vb)[A_z], 0.0, atol=1e-12)            # (K v^b)_i = 0, i in A_z
            assert all(vb[j] > 0 for j in range(3) if z[j])                        # v^b_j > 0 where z_j = 1
        else:
            assert np.all(vb == 0)
    # the paper's own +-1/36 example satisfies (6) in sup norm (1/36 = 0.0278 <= 1/36) but its
    # Euclidean norm sqrt(3)/36 = 0.048 exceeds our stricter b_max = 0.0408
    assert np.sqrt(3) / 36 > tab.b_max > 1 / 36


def test_example_2_2_hhat_is_max():
    net = make_2lln()
    assert sorted(map(tuple, dual_vertices(net).tolist())) == [(0.0, 1.0), (1.0, 0.0)]
    W = np.random.default_rng(0).uniform(0, 5, (200, 2))
    assert np.allclose(hhat_batch(W, net), W.max(axis=1), atol=1e-12)
    # Prop 2.1 spot check: h.q - h_hat(K M q) = |lambda| d~(q), d~(q) = sqrt(3) min(q1, q2) here
    q = np.array([3.0, 5.0, 2.0])
    assert net.h @ q - hhat_batch(net.workload(q)[None, :], net)[0] == pytest.approx(min(q[0], q[1]))


def test_example_2_1_trivial_case():
    """h = (1,1,2): h beta is orthogonal to ker K, so M is empty and only v^b acts."""
    net = make_2lln(h=(1, 1, 2))
    tab = OfflineTables(net, gamma_c=0.98, gamma_b=0.98)
    assert tab.trivial and tab.u is None and tab.lam == 0.0
    assert tab.M_set == [] and tab.lam_tilde is None
    assert tab.b_max == pytest.approx(net.rho_star / 4)                       # = 1/4
    for z in tab._vb_cache:
        assert np.all(tab.vc(z) == 0) and not tab.in_M(z)
        x = tab.x(z)
        assert np.all(net.K @ x <= net.C + 1e-9) and np.all(x >= net.rho_star / 4 - 1e-9)
    W = np.random.default_rng(0).uniform(0, 5, (200, 2))
    assert np.allclose(hhat_batch(W, net), W.sum(axis=1), atol=1e-12)         # h_hat = w1 + w2


def test_reproduces_colab_sweep_entry_bit_for_bit():
    """faithful|r=50|seed=1 of convergence_results.json: ~6 s."""
    store = json.load(open(os.path.join(FLAT, "convergence_results.json")))
    ref = store["faithful|r=50|seed=1"]
    net = make_2lln()
    out = run_JE(net, r=50.0, T=20.0, c1=0.5, c2=10.0, kappa=0.22, seed=1, n_snapshots=1200,
                 hhat_fn=lambda W: np.maximum(W[:, 0], W[:, 1]))
    level = float(np.maximum.reduce(((out["snapshots"] @ (net.K @ net.M).T) / 50.0).T).mean())
    assert out["JE"] == ref["JE"]
    assert out["E_frac"] == ref["E_frac"]
    assert out["ssc_gap"] == ref["gap"]
    assert level == ref["level"]
