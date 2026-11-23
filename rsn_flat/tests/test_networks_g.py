"""Network G (Example 3 of Budhiraja and Johnson (2020), Math. Oper. Res. 45(3), p. 812;
Example 5.6 of arXiv:1710.09042) and the occupancy-aware flag rule in sim/networks.py."""
import os

import numpy as np
import pytest

SIM = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "sim")
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(SIM, "genkern.so")),
                                reason="sim/genkern.so not built (cd sim && ./build.sh)")
G = "G ranking-map fails (BJ20 Ex. 6.6)"


def test_network_G_offline_objects():
    import networks as N
    K, alpha, beta, h, eta = N.SPECS[G]
    assert K == [[1, 0, 0, 1, 0, 1], [0, 1, 0, 1, 1, 1], [0, 0, 1, 0, 1, 1]]        # chi_1..chi_3, chi_12, chi_23, chi_123
    assert h == [5, 5, 5, 7, 8, 11] and beta == [1] * 6                            # as in the 2020 paper
    nw = N.build(G)
    n, t = nw.net, nw.tab
    assert np.array_equal(n.C, [3.0, 4.0, 3.0]) and np.array_equal(n.theta, [-1.5, -2.0, -1.5])
    assert not t.trivial and len(t.M_set) == 42 and t.n_lp_c == 47
    assert t.lam == pytest.approx(-2.121320, abs=1e-6) and t.lam_tilde == pytest.approx(-0.49, abs=1e-9)
    assert t.b_max == pytest.approx(0.49 / (4 * np.sqrt(6) * np.linalg.norm(h)), rel=1e-9)   # 0.002845
    assert t.b_max == pytest.approx(0.002845, abs=1e-6)
    assert len(nw.V) == 5 and nw.vertex_err < 1e-10
    assert np.isnan(N.rule_c2(nw))                      # the 5% rule has no solution: push too small
    U = 660 * r0k(1000.0); d = N.floor_drift(nw, 1000.0, U, N.F_OCC)
    assert 0.5 < d < 0.6                                 # the model-chosen band still carries 56% flag drift


def r0k(r):
    import networks as N
    return r ** N.KAPPA


def test_ssc_occupancy_toy_is_one_half():
    import networks as N
    nw = N.build("A toy (Ex. 2.2)")
    f = N.ssc_occupancy(nw, n_samples=1500, T=8000.0)
    assert abs(f[0] - 0.5) < 0.06 and abs(f[1] - 0.5) < 0.06       # q*_j = 0 iff w_j <= w_other
    assert f[2] < 0.25                                             # shared type rarely at zero


def test_rule_c2_with_per_type_occupancy():
    import networks as N
    nw = N.build("D unequal rates")
    c_half = N.rule_c2(nw)
    c_meas = N.rule_c2(nw, f=np.array([0.28, 0.68, 0.014]))          # measured below-U occupancies
    assert 40 < c_half < 42 and 27 < c_meas < 29 and c_meas < c_half
    assert N.floor_drift(nw, 1000.0, 100.0, 0.5) > N.floor_drift(nw, 1000.0, 100.0, [0.5, 0.5, 0.01])
