"""20 random admissible networks (the seeds of sim/stress_general.py): every admissibility
condition of the offline pass, eq. (6), K x <= C and x >= rho*/4 for every pattern."""
import itertools

import numpy as np
import pytest

from offline import OfflineTables, kernel_basis
from nets import twenty_random

NETS = twenty_random()


@pytest.mark.parametrize("net", NETS, ids=[f"seed{s}" for s in range(len(NETS))])
def test_network_conditions(net):
    K = net.K
    assert set(np.unique(K)) <= {0.0, 1.0}
    assert np.linalg.matrix_rank(K) == net.I
    for i, j in net.local_types.items():                       # Condition 3
        assert K[i, j] == 1 and K[:, j].sum() == 1
    assert np.all(net.theta < 0)                                 # Condition 2
    assert np.all(net.alphas_r(1e6) > 0) and np.all(net.betas_r(1e6) > 0)
    assert np.allclose(1e6 * (net.rhos_r(1e6) - net.rho), net.eta, rtol=1e-5)
    assert np.allclose(net.Sigma, net.Sigma.T) and np.all(np.linalg.eigvalsh(net.Sigma) > 0)
    assert np.allclose(net.C, K @ net.rho) and net.rho_star > 0
    assert np.allclose(kernel_basis(K).T @ kernel_basis(K), np.eye(net.J - net.I))


@pytest.mark.parametrize("net", NETS, ids=[f"seed{s}" for s in range(len(NETS))])
def test_offline_construction(net):
    tab = OfflineTables(net, gamma_c=0.98, gamma_b=0.98)
    K, rho, rs = net.K, net.rho, net.rho_star
    patterns = list(itertools.product((0, 1), repeat=net.J))
    assert tab.vb_enumerated and set(tab._vb_cache) == set(patterns)
    # cost geometry
    if not tab.trivial:
        assert np.allclose(K @ tab.u, 0, atol=1e-12) and np.isclose(np.linalg.norm(tab.u), 1)
        assert tab.lam < 0 and np.isclose(tab.lam, (net.h * net.beta) @ tab.u)
        assert tab.lam_tilde is not None and tab.lam_tilde < 0
        assert tab.lam_tilde == max(tab.lam_c(z) for z in tab.M_set)
        cap = abs(tab.lam_tilde) / (4 * np.linalg.norm(net.beta) * np.linalg.norm(net.h))
    else:
        assert tab.M_set == [] and tab.lam_tilde is None
        cap = np.inf
    assert tab.b_max == pytest.approx(min(rs / 4, cap), rel=1e-12)          # eq. (6)
    M = set(tab.M_set)
    for z in patterns:
        vc, vb, x = tab.vc(z), tab.vb(z), tab.x(z)
        if z in M:                                                           # v^c requirements
            assert np.allclose(K @ vc, 0, atol=1e-9)                         # in ker K
            assert all(vc[j] >= -1e-12 for j in range(net.J) if z[j])        # >= 0 where z_j = 1
            assert vc @ tab.u > 0                                            # cost reducing
            assert np.all(rho - vc > rs / 2 - 1e-12)                          # size condition
            assert tab.lam_c(z) == pytest.approx(tab.lam * (vc @ tab.u))
        else:
            assert np.all(vc == 0)
        if any(z):                                                           # v^b requirements
            assert np.linalg.norm(vb) <= tab.b_max + 1e-12                   # eq. (6)
            assert np.linalg.norm(vb) == pytest.approx(0.98 * tab.b_max)
            A_z = [i for i in range(net.I) if any(K[i, j] == 1 and not z[j] for j in range(net.J))]
            assert np.allclose((K @ vb)[A_z], 0, atol=1e-9)                  # (K v^b)_i = 0 on A_z
            assert all(vb[j] > 0 for j in range(net.J) if z[j])              # v^b_j > 0 where z_j = 1
        else:
            assert np.all(vb == 0)
        assert np.array_equal(x, rho - (vc if z in M else 0) - vb)
        assert np.all(K @ x <= net.C + 1e-8)                                 # capacity
        assert np.all(x >= rs / 4 - 1e-8)                                    # x >= rho*/4
    # down-set property of M
    for z in M:
        for j in range(net.J):
            if z[j]:
                assert z[:j] + (0,) + z[j + 1:] in M
    assert tab.n_lp_c <= 2 ** net.J


def test_b_max_range_matches_oct_29_observation():
    """b_max ranges from 0.0003 to 0.09 on these networks."""
    bm = [OfflineTables(n, gamma_c=0.98, gamma_b=0.98).b_max for n in NETS]
    assert 2e-4 < min(bm) < 5e-4 and 0.05 < max(bm) < 0.1
