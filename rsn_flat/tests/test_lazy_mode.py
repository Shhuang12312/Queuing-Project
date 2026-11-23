"""No lazy approximation of b_max for J > max_enumerate.

M and lambda~ are found by enumeration with monotone pruning (exact for every J), b_max comes from the true
lambda~, and every v^b entry is asserted on creation, also when the table is filled on demand."""
import itertools

import numpy as np
import pytest

from offline import OfflineTables
from rsn import make_2lln, make_3res_5type
from nets import make_J18, random_admissible, six_networks


def _cap(net, tab):
    return abs(tab.lam_tilde) / (4 * np.linalg.norm(net.beta) * np.linalg.norm(net.h))


def test_toy_forced_lazy_matches_full_enumeration():
    net = make_2lln()
    full = OfflineTables(net, gamma_c=0.98, gamma_b=0.98)              # J=3 <= 16
    lazy = OfflineTables(net, gamma_c=0.98, gamma_b=0.98, max_enumerate=2)
    assert full.vb_enumerated and not lazy.vb_enumerated
    assert lazy.M_set == full.M_set == [(0, 0, 0), (0, 0, 1)]
    assert lazy.lam_tilde == full.lam_tilde
    assert lazy.b_max == full.b_max == pytest.approx(0.040833, abs=1e-6)
    for z in itertools.product((0, 1), repeat=3):
        assert np.array_equal(lazy.vb(z), full.vb(z))       # lazily created, asserted
        assert np.array_equal(lazy.x(z), full.x(z))
        assert np.linalg.norm(lazy.vb(z)) <= lazy.b_max + 1e-12


def test_J18_b_max_from_true_lambda_tilde():
    """b_max must come from lambda~: with b_max == rho*/4 the vector v^b would violate eq. (6)."""
    net = make_J18()
    assert net.J == 18
    tab = OfflineTables(net, gamma_c=0.98, gamma_b=0.98)             # J=18 > 16
    assert not tab.trivial and tab.lam_tilde is not None and tab.lam_tilde < 0
    assert len(tab.M_set) == tab.n_lp_c - sum(1 for v in tab._inM_cache.values() if not v)
    cap = _cap(net, tab)
    assert cap < net.rho_star / 4                                   # the lambda~ cap binds
    assert tab.b_max == pytest.approx(min(net.rho_star / 4, cap), rel=1e-12)
    assert tab.b_max < 0.5 * net.rho_star / 4
    rng = np.random.default_rng(0)
    patterns = [(1,) * 18, (0,) * 18] + [tuple(int(b) for b in rng.integers(0, 2, 18)) for _ in range(30)]
    for z in patterns:
        vb = tab.vb(z)                                              # asserted on creation
        x = tab.x(z)
        assert np.linalg.norm(vb) <= tab.b_max + 1e-12
        assert np.all(net.K @ x <= net.C + 1e-8) and np.all(x >= net.rho_star / 4 - 1e-8)
    assert tab.lam_tilde == max(tab.lam_c(z) for z in tab.M_set)


@pytest.mark.parametrize("net", [make_2lln(), make_3res_5type(), make_2lln(h=(1, 1, 5))]
                         + [n for _, n in six_networks()]
                         + [random_admissible(np.random.default_rng(s), 3, 3) for s in range(4)])
def test_pruned_enumeration_equals_bruteforce(net):
    tab = OfflineTables(net, gamma_c=0.98, gamma_b=0.98)
    lam_c = []
    for z in itertools.product((0, 1), repeat=net.J):
        in_M, vc = tab._make_vc(z)                 # brute force: one LP-c per pattern
        assert in_M == tab.in_M(z)
        assert np.array_equal(vc, tab.vc(z))
        if in_M:
            lam_c.append(tab.lam * float(vc @ tab.u))
    assert (max(lam_c) if lam_c else None) == tab.lam_tilde
    # monotonicity (down-set) holds on the computed set
    M = set(tab.M_set)
    for z in M:
        for j in range(net.J):
            if z[j]:
                assert z[:j] + (0,) + z[j + 1:] in M
