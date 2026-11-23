"""The policy asserts feasibility and never projects; the r-large-enough checks raise."""
import itertools

import numpy as np
import pytest

from offline import OfflineTables
from policy import GeneralThresholdPolicy, FeasibilityError
from rsn import make_2lln, make_3res_5type
from simulation import gillespie_ergodic
from nets import six_networks


@pytest.mark.parametrize("net", [make_2lln(), make_3res_5type()] + [n for _, n in six_networks()])
def test_allocation_is_table_times_flag_and_feasible(net):
    tab = OfflineTables(net, gamma_c=0.98, gamma_b=0.98)
    pol = GeneralThresholdPolicy(net, c1=0.1, c2=30.0, kappa=0.24, tables=tab)
    r = 100.0
    L, U = pol.thresholds(r)
    rng = np.random.default_rng(1)
    for _ in range(300):
        q = np.floor(rng.uniform(0, 2 * U, net.J))
        b, Z, E = pol.allocation(q, r)
        assert np.array_equal(Z, (q < U - 1e-12).astype(int))
        assert np.array_equal(b, (1.0 - E) * tab.x(Z))          # exactly, no projection
        assert np.all(b >= 0) and np.all(net.K @ b <= net.C + 1e-9)
    assert pol.n_allocations == 300 and pol.n_violations == 0


def test_hysteresis_flag_transitions():
    net = make_2lln()
    pol = GeneralThresholdPolicy(net, c1=1.0, c2=4.0, kappa=0.15, tables=OfflineTables(net))
    r = 100.0
    L, U = pol.thresholds(r)
    q = np.array([U + 1, U + 1, U + 1])
    b, Z, E = pol.allocation(q, r)
    assert np.all(E == 0) and np.all(Z == 0)
    q[0] = np.floor(L) - 1                     # below L: flag on, service stops
    b, Z, E = pol.allocation(q, r)
    assert E[0] == 1 and b[0] == 0 and Z[0] == 1
    q[0] = 0.5 * (L + U)                       # inside the band: flag stays on
    b, Z, E = pol.allocation(q, r)
    assert E[0] == 1 and b[0] == 0
    q[0] = U                                   # at U: flag off, service resumes
    b, Z, E = pol.allocation(q, r)
    assert E[0] == 0 and b[0] > 0 and Z[0] == 0


def test_tampered_table_raises_and_counts():
    net = make_2lln()
    tab = OfflineTables(net, gamma_c=0.98, gamma_b=0.98)
    pol = GeneralThresholdPolicy(net, c1=1.0, c2=4.0, kappa=0.15, tables=tab)
    r = 100.0
    pol.allocation(np.array([50.0, 50.0, 50.0]), r)
    tab.x = lambda z: net.rho + 1.0            # K x > C: must be caught, not projected
    with pytest.raises(FeasibilityError):
        pol.allocation(np.array([50.0, 50.0, 50.0]), r)
    assert pol.n_violations == 1
    tab.x = lambda z: -np.ones(net.J)          # x < 0
    with pytest.raises(FeasibilityError):
        pol.allocation(np.array([50.0, 50.0, 50.0]), r)
    assert pol.n_violations == 2 and pol.n_allocations == 3
    assert not hasattr(pol, "project_capacity")


def test_r_too_small_for_heavy_traffic_raises():
    net = make_2lln()                          # eta = -0.5, rho* = 1: need 0.5/r < 1/4, r > 2
    assert net.r_min_heavy_traffic() == pytest.approx(2.0)
    with pytest.raises(ValueError, match="too small for heavy traffic"):
        net.check_r(2.0)
    with pytest.raises(ValueError):
        net.check_r(1)
    rho_r = net.check_r(2.01)
    assert np.allclose(rho_r, net.rho + net.eta / 2.01)
    pol = GeneralThresholdPolicy(net, c1=1.0, c2=10.0, kappa=0.22, tables=OfflineTables(net))
    with pytest.raises(ValueError):
        pol.allocation(np.zeros(3), 1.0)
    with pytest.raises(ValueError):
        gillespie_ergodic(net, pol, r=1.0, T_diff=0.01, n_snapshots=10,
                          rng=np.random.default_rng(0))


def test_narrow_hysteresis_band_raises_not_warns():
    net = make_2lln()
    pol = GeneralThresholdPolicy(net, c1=1.0, c2=1.5, kappa=0.1, tables=OfflineTables(net))
    # (c2~ - c1~) r^kappa = 0.5 * 10^0.1 = 0.63 <= 1
    with pytest.raises(ValueError, match="hysteresis band"):
        pol.thresholds(10.0)
    with pytest.raises(ValueError):
        gillespie_ergodic(net, pol, r=10.0, T_diff=0.01, n_snapshots=10,
                          rng=np.random.default_rng(0))
    L, U = pol.thresholds(2000.0)              # 0.5 * 2000^0.1 = 1.07 > 1: fine
    assert U - L > 1.0


def test_heterogeneous_beta_r_min():
    net = make_3res_5type(eta=[-0.3, -0.9, -0.5, -0.2, -0.7])
    assert net.r_min_heavy_traffic() == pytest.approx(4 * 0.9 / net.rho_star)
    with pytest.raises(ValueError):
        net.check_r(net.r_min_heavy_traffic() * 0.99)
    net.check_r(net.r_min_heavy_traffic() * 1.01)
