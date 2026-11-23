"""Tests for three items of docs/defects.md that the toy tests cannot see.

* defect 7 (covariance from squared SDs): on the toy alpha = beta = 1 makes the two conventions coincide; this
  asserts Sigma on network D (alpha, beta != 1) against the renewal-variance-rate formula and against the wrong one;
* defect 4 (fixed internal seed): different seeds must give different paths, the same seed the same path, in
  both engines;
* the discounted benchmark HGI_D: toy value 1.33 from an empty start."""
import os

import numpy as np
import pytest

from rsn import RSN, make_2lln
from offline import OfflineTables
from je import run_JE
from simulation import benchmark_hgi_d
from nets import SEVEN

SIM = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "sim")


def test_sigma_uses_renewal_variance_rates_on_heterogeneous_network():
    K, alpha, beta, h, eta = SEVEN["D unequal rates"]
    net = RSN(K, alpha, beta, h, eta=eta)
    K, a, b = np.asarray(K, float), np.asarray(alpha, float), np.asarray(beta, float)
    M, R = np.diag(1 / b), np.diag(a / b)
    right = K @ M @ (np.diag(a) + np.diag(b) @ R) @ M.T @ K.T              # renewal variance rates alpha_j, beta_j
    wrong = K @ M @ (np.diag(1 / a ** 2) + np.diag(1 / b ** 2) @ R) @ M.T @ K.T   # squared SDs of the primitives
    assert np.allclose(net.Sigma, right, rtol=0, atol=1e-12)
    assert np.allclose(net.Sigma, [[3.8, 2.4], [2.4, 2 / 1.69 + 2.4]], atol=1e-12)
    assert not np.allclose(net.Sigma, wrong, atol=1e-3)                    # the defect would be visible here
    toy = make_2lln()                                                       # ... and invisible on the toy
    Kt, Mt = toy.K, toy.M
    assert np.allclose(Kt @ Mt @ (np.diag(1 / toy.alpha ** 2) + np.diag(1 / toy.beta ** 2) @ toy.R) @ Mt.T @ Kt.T, toy.Sigma)
    assert np.array_equal(net.theta, K @ np.asarray(eta, float))


def test_seed_sensitivity_pure_python():
    net = make_2lln()
    tab = OfflineTables(net, gamma_c=0.98, gamma_b=0.98)
    kw = dict(r=50.0, T=4.0, c1=0.1, c2=60.0, kappa=0.24, n_snapshots=100, tables=tab)
    a1, a2, b = run_JE(net, seed=1, **kw), run_JE(net, seed=1, **kw), run_JE(net, seed=2, **kw)
    assert a1["JE"] == a2["JE"] and a1["E_frac"] == a2["E_frac"]           # deterministic given the seed
    assert a1["JE"] != b["JE"]                                             # different seed, different path


@pytest.mark.skipif(not os.path.exists(os.path.join(SIM, "genkern.so")), reason="sim/genkern.so not built")
def test_seed_sensitivity_compiled():
    import networks as N
    nw = N.build("A toy (Ex. 2.2)")
    a1, a2, b = (nw.run(100, 10.0, 0.1, 60.0, 0.24, s) for s in (1, 1, 2))
    assert a1["JE"] == a2["JE"] and a1["hgi_path"] == a2["hgi_path"]
    assert a1["JE"] != b["JE"] and a1["hgi_path"] != b["hgi_path"]


def test_hgi_d_toy_from_empty():
    """Reduced size (~10 s): one leg, dt = 1e-3, 2000 paths. The fixed-dt value is 1.300 +- 0.006 with 10000 paths
    (biased low by ~3%); the sqrt(dt)-extrapolated benchmark is 1.34 +- 0.02 (sim/results/hgi_d_toy.json) and the
    recorded value 1.33. Extrapolating from 2000-path legs amplifies noise 2.4x, so the test checks one leg only."""
    m, se = benchmark_hgi_d(make_2lln(), dt=1e-3, n_paths=2000)
    assert abs(m - 1.300) < 0.05 and 0 < se < 0.03
    # unnormalized convention (paper eq. (11)): a larger varsigma gives a smaller value, roughly like 1/varsigma
    m2, _ = benchmark_hgi_d(make_2lln(), varsigma=2.0, dt=2e-3, n_paths=1000, T=10.0)
    assert 0.3 < m2 < 0.7
    with pytest.raises(ValueError):
        benchmark_hgi_d(make_2lln(), varsigma=0.0)
    with pytest.raises(ValueError):
        benchmark_hgi_d(make_2lln(), dt=(1e-3, 2e-3))
