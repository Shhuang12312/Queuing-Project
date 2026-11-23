"""Condition 2 in the paper's (alpha_bar, beta_bar) form.

eta = beta^-2 (alpha_bar beta - alpha beta_bar); alpha^r = alpha + alpha_bar/r,
beta^r = beta + beta_bar/r. Three splits with the same eta must give the same theta and
Sigma, and the same J_E within noise at r = 100 (the simulators use beta^r).
"""
import os

import numpy as np
import pytest

from rsn import RSN, make_2lln
from offline import OfflineTables
from policy import GeneralThresholdPolicy
from simulation import gillespie_ergodic

K2 = [[1, 0, 1], [0, 1, 1]]
ETA = np.array([-0.5, -0.5, -0.5])
ALPHA, BETA = np.ones(3), np.ones(3)


def three_splits():
    """Same eta, three ways: all in alpha (canonical), all in beta, half and half."""
    a = RSN(K2, ALPHA, BETA, [1, 1, 1], eta=ETA, name="split-alpha")
    b = RSN(K2, ALPHA, BETA, [1, 1, 1], alpha_bar=np.zeros(3), beta_bar=-ETA * BETA ** 2 / ALPHA,
            name="split-beta")
    c = RSN(K2, ALPHA, BETA, [1, 1, 1], alpha_bar=0.5 * BETA * ETA, beta_bar=-0.5 * ETA * BETA ** 2 / ALPHA,
            name="split-mixed")
    return a, b, c


def test_canonical_split_reproduces_old_rates_exactly():
    net = make_2lln()
    assert np.array_equal(net.alpha_bar, net.beta * net.eta) and np.all(net.beta_bar == 0)
    for r in (50.0, 1000.0):
        assert np.array_equal(net.alphas_r(r), net.alpha + net.beta * net.eta / r)   # the canonical split
        assert np.array_equal(net.betas_r(r), net.beta)
        assert np.allclose(net.rhos_r(r), net.rho + net.eta / r)


def test_splits_share_eta_theta_sigma():
    a, b, c = three_splits()
    for n in (b, c):
        assert np.allclose(n.eta, a.eta, rtol=1e-12, atol=0)
        assert np.array_equal(n.theta, a.theta) or np.allclose(n.theta, a.theta, rtol=1e-12)
        assert np.array_equal(n.Sigma, a.Sigma)
        assert np.allclose(n.rhos_r(100.0) - n.rho, a.eta / 100.0, rtol=1e-2)    # O(1/r^2) apart
        assert 100.0 * np.max(np.abs(n.rhos_r(100.0) - a.rhos_r(100.0))) < 1e-2
    assert np.all(b.betas_r(100.0) > b.beta)        # service really is faster in split b
    assert np.array_equal(b.alphas_r(100.0), b.alpha)


def test_inconsistent_or_unstable_inputs_raise():
    with pytest.raises(ValueError, match="inconsistent"):
        RSN(K2, ALPHA, BETA, [1, 1, 1], eta=ETA, alpha_bar=np.zeros(3), beta_bar=np.zeros(3))
    with pytest.raises(ValueError, match="Condition 2"):
        RSN(K2, ALPHA, BETA, [1, 1, 1], eta=[0.5, -0.5, -0.5])         # theta_1 = 0
    with pytest.raises(ValueError, match="Condition 2"):
        RSN(K2, ALPHA, BETA, [1, 1, 1], alpha_bar=[1, 0, 0])             # eta_1 > 0
    net = RSN(K2, ALPHA, BETA, [1, 1, 1], eta=ETA, alpha_bar=BETA * ETA, beta_bar=np.zeros(3))
    assert np.array_equal(net.eta, ETA)
    with pytest.raises(ValueError):                                       # beta^r <= 0 at tiny r
        RSN(K2, ALPHA, BETA, [1, 1, 1], alpha_bar=np.zeros(3), beta_bar=[-2, -2, -2]).check_r(0.5)


def test_offline_tables_do_not_depend_on_the_split():
    a, b, c = three_splits()
    ta, tb, tc = (OfflineTables(n, gamma_c=0.98, gamma_b=0.98) for n in (a, b, c))
    assert ta.b_max == tb.b_max == tc.b_max
    for z in ta._vb_cache:
        assert np.array_equal(ta.x(z), tb.x(z)) and np.array_equal(ta.x(z), tc.x(z))


def test_pure_python_simulator_uses_beta_r():
    """With all of eta in beta_bar, alpha^r = alpha: ignoring beta^r would give theta = 0."""
    a, b, _ = three_splits()
    res = {}
    for n in (a, b):
        tab = OfflineTables(n, gamma_c=0.98, gamma_b=0.98)
        pol = GeneralThresholdPolicy(n, c1=0.1, c2=60.0, kappa=0.24, tables=tab)
        out = gillespie_ergodic(n, pol, r=100.0, T_diff=1.0, burn_frac=0.3, n_snapshots=50,
                                rng=np.random.default_rng(3))
        res[n.name] = out["JE"]
    # same seed, rates differ at O(1/r): the paths must differ (beta^r is in use) but stay comparable
    assert res["split-alpha"] != res["split-beta"]
    assert 0.3 < res["split-beta"] / res["split-alpha"] < 3.0


_SKIP_SO = pytest.mark.skipif(not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "..", "sim", "genkern.so")),
                              reason="sim/genkern.so not built (cd sim && ./build.sh)")


def _compiled_nets():
    from gensim import Net
    a, b, c = three_splits()
    return [Net(n.name, K2, ALPHA, BETA, [1, 1, 1], eta=None if np.any(n.beta_bar) else ETA,
                alpha_bar=None if not np.any(n.beta_bar) else n.alpha_bar,
                beta_bar=None if not np.any(n.beta_bar) else n.beta_bar) for n in (a, b, c)]


@_SKIP_SO
def test_JE_within_noise_across_splits_independent_seeds():
    """The statistical test: compiled engine, r = 100, (0.1, 60, 0.24), T = 150, 4 seeds per split with DIFFERENT
    seeds across splits (1-4, 11-14, 21-24), so the three samples are independent; |z| < 3 pairwise on J_E and on
    the same-path gap J_E - h^(G). (With shared seeds the one-stream kernel gives nearly identical paths and the
    comparison is a code-consistency check, not a test of invariance in distribution: see the next test.)"""
    nets = _compiled_nets()
    seeds = {"split-alpha": (1, 2, 3, 4), "split-beta": (11, 12, 13, 14), "split-mixed": (21, 22, 23, 24)}
    stats = {}
    for nw in nets:
        runs = [nw.run(100, 150.0, 0.1, 60.0, 0.24, s) for s in seeds[nw.name]]
        je = np.array([o["JE"] for o in runs]); gap = np.array([o["JE"] - o["hgi_path"] for o in runs])
        assert np.all(np.isfinite(je)) and np.all(je > 0)
        stats[nw.name] = {k: (v.mean(), v.std(ddof=1) / np.sqrt(len(v))) for k, v in (("JE", je), ("gap", gap))}
    names = list(stats)
    for i in range(3):
        for j in range(i + 1, 3):
            for k in ("JE", "gap"):
                (m1, s1), (m2, s2) = stats[names[i]][k], stats[names[j]][k]
                z = (m1 - m2) / np.hypot(s1, s2)
                assert abs(z) < 3.0, f"{k}: {names[i]} vs {names[j]}: {m1:.3f}±{s1:.3f} vs {m2:.3f}±{s2:.3f}, z={z:.2f}"


@_SKIP_SO
def test_same_seed_splits_are_consistent_code_check():
    """Code-consistency check, NOT a statistical test: the SAME seeds (1, 2) for all three splits. The kernel draws
    competing clocks from one stream and the splits differ in the rates only at O(1/r), so the event sequences
    are nearly identical and J_E must agree to well within 1% at r = 100, T = 40 — this checks that beta^r and
    alpha^r are applied consistently, nothing about invariance in distribution."""
    nets = _compiled_nets()
    je = {nw.name: np.array([nw.run(100, 40.0, 0.1, 60.0, 0.24, s)["JE"] for s in (1, 2)]) for nw in nets}
    ref = je["split-alpha"]
    for name in ("split-beta", "split-mixed"):
        assert np.all(je[name] != ref)                                   # beta^r is really in use
        assert np.all(np.abs(je[name] - ref) / ref < 0.01), (name, je[name], ref)
