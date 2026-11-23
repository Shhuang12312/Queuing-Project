"""The pattern-dependent cap vb_cap="per_pattern" (LABELED EXPERIMENT, outside eq. (6)).

Checks (all seven networks of sim/networks.py, pure Python):
  * default tables are unchanged (vb_cap="uniform" is eq. (6), identical to the construction without the option);
  * per-pattern tables: b_cap(z) = min(rho*/4, |lambda_c(z)|/(4|beta||h|)) on M, rho*/4 off M; ||v^b(z)|| = gamma_b b_cap(z);
  * the inequality step (47) of the proof of Prop 3.5 needs, h beta.(v^c + v^b) <= -3/4|lambda_c(z)| <= -3/4|lambda~|,
    holds on every z in M under both caps (check_step47), and every x(z) stays feasible (Kx <= C, x >= rho*/4);
  * the per-pattern push is never smaller than eq. (6)'s, equals it at the patterns attaining lambda~, and is
    rho*/4 (times gamma_b) off M;
  * every output is labeled "outside eq. (6)"; b_max still reports the eq. (6) value.
"""
import itertools

import numpy as np
import pytest

from offline import OfflineTables
from nets import seven_networks

NETS = seven_networks()
IDS = [n for n, _ in NETS]


@pytest.fixture(scope="module")
def tables():
    return {name: (OfflineTables(net, gamma_c=0.98, gamma_b=0.98),
                   OfflineTables(net, gamma_c=0.98, gamma_b=0.98, vb_cap="per_pattern"))
            for name, net in NETS}


def test_bad_option():
    with pytest.raises(ValueError):
        OfflineTables(NETS[0][1], vb_cap="something")


@pytest.mark.parametrize("name", IDS)
def test_default_is_eq6_and_unchanged(name, tables):
    tu, tp = tables[name]
    ref = OfflineTables(tu.rsn, gamma_c=0.98, gamma_b=0.98)
    assert tu.vb_cap == "uniform" and tu.label == "Definition 2.3, eq. (6)"
    for z in itertools.product((0, 1), repeat=tu.rsn.J):
        assert np.array_equal(tu.vb(z), ref.vb(z)) and np.array_equal(tu.x(z), ref.x(z))
        assert tu.b_cap(z) == tu.b_max
    # M, lambda~, b_max (eq. (6)) are identical under both options
    assert tp.M_set == tu.M_set and tp.lam_tilde == tu.lam_tilde and tp.b_max == tu.b_max


@pytest.mark.parametrize("name", IDS)
def test_per_pattern_cap_values_and_push(name, tables):
    tu, tp = tables[name]
    n = tp.rsn
    denom = 4 * np.linalg.norm(n.beta) * np.linalg.norm(n.h)
    for z in itertools.product((0, 1), repeat=n.J):
        cap = n.rho_star / 4
        if tp.in_M(z):
            cap = min(cap, abs(tp.lam_c(z)) / denom)
        assert tp.b_cap(z) == pytest.approx(cap, rel=1e-12)
        assert tp.b_cap(z) >= tu.b_cap(z) - 1e-15                      # never less push than eq. (6)
        if any(z):
            assert np.linalg.norm(tp.vb(z)) == pytest.approx(0.98 * cap, rel=1e-9)
            assert np.linalg.norm(tp.vb(z)) >= np.linalg.norm(tu.vb(z)) - 1e-12
            # direction unchanged: same LP-b optimizer, only the scale differs
            assert np.allclose(tp.vb(z) / np.linalg.norm(tp.vb(z)), tu.vb(z) / np.linalg.norm(tu.vb(z)))
        else:
            assert not tp.vb(z).any()
        if not tp.in_M(z):
            assert tp.b_cap(z) == n.rho_star / 4
    if tp.lam_tilde is not None:                                        # Case B: agree where lambda~ is attained
        zstar = [z for z in tp.M_set if tp.lam_c(z) == pytest.approx(tp.lam_tilde, rel=1e-12)]
        assert zstar and all(tp.b_cap(z) == pytest.approx(tu.b_max, rel=1e-12) for z in zstar)
    tp.assert_feasibility()                                             # K x <= C, x >= rho*/4 for every z


@pytest.mark.parametrize("name", IDS)
def test_step47_inequality_both_caps(name, tables):
    tu, tp = tables[name]
    n = tp.rsn
    g = n.h * n.beta
    for t in (tu, tp):
        out = t.check_step47()
        assert out["ok"]
        if t.lam_tilde is None:
            assert out["n"] == 0
            continue
        assert out["n"] == len(t.M_set) and out["min_margin"] >= 0 and out["min_margin_tilde"] >= 0
        for z in t.M_set:
            lhs = g @ (t.vc(z) + t.vb(z))
            assert lhs <= -0.75 * abs(t.lam_c(z)) + 1e-12 <= -0.75 * abs(t.lam_tilde) + 1e-12
            # and the mechanism: h beta . v^b(z) <= |h||beta| ||v^b(z)|| <= gamma_b |lambda_c(z)|/4
            assert g @ t.vb(z) <= 0.98 * abs(t.lam_c(z)) / 4 + 1e-12
            assert g @ t.vc(z) == pytest.approx(t.lam_c(z), rel=1e-9)


@pytest.mark.parametrize("name", IDS)
def test_labels(name, tables):
    tu, tp = tables[name]
    assert "outside eq. (6)" in tp.label and "outside eq. (6)" in tp.push_stats()["label"]
    assert "OUTSIDE eq. (6)" in tp.summary() and "OUTSIDE" not in tu.summary()
    assert tp.b_max == tu.b_max                                         # b_max is always the eq. (6) value
    ps = tp.push_stats()
    if tp.lam_tilde is not None:
        assert ps["in_M"]["min"] == pytest.approx(0.98 * tu.b_max, rel=1e-9)
    if ps["off_M"]["n"]:
        assert ps["off_M"]["min"] == pytest.approx(0.98 * tp.rsn.rho_star / 4, rel=1e-9)
