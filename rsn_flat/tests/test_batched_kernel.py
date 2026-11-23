"""The one-pass kernel: per-batch integrals, M^r in W, pathwise assertions.

Skips when sim/genkern.so is not built (cd sim && ./build.sh)."""
import os

import numpy as np
import pytest

SIM = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "sim")
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(SIM, "genkern.so")),
                                reason="sim/genkern.so not built (cd sim && ./build.sh)")
K2 = [[1, 0, 1], [0, 1, 1]]


def test_batches_reconstruct_the_window_averages():
    import networks as N
    nw = N.build("A toy (Ex. 2.2)")
    o = nw.run(200, 20.0, 0.1, 60.0, 0.24, 1, burn=0.3, batch=2.0)        # window [6, 20] = 7 batches of 2
    b = o["batches"]
    assert b["n"] == 7 and b["len"] == 2.0
    assert np.mean(b["hq"]) == pytest.approx(o["JE"], abs=1e-12)
    assert np.mean(b["hhat"]) == pytest.approx(o["level"], abs=1e-12)
    assert np.mean(b["hg"]) == pytest.approx(o["hgi_path"], abs=1e-12)
    # per batch: h.q >= h^(W) >= h^(G) on the toy (h^ monotone)
    assert np.all(np.array(b["hq"]) >= np.array(b["hhat"]) - 1e-12)
    assert np.all(np.array(b["hhat"]) >= np.array(b["hg"]) - 1e-12)
    assert o["checks"] == "h.q>=h^(W); G<=W; h^(G)<=h^(W)"
    # the totals are unchanged by batching (same pass, same arithmetic)
    o2 = nw.run(200, 20.0, 0.1, 60.0, 0.24, 1, burn=0.3)
    assert o2["JE"] == o["JE"] and o2["hgi_path"] == o["hgi_path"] and "batches" not in o2
    with pytest.raises(ValueError):
        nw.run(200, 20.0, 0.1, 60.0, 0.24, 1, burn=0.3, batch=50.0)


def test_monotone_flag_and_checks_per_network():
    import networks as N
    mono = {name: N.build(name).hhat_monotone for name in N.SPECS}
    assert mono["A toy (Ex. 2.2)"] and mono["B cost-trivial (Ex. 2.1)"] and mono["C shared type costly"]
    assert not mono["D unequal rates"] and not mono["E unequal loads, costs"]
    o = N.build("D unequal rates").run(100, 10.0, 0.1, 41.0, 0.24, 2)      # G <= W and h.q >= h^(W) still asserted
    assert o["checks"] == "h.q>=h^(W); G<=W"


def test_assertion_aborts_with_diagnostics():
    import networks as N
    nw = N.build("A toy (Ex. 2.2)")
    nw.net.h = nw.net.h * 0.5                       # kernel gets h/2 while V is built from h: h.q < h^(W)
    with pytest.raises(AssertionError, match="h.q >= h"):
        nw.run(200, 5.0, 0.1, 60.0, 0.24, 1)


def test_beta_split_uses_M_r():
    import gensim
    a = gensim.Net("beta-split", K2, [1, 1, 1], [1, 1, 1], [1, 1, 1], alpha_bar=[0, 0, 0], beta_bar=[0.5] * 3)
    c = gensim.Net("canonical", K2, [1, 1, 1], [1, 1, 1], [1, 1, 1], eta=[-0.5] * 3)
    oa, oc = a.run(100, 20.0, 0.1, 60.0, 0.24, 3), c.run(100, 20.0, 0.1, 60.0, 0.24, 3)
    assert np.allclose(a.net.theta, c.net.theta) and oa["checks"] == oc["checks"]
    assert 0.5 < oa["hgi_path"] / oc["hgi_path"] < 2.0   # same limit; M^r = diag(1/(1 + 0.5/r)) differs from M by 0.5%


def test_ygrid_samples_the_free_process():
    import networks as N
    nw = N.build("A toy (Ex. 2.2)")
    o = nw.run(200, 30.0, 0.1, 60.0, 0.24, 4, burn=0.3, ygrid=1.0)
    Y = o["ygrid"]["Y"]
    assert Y.shape == (22, 2) and o["ygrid"]["t0"] == 9.0            # 21 unit steps over [9, 30]
    dX = np.diff(Y, axis=0)
    # drift of the free process is theta = (-1, -1): 21 increments of variance ~4 give SE ~0.44 per coordinate
    assert np.all(np.abs(dX.mean(axis=0) - nw.net.theta) < 4 * np.sqrt(4.0 / 21))
    # the grid does not change the window totals
    o2 = nw.run(200, 30.0, 0.1, 60.0, 0.24, 4, burn=0.3)
    assert o2["JE"] == o["JE"] and o2["hgi_path"] == o["hgi_path"]
