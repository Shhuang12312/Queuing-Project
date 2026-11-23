"""The GI (renewal-arrival, general-service) engine sim/genkern_gi.c.

With exponential clocks it is the same CTMC as genkern (equal in law): |z| < 3 at r = 50.
Sigma uses the renewal variance rates alpha scv_u, beta scv_v."""
import os

import numpy as np
import pytest

SIM = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "sim")
pytestmark = pytest.mark.skipif(not (os.path.exists(os.path.join(SIM, "genkern_gi.so"))
                                     and os.path.exists(os.path.join(SIM, "genkern.so"))),
                                reason="sim/genkern_gi.so not built (cd sim && ./build.sh)")
K2 = [[1, 0, 1], [0, 1, 1]]
TOY = (K2, [1, 1, 1], [1, 1, 1], [1, 1, 1], [-0.5] * 3)


def test_distribution_coding_and_sigma():
    from gensim_gi import GINet, h2_balanced, dist_code
    p, mu1, mu2 = h2_balanced(1.0, 2.0)
    mean = p / mu1 + (1 - p) / mu2
    var = 2 * (p / mu1 ** 2 + (1 - p) / mu2 ** 2) - mean ** 2
    assert mean == pytest.approx(1.0) and var / mean ** 2 == pytest.approx(2.0)      # SCV 2
    assert dist_code(("erlang", 2), 3.0) == ([1.0, 2.0, 6.0, 0.0], 0.5)
    assert dist_code("exp", 3.0) == ([0.0, 3.0, 0.0, 0.0], 1.0)
    g = GINet("toy-gi", *TOY, arrivals=("h2", 2.0), services=("erlang", 2))
    assert np.array_equal(g.net.scv_u, [2, 2, 2]) and np.array_equal(g.net.scv_v, [.5, .5, .5])
    assert np.array_equal(g.net.Sigma, 2.5 * np.array([[2, 1], [1, 2]]))                 # 2.5 K K^T
    e = GINet("toy-gi-exp", *TOY)
    assert np.array_equal(e.net.Sigma, [[4, 2], [2, 4]])
    with pytest.raises(ValueError):
        GINet("bad", *TOY, arrivals=("lognormal", 1.0))                                   # not Condition 1


def test_exponential_clocks_match_genkern_r50():
    from gensim import Net
    from gensim_gi import GINet
    gi, ex = GINet("toy-gi-exp", *TOY), Net("toy", *TOY)
    a = [gi.run(50, 20, 0.1, 60, 0.24, s) for s in range(1, 9)]
    b = [ex.run(50, 20, 0.1, 60, 0.24, s) for s in range(101, 109)]
    for k in ("JE", "gap", "E_frac"):
        x, y = np.array([o[k] for o in a]), np.array([o[k] for o in b])
        z = (x.mean() - y.mean()) / np.hypot(x.std(ddof=1) / np.sqrt(8), y.std(ddof=1) / np.sqrt(8))
        assert abs(z) < 3.0, f"{k}: GI {x.mean():.4f} vs C {y.mean():.4f}, z={z:.2f}"
    assert all(o["events"] > 0 for o in a)


def test_erlang_h2_run_is_sane():
    from gensim_gi import GINet
    g = GINet("toy-gi", *TOY, arrivals=("h2", 2.0), services=("erlang", 2))
    o = g.run(100, 10, 0.1, 60, 0.24, 1)
    assert o["JE"] > 0 and np.isfinite(o["hgi_path"]) and 0 <= o["E_frac"] <= 1
    assert o["scv_u"] == [2.0] * 3 and o["scv_v"] == [0.5] * 3
