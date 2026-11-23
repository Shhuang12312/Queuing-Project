"""Statistical cross-check of rsn_flat (pure Python) against sim/genkern (C) at r = 50.

Same process, independent random streams: window means must agree within noise,
|z| < 3 on J_E, the SSC gap (J_E - h_hat level) and the E-fraction.
Constants (0.1, 60, 0.24): stationary at r = 50 (predicted flag drift 8% of |theta|).
Pure Python: 4 seeds x T = 20 (~6 s each); C: 12 seeds x T = 20.
"""
import os

import numpy as np
import pytest

from rsn import make_2lln
from offline import OfflineTables
from je import run_JE

SIM = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "sim")
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(SIM, "genkern.so")),
                                reason="sim/genkern.so not built (cd sim && ./build.sh)")

R, T, C1, C2, KAPPA = 50.0, 20.0, 0.1, 60.0, 0.24
HHAT = lambda W: np.maximum(W[:, 0], W[:, 1])   # noqa: E731  (analytic toy h_hat)


def _mean_se(x):
    x = np.asarray(x, float)
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


def test_rsn_flat_vs_genkern_r50():
    from gensim import Net
    net = make_2lln()
    tab = OfflineTables(net, gamma_c=0.98, gamma_b=0.98)
    py = [run_JE(net, r=R, T=T, c1=C1, c2=C2, kappa=KAPPA, seed=s, n_snapshots=800,
                 hhat_fn=HHAT, tables=tab) for s in (11, 12, 13, 14)]
    nw = Net("toy", [[1, 0, 1], [0, 1, 1]], [1, 1, 1], [1, 1, 1], [1, 1, 1], [-0.5] * 3)
    c = [nw.run(int(R), T, C1, C2, KAPPA, s) for s in range(101, 113)]
    report = []
    for key_py, key_c in (("JE", "JE"), ("ssc_gap", "gap"), ("E_frac", "E_frac")):
        m1, s1 = _mean_se([o[key_py] for o in py])
        m2, s2 = _mean_se([o[key_c] for o in c])
        z = (m1 - m2) / np.hypot(s1, s2)
        report.append(f"{key_py}: python {m1:.4f}±{s1:.4f} (4 seeds) vs C {m2:.4f}±{s2:.4f} (12 seeds), z={z:.2f}")
        assert abs(z) < 3.0, "\n".join(report)
    print("\n" + "\n".join(report))
    assert all(o["n_violations"] == 0 for o in py) if "n_violations" in py[0] else True
