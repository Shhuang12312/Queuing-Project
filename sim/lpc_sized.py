"""Does building the size condition into LP-c (instead of box + rescale)
enlarge |lambda~| and hence b_max, while staying inside Definition 2.3?"""
import itertools, sys
import numpy as np
from scipy.optimize import linprog
import os as _os
FLAT = sys.argv[1] if len(sys.argv) > 1 else _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "rsn_flat")
sys.path.insert(0, FLAT)
from rsn import RSN, make_2lln, make_3res_5type
from offline import OfflineTables


class SizedLP(OfflineTables):
    """LP-c: max v.u  s.t. Kv=0, v_j>=0 (z_j=1), v_j <= gamma_c(rho_j - rho*/2).
    The size condition rho - v^c > rho*/2 is a bound, so the optimizer is
    already admissible -- no rescaling. Bounded because {v in ker K, v<=0}={0}."""
    def _solve_lp_c(self, z):
        if self.trivial:
            return 0.0, None
        n = self.rsn
        ub = self.gamma_c * (n.rho - n.rho_star / 2.0)
        bounds = [(0.0, ub[j]) if z[j] else (None, ub[j]) for j in range(n.J)]
        res = linprog(-self.u, A_eq=n.K, b_eq=np.zeros(n.I), bounds=bounds, method="highs")
        return (float(-res.fun), res.x) if res.success else (0.0, None)

    def _make_vc(self, z):
        t, v = self._solve_lp_c(z)
        return (True, v) if t > self.tol else (False, np.zeros(self.rsn.J))


def random_network(rng, I, extra):
    cols = [np.eye(I)[:, i] for i in range(I)]
    for _ in range(extra):
        while True:
            c = (rng.random(I) < 0.5).astype(float)
            if c.sum() >= 1: break
        cols.append(c)
    K = np.stack(cols, axis=1); J = K.shape[1]
    return RSN(K, rng.uniform(.5, 2, J), rng.uniform(.5, 2, J),
               rng.uniform(.5, 3, J), eta=-rng.uniform(.2, 1, J), name="rand")


nets = [("toy", make_2lln()), ("3res5type", make_3res_5type())]
for s in range(20):
    rng = np.random.default_rng(s)
    nets.append((f"rand{s}", random_network(rng, int(rng.integers(2, 5)), int(rng.integers(1, 4)))))
print(f"{'network':10s} {'b_max now':>10s} {'b_max sized':>12s} {'gain':>6s}  M same")
gains = []
for name, net in nets:
    a = OfflineTables(net, gamma_c=0.98, gamma_b=0.98)
    b = SizedLP(net, gamma_c=0.98, gamma_b=0.98)        # asserts Kx<=C, x>=rho*/4
    same = a.M_set == b.M_set
    g = b._get_b_max() / a._get_b_max()
    gains.append(g)
    print(f"{name:10s} {a._get_b_max():10.4f} {b._get_b_max():12.4f} {g:6.2f}x  {same}")
print(f"median gain {np.median(gains):.2f}x, max {max(gains):.2f}x, min {min(gains):.2f}x")
