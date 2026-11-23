"""Non-exponential primitives: event-driven simulation with residual clocks (genkern_gi.c).

    from gensim_gi import GINet
    nw = GINet("toy", K, alpha, beta, h, eta, arrivals=("h2", 2.0), services=("erlang", 2))
    out = nw.run(1000, 40, 0.1, 60, 0.24, seed=1)      # same fields as gensim.Net.run
    nw.benchmark()                                      # HGI_E for the matching Sigma

Distribution specs (Condition-1 families only — exponential moments; NOT lognormal/Pareto):
    "exp"                 exponential (SCV 1)
    ("erlang", k)         Erlang-k, SCV 1/k
    ("h2", scv)           balanced-means hyperexponential H2 with the given SCV > 1
A spec applies to every type; a list of J specs sets them per type. Interarrival means are
1/alpha^r_j and service-requirement means 1/beta^r_j (Condition 2 as in gensim.Net).
The limit covariance is Sigma = K M (Sigma^u + Sigma^v R) M^T K^T with the renewal variance
rates Sigma^u_j = alpha_j scv_u_j, Sigma^v_j = beta_j scv_v_j (rsn.RSN handles this).
"""
import ctypes
import math
import os

import numpy as np

import gensim
from gensim import Net, _D

HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(os.environ.get("RSN_GENKERN_GI_SO", os.path.join(HERE, "genkern_gi.so")))
        lib.simulate_gi.argtypes = ([ctypes.c_int, ctypes.c_int] + [_D] * 5 + [ctypes.c_int, _D] + [_D] * 2
                                    + [ctypes.c_double] * 5 + [ctypes.c_uint64, _D])
        _LIB = lib
    return _LIB


def h2_balanced(rate, scv):
    """Balanced-means H2 with mean 1/rate and squared CV scv > 1: (p, mu1, mu2)."""
    if scv <= 1.0:
        raise ValueError("H2 needs scv > 1")
    p = 0.5 * (1.0 + math.sqrt((scv - 1.0) / (scv + 1.0)))
    return p, 2.0 * p * rate, 2.0 * (1.0 - p) * rate


def _normalize(spec, J):
    """One spec for all types ("exp" or (kind, param)) or a list of J specs."""
    single = isinstance(spec, str) or (isinstance(spec, (tuple, list)) and len(spec) >= 1
                                       and isinstance(spec[0], str))
    if single:
        return [spec] * J
    if len(spec) != J:
        raise ValueError(f"need one primitive spec or {J} of them, got {len(spec)}")
    return list(spec)


def dist_code(spec, rate):
    """[code, p1, p2, p3] for one type with the given rate (mean 1/rate), plus its SCV."""
    if spec == "exp" or spec == ("exp",):
        return [0.0, float(rate), 0.0, 0.0], 1.0
    kind = spec[0].lower()
    if kind == "erlang":
        k = int(spec[1])
        if k < 1:
            raise ValueError("Erlang needs k >= 1")
        return [1.0, float(k), float(k * rate), 0.0], 1.0 / k
    if kind == "h2":
        p, mu1, mu2 = h2_balanced(float(rate), float(spec[1]))
        return [2.0, p, mu1, mu2], float(spec[1])
    raise ValueError(f"unknown primitive family {spec!r} (allowed: 'exp', ('erlang', k), ('h2', scv))")


class GINet(Net):
    def __init__(self, name, K, alpha, beta, h, eta=None, gamma=0.98, alpha_bar=None, beta_bar=None,
                 tables=None, arrivals="exp", services="exp"):
        J = np.asarray(K).shape[1]
        self.arr_spec, self.srv_spec = _normalize(arrivals, J), _normalize(services, J)
        scv_u = [dist_code(s_, 1.0)[1] for s_ in self.arr_spec]
        scv_v = [dist_code(s_, 1.0)[1] for s_ in self.srv_spec]
        super().__init__(name, K, alpha, beta, h, eta=eta, gamma=gamma, alpha_bar=alpha_bar,
                         beta_bar=beta_bar, tables=tables, scv_u=scv_u, scv_v=scv_v)

    def primitives(self, r):
        """The 4J-vectors of coded distributions at scale r (rates alpha^r, beta^r)."""
        n = self.net
        ar, br = n.alphas_r(r), n.betas_r(r)
        arr = np.array([dist_code(s_, ar[j])[0] for j, s_ in enumerate(self.arr_spec)]).ravel()
        srv = np.array([dist_code(s_, br[j])[0] for j, s_ in enumerate(self.srv_spec)]).ravel()
        return np.ascontiguousarray(arr), np.ascontiguousarray(srv)

    def run(self, r, T, c1, c2, kappa, seed, burn=0.3):
        n = self.net
        n.check_r(r)
        L, U = self.thresholds(r, c1, c2, kappa)
        if not (U - L) > 1.0:
            raise ValueError(f"(c2~ - c1~) r^kappa = {U - L:.3f} <= 1 at r={r}")
        arr = lambda a: np.ascontiguousarray(a, dtype=float)
        K, ib = arr(n.K.ravel()), arr(1 / n.beta)
        h, C, X, V = arr(n.h), arr(n.C), arr(self.xtab.ravel()), arr(self.V.ravel())
        A, S = self.primitives(r)
        out = np.zeros(8 + 2 * n.J + n.I)
        T_end = r * r * T
        p = lambda a: a.ctypes.data_as(_D)
        _lib().simulate_gi(n.I, n.J, p(K), p(ib), p(h), p(C), p(X), len(self.V), p(V), p(A), p(S),
                           L, U, float(r), T_end, burn * T_end, int(seed), p(out))
        win = out[7]; J, I = n.J, n.I
        return dict(net=self.name, r=r, T=T, c=(c1, c2, kappa), seed=seed, L=L, U=U,
                    arrivals=str(self.arr_spec[0]) if len(set(map(str, self.arr_spec))) == 1 else "per-type",
                    services=str(self.srv_spec[0]) if len(set(map(str, self.srv_spec))) == 1 else "per-type",
                    scv_u=list(n.scv_u), scv_v=list(n.scv_v),
                    JE=out[0] / win / r, level=out[1] / win / r, hgi_path=out[2] / win / r,
                    gap=(out[0] - out[1]) / win / r, E_frac=out[3] / win,
                    JE_2nd=out[4] / (0.5 * win) / r, JE_1st=(out[0] - out[4]) / (0.5 * win) / r,
                    p_flag=list(out[8:8 + J] / win), f_below=list(out[8 + J:8 + 2 * J] / win),
                    idle_drift=list(r * out[8 + 2 * J:8 + 2 * J + I] / win / np.abs(n.theta)),
                    events=out[6])
