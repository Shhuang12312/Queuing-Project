"""Exact stationary flag fraction for a local queue of the toy under Definition 2.3.

For local queue j, whenever z_j = 1 (q_j < U) its service rate is d = x_j(z),
which is the same for every z with z_j = 1 (0.9769 at gamma_b = 0.98), and
arrivals are a = alpha^r_j. Flux balance across each level below U (exact for
the marginal of q_j, whatever the other queues do) gives, with psi = a/d,
    p_{kb+m} = p_kb (psi^{m+1} - 1)/(psi - 1),   m = 0..n-1,  n = ku - kb,
flag onset rate nu = d p_kb, flag duration (n+1)/a, hence
    P(E_j = 1) = f_j * g,   g = d (n+1) / (a S),   S = sum_m (psi^{m+1}-1)/(psi-1),
where f_j = P(q_j < U, E_j = 0). kb = ceil(L), ku = ceil(U).
"""
import math
import numpy as np


def g_flag(L, U, a, d):
    kb, ku = math.ceil(L - 1e-12), math.ceil(U - 1e-12)
    n = ku - kb
    lpsi = math.log(a / d)
    m = np.arange(1, n + 1)
    # S = sum (psi^m - 1)/(psi - 1), computed in logs for large n
    S = float(np.sum(np.expm1(m * lpsi))) / math.expm1(lpsi)
    return d * (n + 1) / (a * S)


def toy_params(r, c1, c2, kappa, d=0.976896):   # = toysim3.TAB.x((1, 0, 0))[0], gamma_b = 0.98
    rk = r ** kappa
    return c1 * rk, c2 * rk, 1.0 - 0.5 / r, d
