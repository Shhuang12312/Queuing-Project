"""Test networks shared by the suite (all satisfy Conditions 1-3 by construction)."""
import numpy as np

from rsn import RSN, make_2lln, make_3res_5type  # noqa: F401

K2 = [[1, 0, 1], [0, 1, 1]]
K35 = [[1, 0, 0, 1, 0], [0, 1, 0, 1, 1], [0, 0, 1, 0, 1]]
# the six networks of sim/networks.py (same primitives), usable without the .so
SIX = {
    "A toy (Ex. 2.2)":          (K2, [1, 1, 1], [1, 1, 1], [1, 1, 1], [-0.5] * 3),
    "B cost-trivial (Ex. 2.1)": (K2, [1, 1, 1], [1, 1, 1], [1, 1, 2], [-0.5] * 3),
    "C shared type costly":     (K2, [1, 1, 1], [1, 1, 1], [1, 1, 5], [-0.5] * 3),
    "D unequal rates":          (K2, [0.7, 1.0, 1.2], [1.0, 1.3, 1.0], [1.5, 1.0, 1.2], [-0.4, -0.5, -0.6]),
    "E unequal loads, costs":   (K2, [1.0, 0.5, 0.8], [1, 1, 1], [2.0, 1.0, 1.5], [-0.3, -0.6, -0.5]),
    "F 3x5 network":            (K35, [1] * 5, [1] * 5, [3, 1, 3, 1, 1], [-0.5] * 5),
}


# network G of sim/networks.py (Example 3 of Budhiraja and Johnson (2020), Math. Oper. Res. 45(3), p. 812 = Example 5.6 of
# arXiv:1710.09042; alpha = 1, eta = -1/2 chosen). The key keeps its historical label: stored result files use it.
K36 = [[1, 0, 0, 1, 0, 1], [0, 1, 0, 1, 1, 1], [0, 0, 1, 0, 1, 1]]
SEVEN = dict(SIX, **{"G ranking-map fails (BJ20 Ex. 6.6)": (K36, [1] * 6, [1] * 6, [5, 5, 5, 7, 8, 11], [-0.5] * 6)})


def six_networks():
    return [(name, RSN(*spec, name=name)) for name, spec in SIX.items()]


def seven_networks():
    return [(name, RSN(*spec, name=name)) for name, spec in SEVEN.items()]


def random_admissible(rng, I, extra, hetero=True):
    """I local types + `extra` random shared columns; heterogeneous alpha, beta, h, eta."""
    cols = [np.eye(I)[:, i] for i in range(I)]
    for _ in range(extra):
        while True:
            c = (rng.random(I) < 0.5).astype(float)
            if c.sum() >= 1:
                break
        cols.append(c)
    K = np.stack(cols, axis=1)
    J = K.shape[1]
    if hetero:
        alpha, beta, h = rng.uniform(.5, 2, J), rng.uniform(.5, 2, J), rng.uniform(.5, 3, J)
    else:
        alpha, beta, h = np.ones(J), np.ones(J), rng.uniform(.5, 3, J)
    return RSN(K, alpha, beta, h, eta=-rng.uniform(.2, 1, J), name=f"rand(I={I},J={J})")


def twenty_random():
    """The 20 random admissible networks of sim/stress_general.py (same seeds)."""
    out = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        out.append(random_admissible(rng, int(rng.integers(2, 5)), int(rng.integers(1, 4))))
    return out


def make_J18(I=14, J=18, seed=3, p=0.3):
    """A J = 18 network (14 local + 4 shared types) for the large-J path.

    Offline pass ~25 s. |M| = 72704 of 2^18; the eq. (6) cap |lambda~|/(4|beta||h|)
    is ~1e-4, far below rho*/4 ~ 0.13, so it binds.
    """
    rng = np.random.default_rng(seed)
    cols = [np.eye(I)[:, i] for i in range(I)]
    seen = set()
    while len(cols) < J:
        c = (rng.random(I) < p).astype(float)
        if c.sum() >= 2 and tuple(c) not in seen:
            seen.add(tuple(c))
            cols.append(c)
    K = np.stack(cols, axis=1)
    alpha = rng.uniform(0.6, 1.4, J)
    beta = rng.uniform(0.8, 1.6, J)
    h = rng.uniform(0.5, 2.5, J)
    return RSN(K, alpha, beta, h, eta=-rng.uniform(0.3, 0.8, J), name=f"J{J}")
