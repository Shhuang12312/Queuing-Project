# Standalone module with plain imports; pure Python. The compiled kernels are in sim/.

"""
policy.py — the control policy of Definition 2.3, for an arbitrary RSN.

Online state per job type j: queue length Q_j and a hysteresis flag
E_j in {0,1}:
    E_j: 0 -> 1  when Q_j drops below  L = c1~ r^kappa   (service stops),
    E_j: 1 -> 0  when Q_j climbs to    U = c2~ r^kappa   (service resumes),
with c_i~ = (min_j beta_j) * c_i. At every instant:

    Z_j = 1{ Q_j < U }
    x   = rho - v^c(Z) 1{Z in M} - v^b(Z)      (offline lookup tables)
    b_j = x_j * 1{E_j = 0}

The whole online policy is one table lookup and a multiplication; all the
geometry (u, M, v^c, v^b) lives in offline.OfflineTables.

Checks (the paper's standing assumptions, raised — never warned or patched):
  * x(Z) must satisfy K x <= C and x >= 0 — true by construction (asserted
    offline for every tabulated pattern), so `allocation` ASSERTS it instead of
    projecting b back onto the capacity set. A violation raises FeasibilityError and is counted in
    `n_violations`.
  * r must be large enough: |rho^r - rho| < rho*/4 componentwise (RSN.check_r)
    and (c2~ - c1~) r^kappa > 1 (a hysteresis band of at least one job).
    Both raise ValueError.
"""

import numpy as np
from offline import OfflineTables


class FeasibilityError(AssertionError):
    """x(Z) left the capacity set — impossible by construction; a bug if seen."""


class GeneralThresholdPolicy:
    def __init__(self, rsn, c1=1.0, c2=4.0, kappa=0.15, gamma=0.9,
                 tables=None, E_mask=None):
        assert 0.0 < c1 < c2, "need 0 < c1 < c2 (hysteresis)"
        assert 0.0 < kappa < 0.25, "paper requires kappa in (0, 1/4)"
        self.rsn = rsn
        self.c1, self.c2, self.kappa = float(c1), float(c2), float(kappa)
        self.beta_min = float(np.min(rsn.beta))
        self.tables = OfflineTables(rsn, gamma=gamma) if tables is None else tables
        self.E = np.zeros(rsn.J)
        # E_mask: bitmask of buffers that use hysteresis flags (default: all).
        # E_mask=0 disables the shutoff entirely: a labeled CONTROL EXPERIMENT
        # outside Definition 2.3, never a default (see docs/findings_toy_convergence.md).
        self.E_mask = (1 << rsn.J) - 1 if E_mask is None else int(E_mask)
        self._masked = np.array([(self.E_mask >> j) & 1 for j in range(rsn.J)], dtype=bool)
        self._checked_r = None          # r for which the standing assumptions were verified
        self.n_allocations = 0
        self.n_violations = 0
        self.feas_tol = 1e-9

    # ------------------------------------------------------------------
    def check_r(self, r):
        """Raise unless r satisfies the paper's standing assumptions for this policy."""
        r = float(r)
        self.rsn.check_r(r)                                   # |rho^r - rho| < rho*/4
        band = (self.c2 - self.c1) * self.beta_min * r ** self.kappa
        if not band > 1.0:
            raise ValueError(
                f"(c2~ - c1~) r^kappa = {band:.3f} <= 1 at r={r:g}: the hysteresis band "
                f"must exceed one job (paper, after Definition 2.3). Increase r, c2 - c1 "
                f"or kappa; r > {(1.0 / ((self.c2 - self.c1) * self.beta_min)) ** (1.0 / self.kappa):.3g} is needed.")
        self._checked_r = r

    def thresholds(self, r):
        if r != self._checked_r:
            self.check_r(r)
        rk = float(r) ** self.kappa
        return self.c1 * self.beta_min * rk, self.c2 * self.beta_min * rk

    def reset_E(self, value=0.0):
        self.E[:] = float(value)

    # ------------------------------------------------------------------
    def allocation(self, q, r):
        """Returns (b, Z, E). b >= 0 and K b <= C hold by construction and are asserted."""
        q = np.asarray(q, dtype=float)
        L, U = self.thresholds(r)

        Z = (q < U - 1e-12).astype(int)
        masked = self._masked
        self.E[masked & (q < L - 1e-12)] = 1.0
        self.E[masked & (q >= U - 1e-12)] = 0.0
        E = self.E.copy()

        x = self.tables.x(Z)                    # rho - v^c 1{Z in M} - v^b(Z)
        self.n_allocations += 1
        rsn = self.rsn
        if np.any(x < -self.feas_tol) or np.any(rsn.K @ x > rsn.C + self.feas_tol):
            self.n_violations += 1
            raise FeasibilityError(
                f"x(Z) infeasible at Z={tuple(Z)}: x={x}, Kx={rsn.K @ x}, C={rsn.C} "
                f"(offline tables are asserted; this should be impossible)")
        b = (1.0 - E) * x
        return b, Z, E
