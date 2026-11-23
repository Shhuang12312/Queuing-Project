# Standalone module with plain imports; pure Python. The compiled kernels are in sim/.

"""
rsn.py — general Resource Sharing Network (I resources, J job types).

Implements the model of Budhiraja & Johnson (2024), "Simple form control
policies for resource sharing networks with HGI performance", for an
arbitrary network (K, alpha, beta, h); the toy of Example 2.2 is `make_2lln()`.
Verifies the paper's standing assumptions programmatically:

  Condition 2 (heavy traffic / stability), in the paper's form:
      alpha^r = alpha + alpha_bar/r,   beta^r = beta + beta_bar/r,
      rho = alpha/beta,  C := K rho  (criticality is automatic by choice of C),
      eta = beta^{-2} (alpha_bar beta - alpha beta_bar)  [= lim r (rho^r - rho)],
      theta = K eta < 0  (drift of the limiting workload RBM) — asserted.
    Input is EITHER eta (default; the canonical split alpha_bar = beta*eta,
    beta_bar = 0, so rho^r = rho + eta/r exactly and beta^r = beta) OR the pair
    (alpha_bar, beta_bar). Only eta enters the limit (theta, Sigma, HGI); the
    split changes the rth system at O(1/r^2) in rho^r. The simulator must use
    beta^r (it does: `betas_r`).
    Standing assumption on the scale (text after Definition 2.3):
    |rho^r - rho| < rho*/4 componentwise — `check_r(r)` raises otherwise.

  Condition 3 (local traffic): every resource i has a job type whose
      K-column is e_i. Recorded as s_i (used by the constructive v^b
      fallback and required for the workload state space to be all of R^I_+).

Covariance of the limiting RBM (exponential primitives):
      Sigma = K M (Sigma_u + Sigma_v R) M^T K^T,
  where Sigma_u, Sigma_v are the RENEWAL VARIANCE RATES
      Sigma_u_j = alpha_j^3 (sigma_u_j)^2  = alpha_j   (exponential),
      Sigma_v_j = beta_j^3  (sigma_v_j)^2  = beta_j    (exponential),
  R = diag(rho), M = diag(1/beta). For alpha=beta=1 this is 2 K K^T.
"""

import numpy as np


class RSN:
    def __init__(self, K, alpha, beta, h, eta=None, name="RSN",
                 alpha_bar=None, beta_bar=None, scv_u=None, scv_v=None):
        self.name  = str(name)
        self.K     = np.asarray(K, dtype=float)
        self.I, self.J = self.K.shape
        self.alpha = np.asarray(alpha, dtype=float).reshape(self.J)
        self.beta  = np.asarray(beta,  dtype=float).reshape(self.J)
        self.h     = np.asarray(h,     dtype=float).reshape(self.J)

        if np.any(self.alpha <= 0) or np.any(self.beta <= 0) or np.any(self.h <= 0):
            raise ValueError("alpha, beta, h must be strictly positive")
        if not np.all(np.isin(self.K, (0.0, 1.0))):
            raise ValueError("K must be a 0/1 incidence matrix")
        if np.any(self.K.sum(axis=0) == 0):
            raise ValueError("every job type must use at least one resource")

        # ---- nominal load and capacities (criticality by construction) ----
        self.rho      = self.alpha / self.beta
        self.C        = self.K @ self.rho
        self.rho_star = float(np.min(self.rho))
        self.M        = np.diag(1.0 / self.beta)
        self.R        = np.diag(self.rho)          # diag(rho): NOT routing!

        # ---- Condition 3: local traffic ----
        self.local_types = self._find_local_types()   # s_i for each resource i
        if np.linalg.matrix_rank(self.K) != self.I:
            raise ValueError("K must have full row rank (implied by Condition 3)")

        # ---- Condition 2: eta, or the paper's (alpha_bar, beta_bar) ----
        self.eta, self.alpha_bar, self.beta_bar = self._condition2(eta, alpha_bar, beta_bar)
        self.theta = self.K @ self.eta
        if not np.all(self.theta < 0):
            raise ValueError(
                f"theta = K eta = {self.theta} is not strictly negative (Condition 2): the "
                f"limiting RBM has no stationary distribution and HGI_E is undefined. "
                f"Choose eta (equivalently alpha_bar, beta_bar) with K eta < 0.")

        # ---- RBM covariance (renewal variance rates) ----
        # Sigma_u_j = alpha_j^3 sigma_u_j^2 = alpha_j * scv_u_j,  Sigma_v_j = beta_j^3 sigma_v_j^2
        # = beta_j * scv_v_j  (scv = squared coefficient of variation; 1 for exponential,
        # 1/k for Erlang-k, > 1 for hyperexponential).
        self.scv_u = (np.ones(self.J) if scv_u is None
                      else np.broadcast_to(np.asarray(scv_u, float), (self.J,)).copy())
        self.scv_v = (np.ones(self.J) if scv_v is None
                      else np.broadcast_to(np.asarray(scv_v, float), (self.J,)).copy())
        if np.any(self.scv_u <= 0) or np.any(self.scv_v <= 0):
            raise ValueError("scv_u, scv_v must be positive")
        Sigma_u = np.diag(self.alpha * self.scv_u)
        Sigma_v = np.diag(self.beta * self.scv_v)
        self.Sigma = self.K @ self.M @ (Sigma_u + Sigma_v @ self.R) \
                     @ self.M.T @ self.K.T

    # ------------------------------------------------------------------
    def _condition2(self, eta, alpha_bar, beta_bar):
        """Resolve (eta) or (alpha_bar, beta_bar) into all three, consistently."""
        J = self.J
        if alpha_bar is None and beta_bar is None:
            eta = (np.full(J, -0.5) if eta is None
                   else np.asarray(eta, dtype=float).reshape(J))
            return eta, self.beta * eta, np.zeros(J)            # canonical split
        a_bar = (np.zeros(J) if alpha_bar is None
                 else np.asarray(alpha_bar, dtype=float).reshape(J))
        b_bar = (np.zeros(J) if beta_bar is None
                 else np.asarray(beta_bar, dtype=float).reshape(J))
        eta_from_bars = (a_bar * self.beta - self.alpha * b_bar) / self.beta ** 2
        if eta is not None:
            eta = np.asarray(eta, dtype=float).reshape(J)
            if not np.allclose(eta, eta_from_bars, rtol=1e-9, atol=1e-12):
                raise ValueError(
                    f"eta={eta} is inconsistent with (alpha_bar, beta_bar): "
                    f"beta^-2 (alpha_bar beta - alpha beta_bar) = {eta_from_bars}")
        return eta_from_bars, a_bar, b_bar

    def _find_local_types(self):
        local = {}
        for i in range(self.I):
            for j in range(self.J):
                col = self.K[:, j]
                if col[i] == 1.0 and col.sum() == 1.0:
                    local[i] = j
                    break
        missing = [i for i in range(self.I) if i not in local]
        if missing:
            raise ValueError(
                f"Local traffic condition (Condition 3) fails for resources "
                f"{missing}: each resource needs a job type using it alone.")
        return local

    # ------------------------------------------------------------------
    def alphas_r(self, r):
        """alpha^r = alpha + alpha_bar/r  (canonical split: alpha + beta*eta/r)."""
        return self.alpha + self.alpha_bar / float(r)

    def betas_r(self, r):
        """beta^r = beta + beta_bar/r  (canonical split: beta_bar = 0, beta^r = beta)."""
        return self.beta + self.beta_bar / float(r)

    def rhos_r(self, r):
        """rho^r = alpha^r / beta^r; r (rho^r - rho) -> eta."""
        return self.alphas_r(r) / self.betas_r(r)

    def check_r(self, r):
        """
        The paper's standing assumption on the scale (text after Definition 2.3):
        |rho^r - rho| < rho*/4 componentwise, rho^r = alpha^r/beta^r. Raises
        ValueError if r is too small (or if alpha^r, beta^r are not positive);
        returns rho^r otherwise.
        """
        r = float(r)
        if not r > 0:
            raise ValueError(f"r must be positive, got {r!r}")
        a_r, b_r = self.alphas_r(r), self.betas_r(r)
        if np.any(a_r <= 0) or np.any(b_r <= 0):
            raise ValueError(f"alpha^r={a_r}, beta^r={b_r} must be positive at r={r:g}")
        rho_r = a_r / b_r
        dev = np.abs(rho_r - self.rho)
        if not np.all(dev < self.rho_star / 4.0):
            j = int(np.argmax(dev))
            raise ValueError(
                f"r={r:g} too small for heavy traffic: |rho^r_{j} - rho_{j}| = {dev[j]:.4g} "
                f">= rho*/4 = {self.rho_star / 4.0:.4g}; need r > "
                f"{self.r_min_heavy_traffic():.4g} (paper: |rho^r - rho| < rho*/4).")
        return rho_r

    def r_min_heavy_traffic(self):
        """Smallest r with |rho^r - rho| < rho*/4 (exact 4 max|eta|/rho* when beta^r = beta)."""
        def ok(rr):
            a_r, b_r = self.alphas_r(rr), self.betas_r(rr)
            return bool(np.all(a_r > 0) and np.all(b_r > 0)
                        and np.all(np.abs(a_r / b_r - self.rho) < self.rho_star / 4.0))
        r = max(4.0 * float(np.max(np.abs(self.eta))) / self.rho_star, 1e-12)
        if ok(r * (1 + 1e-9)) and not ok(r * (1 - 1e-9)):
            return r                                   # the canonical-split formula is exact
        lo = hi = r                                    # otherwise bracket and bisect
        while not ok(hi):
            hi *= 2.0
        while ok(lo) and lo > 1e-12:
            lo /= 2.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if ok(mid) else (mid, hi)
        return hi

    def project_capacity(self, b, max_iter=20, tol=1e-10):
        """
        Project allocation b onto {b >= 0 : K b <= C} by per-station scaling.
        DIAGNOSTIC ONLY: the policy does not call this (Definition 2.3's x(z)
        is feasible by construction and policy.allocation asserts it).
        """
        b = np.asarray(b, dtype=float).clip(min=0.0)
        for _ in range(max_iter):
            usage = self.K @ b
            over  = usage - self.C
            if np.all(over <= tol):
                break
            for i in range(self.I):
                if over[i] > tol and usage[i] > 0.0:
                    A_i = self.K[i] > 0.0
                    b[A_i] *= float(self.C[i] / usage[i])
        return b.clip(min=0.0)

    def workload(self, q):
        """W = K M q  (I-dimensional workload from queue lengths)."""
        return self.K @ self.M @ np.asarray(q, dtype=float)

    def __repr__(self):
        split = ("canonical split (alpha_bar = beta*eta, beta_bar = 0)"
                 if np.all(self.beta_bar == 0) else
                 f"alpha_bar={self.alpha_bar}  beta_bar={self.beta_bar}")
        return (f"{self.name}(I={self.I}, J={self.J})\n"
                f"  alpha={self.alpha}\n  beta ={self.beta}\n"
                f"  h    ={self.h}\n  rho  ={self.rho}  C={self.C}\n"
                f"  eta  ={self.eta}  [{split}]\n  theta={self.theta}  [stable]\n"
                f"  local types s_i = {self.local_types}")


def make_2lln(h=(1.0, 1.0, 1.0), eta=(-0.5, -0.5, -0.5), **cond2):
    """The paper's Example 2.2 toy: S1 serves B1,B3; S2 serves B2,B3.
    `cond2` may carry alpha_bar/beta_bar instead of eta (then pass eta=None)."""
    K = [[1.0, 0.0, 1.0],
         [0.0, 1.0, 1.0]]
    return RSN(K, alpha=[1, 1, 1], beta=[1, 1, 1], h=h, eta=eta, name="2LLN", **cond2)


def make_3res_5type(h=(1.0, 1.0, 1.0, 1.0, 1.0), eta=None, **cond2):
    """
    A larger test network: I=3 resources, J=5 job types.
      B1, B2, B3 local to S1, S2, S3;  B4 uses S1+S2;  B5 uses S2+S3.
    dim ker(K) = 2, so C^h_K is 1-dimensional and the cost geometry is
    nontrivial for generic h.
    """
    K = [[1.0, 0.0, 0.0, 1.0, 0.0],
         [0.0, 1.0, 0.0, 1.0, 1.0],
         [0.0, 0.0, 1.0, 0.0, 1.0]]
    if eta is None and not cond2:
        eta = [-0.5] * 5
    return RSN(K, alpha=[1] * 5, beta=[1] * 5, h=h, eta=eta, name="3res5type", **cond2)
