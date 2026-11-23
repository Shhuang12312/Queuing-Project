# Standalone module with plain imports; pure Python. The compiled kernels are in sim/.

"""
offline.py — the offline pass: the vectors of Definition 2.3 for an arbitrary network.

Everything the paper leaves as "there exists a vector such that ..." is
computed here, once, before simulation:

  1. cost geometry: orthonormal kernel basis N of ker(K) via SVD,
     g = h*beta, g_ker = N N^T g,
       Case A (trivial):   g_ker = 0  ->  v^c = 0, M_set = {} (only v^b active)
       Case B (nontrivial): u = -g_ker/|g_ker|, lambda = g.u = -|g_ker| < 0
  2. the set M of Z-configurations where cost reduction is possible:
       z in M  <=>  max{ v.u : Kv=0, v_j>=0 if z_j=1, |v|<=1 } > tol   (LP-c)
  3. cost-reduction vectors v^c(z): scaled LP-c maximizer, obeying the
     size condition rho - v^c > rho*/2 componentwise;
     lambda_c(z) = lambda*(v^c(z).u) < 0,  lambda_tilde = max_z lambda_c(z)
  4. boundary-repulsion vectors v^b(z): LP-b
       max t  s.t. (Kv)_i = 0 for i in A_z, v_j >= t if z_j = 1,
                   |v| <= 1, t >= 0,
     scaled to norm gamma*b_max with
       b_max = min(rho*/4, |lambda_tilde| / (4 |beta|_2 |h|_2))   (Case B)
       b_max = rho*/4                                             (Case A)
     with the paper's constructive fallback (proof of Prop 2.3) if the LP
     ever returns t* ~ 0.
  5. feasibility assertions for every z:
       x(z) = rho - v^c(z) 1{z in M} - v^b(z)  satisfies
       K x <= C  and  x >= rho*/4 > 0.

Pattern-dependent cap (LABELED EXPERIMENT, off by default).
`vb_cap="per_pattern"` replaces the uniform eq. (6) cap by
    ||v^b(z)|| <= min(rho*/4, |lambda_c(z)| / (4 |beta|_2 |h|_2))   for z in M,
    ||v^b(z)|| <= rho*/4                                             for z not in M.
Rationale: in the proof of Proposition 3.5 (paper section 7.2) the size of
v^b enters only through step (47), which needs, for Z^r(s) in M,
    h beta . (v^c(z) + v^b(z)) <= -|lambda~| + |h||beta| max_z|v^b(z)| <= -3/4 |lambda~|;
with the per-pattern cap one gets the stronger pointwise chain
    h beta . (v^c(z) + v^b(z)) <= -3/4 |lambda_c(z)| <= -3/4 |lambda~|
(see `check_step47`). Off M, (47) is never invoked (Lemma 7.4 / Prop 7.7
handle those configurations) and only x >= rho*/4 (Prop 2.6(c)) is needed.
Everything produced with this option is OUTSIDE eq. (6) of Definition 2.3
and is labeled so (`label`, `summary()`); `b_max` always reports the eq. (6)
value, the per-pattern value is `b_cap(z)`.

Enumeration. M is a down-set of {0,1}^J: if z in M then every z' <= z
(componentwise) is in M, because the LP-c feasible region only grows when
constraints v_j >= 0 are dropped. Equivalently, if z is NOT in M then no
z' >= z is. The pass therefore walks the patterns by increasing popcount and
solves LP-c only for the patterns all of whose immediate predecessors
(z minus one bit) are members. The LP-c solves are exactly the ones the
brute-force enumeration would have found nonzero, plus the minimal
non-members, so M, every v^c(z), lambda_tilde and hence b_max are identical
to the full 2^J enumeration — for every J, with no "lazy" approximation.

The v^b table (one LP-b per pattern, 2^J of them) is precomputed and
asserted in full when J <= max_enumerate (default 16). Above that it is
filled on first visit and EVERY entry is asserted (K x <= C, x >= rho*/4)
the moment it is created; b_max is already exact at that point. Call
`enumerate_vb()` to force the full table.
"""

import itertools
import numpy as np
from scipy.optimize import linprog

_TOL = 1e-9


def kernel_basis(K, tol=1e-10):
    """Orthonormal basis (columns) of ker(K) via SVD."""
    K = np.asarray(K, dtype=float)
    _, s, Vt = np.linalg.svd(K)
    rank = int(np.sum(s > tol * max(K.shape) * (s[0] if len(s) else 1.0)))
    return Vt[rank:].T          # J x (J - rank)


class OfflineTables:
    VB_CAPS = ("uniform", "per_pattern")

    def __init__(self, rsn, gamma=0.9, max_enumerate=16, tol=_TOL,
                 vb_norm=None, gamma_c=None, gamma_b=None, vb_cap="uniform"):
        """
        gamma_c: safety factor for the v^c scaling toward the size condition
                 rho - v^c > rho*/2. NOTE (within Definition 2.3): b_max =
                 min(rho*/4, |lambda~|/(4|beta||h|)) depends on the CHOSEN
                 v^c through lambda~, so taking gamma_c close to 1 legally
                 enlarges the admissible ||v^b||. Default: `gamma`.
        gamma_b: fraction of b_max used for ||v^b(z)||. Default: `gamma`.
        vb_norm: CONTROL-EXPERIMENT override of ||v^b(z)|| ignoring b_max.
        Values above b_max leave the paper's construction (eq. (6)) — a run
        using it is a labeled control experiment OUTSIDE Definition 2.3, never
        a default; feasibility (Kx <= C, x >= rho*/4) is still asserted.
        max_enumerate: J above which the v^b table is filled on demand
        instead of precomputed. M, lambda~ and b_max are always exact.
        vb_cap: "uniform" (default) = eq. (6), ||v^b(z)|| = gamma_b * b_max for
        every z; "per_pattern" = the LABELED EXPERIMENT, OUTSIDE
        eq. (6): ||v^b(z)|| = gamma_b * b_cap(z) with
        b_cap(z) = min(rho*/4, |lambda_c(z)|/(4|beta||h|)) on M, rho*/4 off M.
        It preserves the inequality that step (47) of the proof needs
        (`check_step47`); whether it is admissible for Theorems 2.4-2.5 is
        the question put to the advisor (docs/note_for_advisor.md).
        """
        self.rsn     = rsn
        self.gamma   = float(gamma)
        self.gamma_c = float(gamma if gamma_c is None else gamma_c)
        self.gamma_b = float(gamma if gamma_b is None else gamma_b)
        self.tol     = float(tol)
        self.vb_norm = None if vb_norm is None else float(vb_norm)
        if vb_cap not in self.VB_CAPS:
            raise ValueError(f"vb_cap must be one of {self.VB_CAPS}, got {vb_cap!r}")
        self.vb_cap = vb_cap
        self.max_enumerate = int(max_enumerate)

        K, beta, h, rho = rsn.K, rsn.beta, rsn.h, rsn.rho
        self.N = kernel_basis(K)                       # J x (J-I)
        g = h * beta
        g_ker = self.N @ (self.N.T @ g)
        norm = float(np.linalg.norm(g_ker))

        self.trivial = norm < 1e-10                    # Case A
        if self.trivial:
            self.u, self.lam = None, 0.0
        else:                                          # Case B
            self.u   = -g_ker / norm                   # steepest cost descent in ker(K)
            self.lam = float(g @ self.u)               # = -norm < 0

        self._M         = set()     # members of M (tuples)
        self._vc_cache  = {}        # z -> v^c(z), members only
        self._lam_c     = {}        # z -> lambda_c(z), members only
        self._inM_cache = {}        # z -> bool, every z on which LP-c was solved
        self._vb_cache  = {}        # z -> v^b(z)
        self.lam_tilde  = None
        self.b_max      = None
        self.vb_enumerated = False

        self._build_M()                                # exact M, lambda~ (pruned)
        self._get_b_max()                              # exact b_max, eq. (6)
        if rsn.J <= self.max_enumerate:
            self.enumerate_vb()                        # full table, all asserted

    # ------------------------------------------------------------------
    # LP-c: cost-reduction direction for configuration z
    # ------------------------------------------------------------------
    def _solve_lp_c(self, z):
        """max v.u  s.t. Kv=0, v_j >= 0 where z_j=1, -1<=v<=1. Returns (t*, v*)."""
        if self.trivial:
            return 0.0, None
        rsn = self.rsn
        bounds = [(0.0, 1.0) if z[j] else (-1.0, 1.0) for j in range(rsn.J)]
        res = linprog(c=-self.u, A_eq=rsn.K, b_eq=np.zeros(rsn.I),
                      bounds=bounds, method="highs")
        if not res.success:
            return 0.0, None
        return float(-res.fun), np.asarray(res.x)

    def _make_vc(self, z):
        """v^c(z): scaled LP-c maximizer meeting rho - v^c > rho*/2."""
        t_star, v_star = self._solve_lp_c(z)
        in_M = (t_star > self.tol)
        if not in_M:
            return False, np.zeros(self.rsn.J)
        rho, rho_star = self.rsn.rho, self.rsn.rho_star
        pos = v_star > self.tol
        if np.any(pos):
            s = self.gamma_c * float(np.min((rho[pos] - rho_star / 2.0) / v_star[pos]))
        else:
            s = 1.0
        return True, s * v_star

    # ------------------------------------------------------------------
    # The set M by enumeration with monotone pruning (M is a down-set)
    # ------------------------------------------------------------------
    def _solve_member(self, zint):
        """Solve LP-c for the pattern zint (bit j = z_j); record the result."""
        J = self.rsn.J
        z = tuple((zint >> j) & 1 for j in range(J))
        in_M, vc = self._make_vc(z)
        self._inM_cache[z] = in_M
        if in_M:
            self._M.add(z)
            self._vc_cache[z] = vc
            self._lam_c[z] = self.lam * float(vc @ self.u)
        return in_M

    def _build_M(self):
        """
        Walk {0,1}^J by popcount. A pattern is a candidate (gets an LP-c
        solve) only if all its immediate predecessors are members; by the
        down-set property every member is reached this way and every
        pattern that is skipped is a non-member. Exact for every J.
        """
        if self.trivial:
            self.lam_tilde = None
            return
        J = self.rsn.J
        level = [0] if self._solve_member(0) else []
        for _k in range(1, J + 1):
            if not level:
                break
            prev = set(level)
            cands = set()
            for m in level:
                for j in range(J):
                    bit = 1 << j
                    if m & bit:
                        continue
                    c = m | bit
                    if c in cands:
                        continue
                    ok, cc = True, c
                    while cc:                      # every c minus one bit in prev?
                        b = cc & -cc
                        if (c ^ b) not in prev:
                            ok = False
                            break
                        cc ^= b
                    if ok:
                        cands.add(c)
            level = [c for c in sorted(cands) if self._solve_member(c)]
        self.lam_tilde = max(self._lam_c.values()) if self._lam_c else None

    # ------------------------------------------------------------------
    # LP-b: boundary repulsion for configuration z
    # ------------------------------------------------------------------
    def _A_z(self, z):
        """Busy resources: i such that some j with K_ij=1 has z_j=0."""
        K = self.rsn.K
        return [i for i in range(self.rsn.I)
                if any(K[i, j] == 1.0 and not z[j] for j in range(self.rsn.J))]

    def _solve_lp_b(self, z):
        """max t s.t. (Kv)_i=0 (i in A_z), v_j>=t (z_j=1), |v|<=1, 0<=t<=1."""
        rsn = self.rsn
        J = rsn.J
        A_z = self._A_z(z)
        ones_idx = [j for j in range(J) if z[j]]
        # variables: [v_1..v_J, t]
        c = np.zeros(J + 1); c[-1] = -1.0
        A_eq = None; b_eq = None
        if A_z:
            A_eq = np.hstack([rsn.K[A_z, :], np.zeros((len(A_z), 1))])
            b_eq = np.zeros(len(A_z))
        A_ub, b_ub = None, None
        if ones_idx:
            rows = []
            for j in ones_idx:                 # t - v_j <= 0
                row = np.zeros(J + 1); row[j] = -1.0; row[-1] = 1.0
                rows.append(row)
            A_ub = np.vstack(rows); b_ub = np.zeros(len(rows))
        bounds = [(-1.0, 1.0)] * J + [(0.0, 1.0)]
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
        if not res.success:
            return 0.0, None
        return float(res.x[-1]), np.asarray(res.x[:J])

    def _vb_constructive(self, z):
        """Paper's constructive proof of Prop 2.3 (fallback if LP-b degenerate)."""
        rsn = self.rsn
        J = rsn.J
        A_z = self._A_z(z)
        S = {rsn.local_types[i] for i in A_z}
        v = np.zeros(J)
        for j in range(J):
            if j in S:
                continue
            v[j] = 1.0 if z[j] else -float(J)
        for i in A_z:
            s_i = rsn.local_types[i]
            v[s_i] = -float(sum(rsn.K[i, j] * v[j] for j in range(J) if j != s_i))
        for i in range(rsn.I):
            s_i = rsn.local_types[i]
            if i not in A_z and z[s_i]:
                v[s_i] = 1.0
        return v

    def _make_vb(self, z):
        if not any(z):                          # z = 0: no repulsion needed
            return np.zeros(self.rsn.J)
        t_star, v_star = self._solve_lp_b(z)
        if v_star is None or t_star <= self.tol:
            v_star = self._vb_constructive(z)
        target = (self.vb_norm if self.vb_norm is not None
                  else self.gamma_b * self.b_cap(z))
        return target * v_star / float(np.linalg.norm(v_star))

    # ------------------------------------------------------------------
    def _get_b_max(self):
        """b_max = min(rho*/4, |lambda~|/(4|beta|_2|h|_2))  — eq. (6)."""
        if self.b_max is not None:
            return self.b_max
        rsn = self.rsn
        cap = rsn.rho_star / 4.0
        if not self.trivial and self.lam_tilde is not None:
            cap = min(cap, abs(self.lam_tilde)
                      / (4.0 * np.linalg.norm(rsn.beta) * np.linalg.norm(rsn.h)))
        self.b_max = cap
        return cap

    def b_cap(self, z):
        """Admissible ||v^b(z)|| before the gamma_b factor.

        "uniform":     b_max = min(rho*/4, |lambda~|/(4|beta||h|))        (eq. (6))
        "per_pattern": min(rho*/4, |lambda_c(z)|/(4|beta||h|)) for z in M,
                       rho*/4 for z not in M                     (OUTSIDE eq. (6))
        The two agree at every z in M attaining lambda~ and whenever rho*/4 binds.
        """
        if self.vb_cap == "uniform":
            return self._get_b_max()
        z = tuple(int(v) for v in z)
        rsn = self.rsn
        cap = rsn.rho_star / 4.0
        if z in self._M:
            cap = min(cap, abs(self._lam_c[z])
                      / (4.0 * np.linalg.norm(rsn.beta) * np.linalg.norm(rsn.h)))
        return cap

    @property
    def label(self):
        """How to label every output derived from these tables."""
        if self.vb_norm is not None:
            return f"CONTROL EXPERIMENT outside Definition 2.3: vb_norm={self.vb_norm}"
        if self.vb_cap == "per_pattern":
            return "outside eq. (6): per-pattern cap |lambda_c(z)|/(4|beta||h|) on M, rho*/4 off M"
        return "Definition 2.3, eq. (6)"

    def check_step47(self, tol=1e-9):
        """
        The inequality that step (47) of the proof of Proposition 3.5 needs, for
        every z in M (it is only applied there):
            h beta . (v^c(z) + v^b(z)) <= -3/4 |lambda_c(z)| <= -3/4 |lambda~|.
        Returns a dict with the worst (largest) value of the left side over M, the
        per-pattern margins, and `ok`. Raises AssertionError if it fails.
        """
        if self.trivial or not self._M:
            return dict(ok=True, n=0, worst_lhs=None, min_margin=None, note="Case A: M empty")
        g = self.rsn.h * self.rsn.beta
        lam_t = abs(self.lam_tilde)
        worst, margins, margins_tilde = -np.inf, [], []
        for z in self._M:
            lhs = float(g @ (self._vc_cache[z] + self.vb(z)))
            lam_z = abs(self._lam_c[z])
            margins.append(-0.75 * lam_z - lhs)           # >= 0 required
            margins_tilde.append(-0.75 * lam_t - lhs)     # >= 0 required (weaker)
            worst = max(worst, lhs)
        out = dict(ok=bool(min(margins) >= -tol and min(margins_tilde) >= -tol), n=len(self._M),
                   worst_lhs=worst, min_margin=float(min(margins)),
                   min_margin_tilde=float(min(margins_tilde)), lam_tilde=self.lam_tilde)
        if not out["ok"]:
            raise AssertionError(f"step-(47) inequality violated: {out}")
        return out

    def push_stats(self):
        """||v^b(z)|| over the tabulated patterns (z != 0): overall, on M and off M.
        Call enumerate_vb() first for the full table."""
        norms = {z: float(np.linalg.norm(v)) for z, v in self._vb_cache.items() if any(z)}
        def stats(vals):
            vals = np.array(sorted(vals))
            return (dict(n=len(vals), min=float(vals.min()), median=float(np.median(vals)),
                         max=float(vals.max())) if len(vals) else dict(n=0))
        return dict(all=stats(norms.values()),
                    in_M=stats([v for z, v in norms.items() if z in self._M]),
                    off_M=stats([v for z, v in norms.items() if z not in self._M]),
                    b_max_eq6=self._get_b_max(), vb_cap=self.vb_cap, label=self.label)

    def enumerate_vb(self):
        """Precompute v^b(z) for all 2^J patterns and assert every x(z)."""
        for z in itertools.product((0, 1), repeat=self.rsn.J):
            if z not in self._vb_cache:
                self._vb_cache[z] = self._make_vb(z)
        self.vb_enumerated = True
        self.assert_feasibility()

    # ------------------------------------------------------------------
    # public lookups
    # ------------------------------------------------------------------
    def in_M(self, z):
        return tuple(int(v) for v in z) in self._M

    def vc(self, z):
        z = tuple(int(v) for v in z)
        return self._vc_cache.get(z, np.zeros(self.rsn.J))

    def lam_c(self, z):
        """lambda_c(z) = lambda (v^c(z).u) for z in M, else 0."""
        return self._lam_c.get(tuple(int(v) for v in z), 0.0)

    def vb(self, z):
        z = tuple(int(v) for v in z)
        if z not in self._vb_cache:
            self._vb_cache[z] = self._make_vb(z)
            self._check_x(z)                    # every lazily created entry
        return self._vb_cache[z]

    def x(self, z):
        """Nominal service-rate vector x(z) = rho - v^c 1{z in M} - v^b(z)."""
        z = tuple(int(v) for v in z)
        x = self.rsn.rho.copy()
        if z in self._M:
            x = x - self._vc_cache[z]
        return x - self.vb(z)

    @property
    def M_set(self):
        return sorted(self._M)

    @property
    def n_lp_c(self):
        """Number of LP-c solves the pruned enumeration needed (|M| + minimal non-members)."""
        return len(self._inM_cache)

    # ------------------------------------------------------------------
    def _check_x(self, z, tol=1e-8):
        x = self.x(z)
        Kx = self.rsn.K @ x
        if not np.all(Kx <= self.rsn.C + tol):
            raise AssertionError(
                f"K x(z) <= C violated at z={z}: Kx={Kx}, C={self.rsn.C}")
        if not np.all(x >= self.rsn.rho_star / 4.0 - tol):
            raise AssertionError(f"x(z) >= rho*/4 violated at z={z}: x={x}")

    def assert_feasibility(self, tol=1e-8):
        """Paper Prop 2.6-style checks: Kx <= C and x >= rho*/4 for every tabulated z."""
        for z in list(self._vb_cache):
            self._check_x(z, tol)

    def summary(self):
        lines = [f"Offline pass for {self.rsn.name} "
                 f"(I={self.rsn.I}, J={self.rsn.J}, dim ker K={self.N.shape[1]})"]
        if self.trivial:
            lines.append("  Case A (trivial): h*beta ⟂ ker(K); v^c ≡ 0, M = {}")
        else:
            lines.append(f"  u_(J-I)   = {np.round(self.u, 6)}")
            lines.append(f"  lambda    = {self.lam:.6f}")
            if len(self._M) <= 32:
                lines.append(f"  M set     = {self.M_set}")
            else:
                lines.append(f"  |M|       = {len(self._M)} of {2 ** self.rsn.J}")
            lines.append(f"  LP-c solves = {self.n_lp_c} (pruned enumeration)")
            lines.append(f"  lambda~   = {self.lam_tilde:.6f}" if self.lam_tilde
                         is not None else "  lambda~   = (M empty)")
        lines.append(f"  b_max     = {self._get_b_max():.6f}  (eq. (6); gamma_c={self.gamma_c}, gamma_b={self.gamma_b})")
        if self.vb_cap == "per_pattern":
            caps = [self.b_cap(z) for z in self._M] if self._M else []
            lines.append("  *** vb_cap = per_pattern: OUTSIDE eq. (6) — labeled experiment ***")
            lines.append(f"  b_cap(z)  = |lambda_c(z)|/(4|beta||h|) on M" +
                         (f": {min(caps):.6f} .. {max(caps):.6f}" if caps else "") +
                         f"; rho*/4 = {self.rsn.rho_star / 4:.6f} off M")
        if self.vb_norm is not None:
            lines.append(f"  *** vb_norm = {self.vb_norm}: CONTROL EXPERIMENT outside Definition 2.3 ***")
        lines.append("  v^b table = " + ("all %d patterns precomputed and asserted" % 2 ** self.rsn.J
                                         if self.vb_enumerated else
                                         "filled on demand, each entry asserted on creation"))
        return "\n".join(lines)
