"""General-network wrapper: rsn_flat builds the network and the offline
tables; genkern.so simulates; h_hat via exact dual vertices."""
import ctypes, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "rsn_flat"))
from rsn import RSN                       # noqa: E402
from offline import OfflineTables         # noqa: E402
from simulation import hhat_batch_lp, benchmark_hgi_e, dual_vertices   # noqa: E402  (rsn_flat)

_D = ctypes.POINTER(ctypes.c_double)
MAXJ, MAXI = 20, 8          # fixed array sizes in genkern.c (#define MAXJ / MAXI)
_LIB = None


def _lib():
    """genkern.so, loaded on first use from $RSN_GENKERN_SO or sim/genkern.so
    (colab_engine.py compiles it elsewhere and sets the variable)."""
    global _LIB
    if _LIB is None:
        path = os.environ.get("RSN_GENKERN_SO", os.path.join(HERE, "genkern.so"))
        lib = ctypes.CDLL(path)
        lib.simulate.argtypes = ([ctypes.c_int, ctypes.c_int] + [_D] * 7 + [ctypes.c_int, _D]
                                 + [ctypes.c_double] * 5 + [ctypes.c_uint64, _D]
                                 + [ctypes.c_double, ctypes.c_int, _D, ctypes.c_int, _D]
                                 + [ctypes.c_double, ctypes.c_int, _D])
        lib.simulate.restype = ctypes.c_int
        _LIB = lib
    return _LIB


class Net:
    def __init__(self, name, K, alpha, beta, h, eta=None, gamma=0.98, alpha_bar=None, beta_bar=None,
                 tables=None, scv_u=None, scv_v=None):
        """Condition 2 via eta (default) or the paper's (alpha_bar, beta_bar); the kernel
        simulates with alpha^r = alpha + alpha_bar/r and beta^r = beta + beta_bar/r.
        tables: a prebuilt OfflineTables for this network (large J: the pass takes minutes).
        scv_u, scv_v: squared coefficients of variation of the primitives, used only in Sigma
        (the exponential kernel here always has scv 1; see gensim_gi for other families)."""
        self.name = name
        self.net = RSN(K, alpha, beta, h, eta=eta, name=name, alpha_bar=alpha_bar, beta_bar=beta_bar,
                       scv_u=scv_u, scv_v=scv_v)
        n = self.net
        if n.J > MAXJ or n.I > MAXI:
            raise ValueError(f"genkern.c is compiled for J <= {MAXJ}, I <= {MAXI} (got J={n.J}, I={n.I}); "
                             f"raise MAXJ/MAXI in genkern.c and rebuild, or use rsn_flat's pure-Python engine")
        if tables is None:
            tables = OfflineTables(self.net, gamma_c=gamma, gamma_b=gamma)
        elif not tables.vb_enumerated:
            tables.enumerate_vb()                # the kernel needs the full 2^J table
        self.tab = tables
        self.xtab = np.array([self.tab.x(tuple((i >> j) & 1 for j in range(n.J)))
                              for i in range(2 ** n.J)])          # row i: bit j of i = z_j
        self.V = dual_vertices(n)                 # rsn_flat/simulation.py (exact h_hat)
        self.hv = lambda W: (np.atleast_2d(W) @ self.V.T).max(axis=1)
        # h_hat(w) = max_k V_k.w is monotone on R^I_+ iff every vertex is >= 0; then hhat(G) <= hhat(W)
        # follows from G <= W (eq. (12)) and is asserted pathwise in the kernel (toy, B, C: yes; D-G: no).
        self.hhat_monotone = bool(np.all(self.V >= -1e-12))
        W = np.random.default_rng(0).uniform(0, 5, (200, n.I))
        self.vertex_err = float(np.max(np.abs(self.hv(W) - hhat_batch_lp(W, n))))

    def thresholds(self, r, c1, c2, kappa):
        bm = float(self.net.beta.min())
        return c1 * bm * r ** kappa, c2 * bm * r ** kappa

    def run(self, r, T, c1, c2, kappa, seed, burn=0.3, batch=None, ygrid=None):
        """One run of r^2 T physical time, read-outs averaged over [burn T, T] (diffusion units).

        batch: batch length in diffusion time units; if given, the result also carries
          batches = dict(len=batch, n=n_batch, hq=…, hhat=…, hg=…) — per-batch time averages of
          h.Q^, h^(W^) and h^(G), G = Gamma(w^ + X^) (same pass, same path).
        ygrid: grid spacing in diffusion time units; if given, the result carries
          ygrid = dict(dt=…, t0=burn*T, Y=array (n x I)) with Y = w^ + X^ = (W^ - U^) sampled at
          burn*T + k*dt (exactly, inside holding intervals), so np.diff(Y, axis=0) are the increments
          of the free process X^ over dt (the increment test: mean -> theta dt, cov -> Sigma dt).
        The kernel asserts pathwise at every event: h.q >= h^(W); G <= W; and h^(G) <= h^(W) when
        h^ is monotone (`self.hhat_monotone`). A violation raises AssertionError with (t, lhs, rhs).
        W^ uses M^r = diag(1/beta^r) (the paper's definition; = M for eta-splits)."""
        n = self.net
        L, U = self.thresholds(r, c1, c2, kappa)
        arr = lambda a: np.ascontiguousarray(a, dtype=float)
        br, ar = arr(n.betas_r(r)), arr(n.alphas_r(r))
        K, ib = arr(n.K.ravel()), arr(1.0 / br)                       # M^r
        h, C, X, V = arr(n.h), arr(n.C), arr(self.xtab.ravel()), arr(self.V.ravel())
        out = np.zeros(8 + 2 * n.J + n.I)
        T_end = r * r * T
        T_burn = burn * T_end
        if batch is None:
            n_batch, batch_len = 0, 0.0
        else:
            batch_len = float(batch) * r * r
            n_batch = int(np.floor((T_end - T_burn) / batch_len + 1e-9))
            if n_batch < 1:
                raise ValueError("batch longer than the post-burn window")
        bout = np.zeros(max(3 * n_batch, 1)); diag = np.zeros(4)
        if ygrid is None:
            n_grid, grid_dt = 0, 0.0
        else:
            grid_dt = float(ygrid) * r * r
            n_grid = int(np.floor((T_end - T_burn) / grid_dt + 1e-9)) + 1
        yg = np.zeros(max(n_grid * n.I, 1))
        p = lambda a: a.ctypes.data_as(_D)
        code = _lib().simulate(n.I, n.J, p(K), p(ib), p(br), p(ar), p(h), p(C), p(X), len(self.V), p(V),
                               L, U, float(r), T_end, T_burn, int(seed), p(out),
                               batch_len, n_batch, p(bout), int(self.hhat_monotone), p(diag),
                               grid_dt, n_grid, p(yg))
        if code != 0:
            what = {1: "h.q >= h^(W)", 2: "G <= W (eq. (12))", 3: "h^(G) <= h^(W)"}[code]
            raise AssertionError(f"pathwise check '{what}' violated in {self.name} at t = {diag[1] / r / r:.6f} "
                                 f"(diffusion units), r = {r}, seed = {seed}: lhs = {diag[2]!r}, rhs = {diag[3]!r}")
        win = out[7]; J, I = n.J, n.I
        res = dict(net=self.name, r=r, T=T, c=(c1, c2, kappa), seed=seed, L=L, U=U,
                   JE=out[0] / win / r, level=out[1] / win / r, hgi_path=out[2] / win / r,
                   gap=(out[0] - out[1]) / win / r, E_frac=out[3] / win,
                   JE_2nd=out[4] / (0.5 * win) / r, JE_1st=(out[0] - out[4]) / (0.5 * win) / r,
                   p_flag=list(out[8:8 + J] / win), f_below=list(out[8 + J:8 + 2 * J] / win),
                   idle_drift=list(r * out[8 + 2 * J:8 + 2 * J + I] / win / np.abs(n.theta)),
                   events=out[6], checks="h.q>=h^(W); G<=W" + ("; h^(G)<=h^(W)" if self.hhat_monotone else ""))
        if n_batch:
            b = bout.reshape(n_batch, 3) / batch_len / r
            res["batches"] = dict(len=float(batch), n=n_batch, hq=b[:, 0].tolist(), hhat=b[:, 1].tolist(),
                                  hg=b[:, 2].tolist())
        if n_grid:
            res["ygrid"] = dict(dt=float(ygrid), t0=burn * T, Y=(yg.reshape(n_grid, n.I) / r))
        return res

    def benchmark(self, seeds=(0, 1, 2, 3), T=20000.0):
        """HGI_E by the exact reflection map, sqrt(dt)-extrapolated from dt = 4e-3, 1e-3
        (rsn_flat's benchmark_hgi_e; the SE is that of the extrapolated per-seed values)."""
        return benchmark_hgi_e(self.net, hhat_fn=self.hv, dt=(4e-3, 1e-3), T=T, seeds=seeds)
