"""Python wrapper for toykern3.c; policy objects come from rsn_flat."""
import ctypes, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FLAT = os.path.join(HERE, "..", "rsn_flat")
sys.path.insert(0, FLAT)
from rsn import make_2lln            # noqa: E402
from offline import OfflineTables    # noqa: E402

_lib = ctypes.CDLL(os.path.join(HERE, "toykern3.so"))
_D = ctypes.POINTER(ctypes.c_double)
_lib.simulate.argtypes = [_D, ctypes.c_double, ctypes.c_double, _D, _D, ctypes.c_double,
                          ctypes.c_double, ctypes.c_double, ctypes.c_uint64, ctypes.c_int, _D, _D]
NBIN = 6
BIN_LABELS = ["<0.05", "0.05-0.25", "0.25-0.5", "0.5-1", "1-2", ">2"]

NET = make_2lln()
TAB = OfflineTables(NET, gamma_c=0.98, gamma_b=0.98)       # run_JE defaults
XTAB = np.array([TAB.x(((i >> 0) & 1, (i >> 1) & 1, (i >> 2) & 1)) for i in range(8)])

CONSTS = {"tuned": (0.1, 30.0, 0.24), "reference": (0.5, 10.0, 0.22)}


def run(r, T, consts="tuned", seed=1, burn=0.3, n_grid=400, LU=None):
    c1, c2, kap = CONSTS[consts] if isinstance(consts, str) else consts
    rk = float(r) ** kap
    L, U = (c1 * NET.beta.min() * rk, c2 * NET.beta.min() * rk) if LU is None else LU
    alpha_r = np.ascontiguousarray(NET.alphas_r(r), float)
    out = np.zeros(13 + 2 * NBIN + 6); grid = np.zeros(n_grid)
    x = np.ascontiguousarray(XTAB.ravel(), float)
    T_end = r * r * T
    _lib.simulate(x.ctypes.data_as(_D), L, U, alpha_r.ctypes.data_as(_D),
                  np.ascontiguousarray(NET.h).ctypes.data_as(_D), float(r), T_end,
                  burn * T_end, int(seed), n_grid, grid.ctypes.data_as(_D), out.ctypes.data_as(_D))
    win = out[12]
    half = 0.5 * win
    res = dict(r=r, T=T, consts=consts, seed=seed,
               JE=out[0] / win / r, level=out[1] / win / r,
               gap=(out[0] - out[1]) / win / r, E_frac=out[2] / win,
               JE_2nd=out[7] / half / r, level_2nd=out[8] / half / r,
               JE_1st=(out[0] - out[7]) / half / r,
               idle_drift=[r * out[3] / win, r * out[4] / win],          # = |theta| = 1 if stationary
               idle_height=[out[5] / out[3] / r, out[6] / out[4] / r],     # mean W^ where idling happens
               idle_bins=[list(r * out[13:13 + NBIN] / win), list(r * out[13 + NBIN:13 + 2 * NBIN] / win)],
               JD=out[9], JD_level=out[10], events=out[11], grid=list(grid),
               hgi_path=out[29] / win / r, f_below=list(out[25:27] / win), p_flag=list(out[27:29] / win), L=L, U=U)
    return res
