"""colab_engine.py — the compiled general engine, built at import (Google Colab or any machine
with a C compiler), with the pure-Python loop of rsn_flat as the fallback.

    import sys; sys.path.insert(0, "/content/rsn_hgi_workspace/sim")   # after cloning / unzipping
    import colab_engine as ce                                           # compiles genkern.c (~2 s)
    nw  = ce.Net("toy", [[1, 0, 1], [0, 1, 1]], [1, 1, 1], [1, 1, 1], [1, 1, 1], [-0.5] * 3)
    # non-exponential primitives (Erlang-k / H2): from gensim_gi import GINet  (genkern_gi.c is
    # compiled too and exported as $RSN_GENKERN_GI_SO when the C engine is available)
    out = nw.run(1000, 40, 0.1, 60, 0.24, seed=1)      # r, T (diffusion units), c1, c2, kappa
    print(out["JE"], out["hgi_path"], out["E_frac"])
    print(ce.ENGINE)                                   # "c" or "python"

What happens at import:
  1. rsn_flat/ (next to sim/) is put on sys.path if `rsn` is not importable already.
  2. genkern.c (next to this file) is compiled with $CC, cc, gcc or clang into
     `sim/genkern.so` when that directory is writable, else into a temp directory;
     a .so newer than the .c is reused. The path is exported as $RSN_GENKERN_SO so
     that `gensim` finds it, and `Net` is gensim.Net (1.0-1.7e7 events/s by network, results/engine_speed.json).
  3. If no compiler works (or RSN_FORCE_PYTHON=1 is set), `Net` is `PyNet`: the same
     constructor and `run(r, T, c1, c2, kappa, seed)` signature on top of
     rsn_flat.simulation.gillespie_ergodic (~5e4 events/s: r <= 100 is practical).
     `hgi_path` (the same-path HGI of the C engine) is not available there and is
     returned as NaN; `level` (mean h_hat of the workload) is.

Both engines simulate Definition 2.3 verbatim with rsn_flat's OfflineTables.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FLAT = os.path.join(os.path.dirname(HERE), "rsn_flat")
C_SRC = os.path.join(HERE, "genkern.c")
C_SRC_GI = os.path.join(HERE, "genkern_gi.c")        # non-exponential primitives (gensim_gi)

for _p in (FLAT, HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from rsn import RSN                                     # noqa: E402  (rsn_flat)
from offline import OfflineTables                       # noqa: E402
from policy import GeneralThresholdPolicy               # noqa: E402
from simulation import gillespie_ergodic, make_hhat, benchmark_hgi_e   # noqa: E402

ENGINE = None          # "c" or "python" after import
SO_PATH = None
COMPILE_LOG = ""


def _candidate_compilers():
    seen = []
    for name in ([os.environ["CC"]] if os.environ.get("CC") else []) + ["cc", "gcc", "clang"]:
        path = shutil.which(name)
        if path and path not in seen:
            seen.append(path)
    return seen


def compile_kernel(out_dir=None, verbose=False, src=None):
    """Compile genkern.c (or `src`) into out_dir/<name>.so (reused if newer than the source).
    Returns the .so path or None; the compiler output is kept in COMPILE_LOG."""
    global COMPILE_LOG
    src = C_SRC if src is None else src
    so_name = os.path.splitext(os.path.basename(src))[0] + ".so"
    if not os.path.exists(src):
        COMPILE_LOG += f"{src} not found\n"
        return None
    dirs = [out_dir] if out_dir else [HERE, os.path.join(tempfile.gettempdir(), "rsn_hgi_build")]
    for d in dirs:
        try:
            os.makedirs(d, exist_ok=True)
            so = os.path.join(d, so_name)
            if os.path.exists(so) and os.path.getmtime(so) >= os.path.getmtime(src):
                COMPILE_LOG += f"reusing {so}\n"
                return so
            if not os.access(d, os.W_OK):
                continue
            for cc in _candidate_compilers():
                for flags in (["-O3", "-shared", "-fPIC", "-march=native"], ["-O3", "-shared", "-fPIC"]):
                    cmd = [cc] + flags + ["-o", so, src, "-lm"]
                    res = subprocess.run(cmd, capture_output=True, text=True)
                    COMPILE_LOG += f"$ {' '.join(cmd)}\n{res.stdout}{res.stderr}\n"
                    if verbose:
                        print(COMPILE_LOG)
                    if res.returncode == 0 and os.path.exists(so):
                        return so
        except OSError as e:                   # unwritable / odd filesystem: try the next dir
            COMPILE_LOG += f"{d}: {e}\n"
    return None


class PyNet:
    """Pure-Python fallback with gensim.Net's interface (rsn_flat's exact Gillespie loop)."""

    def __init__(self, name, K, alpha, beta, h, eta=None, gamma=0.98, alpha_bar=None, beta_bar=None):
        self.name = name
        self.net = RSN(K, alpha, beta, h, eta=eta, name=name, alpha_bar=alpha_bar, beta_bar=beta_bar)
        self.tab = OfflineTables(self.net, gamma_c=gamma, gamma_b=gamma)
        self.hv = make_hhat(self.net)
        self.V = getattr(self.hv, "vertices", None)

    def thresholds(self, r, c1, c2, kappa):
        bm = float(self.net.beta.min())
        return c1 * bm * r ** kappa, c2 * bm * r ** kappa

    def run(self, r, T, c1, c2, kappa, seed, burn=0.3, n_snapshots=2000):
        n = self.net
        L, U = self.thresholds(r, c1, c2, kappa)
        pol = GeneralThresholdPolicy(n, c1=c1, c2=c2, kappa=kappa, tables=self.tab)
        out = gillespie_ergodic(n, pol, float(r), T_diff=float(T), burn_frac=burn,
                                n_snapshots=n_snapshots, rng=np.random.default_rng(int(seed)))
        snaps = out["snapshots"]
        je_snap = (snaps @ n.h) / float(r)
        half = len(je_snap) // 2
        return dict(net=self.name, r=r, T=T, c=(c1, c2, kappa), seed=seed, L=L, U=U,
                    JE=out["JE"], level=out["HGI_path"], hgi_path=float("nan"),
                    gap=out["ssc_gap"], E_frac=out["E_frac"],
                    JE_1st=float(je_snap[:half].mean()), JE_2nd=float(je_snap[half:].mean()),
                    events=float("nan"), engine="python")

    def benchmark(self, seeds=(0, 1, 2, 3), T=20000.0):
        return benchmark_hgi_e(self.net, hhat_fn=self.hv, dt=(4e-3, 1e-3), T=T, seeds=seeds)


def _setup():
    global ENGINE, SO_PATH, Net
    if os.environ.get("RSN_FORCE_PYTHON", "") not in ("", "0"):
        ENGINE, Net = "python", PyNet
        return
    so = compile_kernel()
    if so is not None:
        os.environ["RSN_GENKERN_SO"] = so
        so_gi = compile_kernel(src=C_SRC_GI)                 # optional: GI primitives engine
        if so_gi is not None:
            os.environ["RSN_GENKERN_GI_SO"] = so_gi
        try:
            import gensim                                    # loads the .so lazily on first run
            gensim._lib()                                    # fail here, not inside run()
            Net, ENGINE, SO_PATH = gensim.Net, "c", so
            return
        except (OSError, ImportError) as e:                  # unloadable .so: fall back
            globals()["COMPILE_LOG"] += f"load failed: {e}\n"
    warnings.warn("colab_engine: could not compile/load genkern.c — using the pure-Python engine "
                  "(~1000x slower; keep r <= 100). See colab_engine.COMPILE_LOG.", RuntimeWarning)
    ENGINE, Net = "python", PyNet


Net = PyNet
_setup()


def demo(r=1000, T=40, c=(0.1, 60.0, 0.24), seed=1):
    """One toy run (Example 2.2) and the comparison with HGI_E = 2.81. r = 1000, T = 40
    takes ~10 s with the C engine (~3e8 events); with the Python engine use r = 50."""
    import time
    if ENGINE == "python" and r > 150:
        print(f"python engine: r={r} would take hours; running r=50, T=10 instead")
        r, T = 50, 10
    nw = Net("toy", [[1, 0, 1], [0, 1, 1]], [1, 1, 1], [1, 1, 1], [1, 1, 1], [-0.5] * 3)
    t0 = time.time()
    o = nw.run(r, T, *c, seed)
    dt = time.time() - t0
    hgi = 2.81
    print(f"engine={ENGINE}  r={r} T={T} c={c} seed={seed}  ({dt:.1f}s"
          + (f", {o['events']:.2e} events)" if np.isfinite(o.get("events", np.nan)) else ")"))
    print(f"J_E = {o['JE']:.3f}   level h_hat(W) = {o['level']:.3f}   same-path HGI = {o['hgi_path']:.3f}"
          f"   E_frac = {o['E_frac']:.2e}")
    print(f"halves {o['JE_1st']:.3f} -> {o['JE_2nd']:.3f} (growth = transient)")
    if np.isfinite(o["hgi_path"]):
        print(f"excess over same-path HGI = {o['JE'] - o['hgi_path']:.3f} "
              f"= {100 * (o['JE'] - o['hgi_path']) / hgi:.1f}% of HGI_E (expect ~31% at r = 1000, c2 = 60)")
    print(f"HGI_E (RBM benchmark) = {hgi}")
    return o


if __name__ == "__main__":
    demo()
