"""Quick end-to-end check of the workspace (~30 s). Run from sim/ after ./build.sh:

    python selftest.py

1. offline pass on the toy reproduces the paper's objects (Example 2.2)
2. rsn_flat (pure Python) reproduces its own sweep entry bit for bit
3. the general engine (genkern) reproduces the toy engine (toykern3) bit for bit
4. the exact flag-occupancy formula matches simulation
"""
import json, os, sys, warnings
import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
import toysim3                      # noqa: E402  (also puts rsn_flat on sys.path)
import gensim                       # noqa: E402
from flagmodel import g_flag        # noqa: E402
from je import run_JE               # noqa: E402  (rsn_flat)

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


# 1. toy offline objects
tab = toysim3.TAB
check("toy u", np.allclose(tab.u, np.array([-1, -1, 1]) / np.sqrt(3)), str(np.round(tab.u, 4)))
check("toy lambda", np.isclose(tab.lam, -1 / np.sqrt(3)), f"{tab.lam:.6f}")
check("toy M set", tab.M_set == [(0, 0, 0), (0, 0, 1)], str(tab.M_set))
check("toy b_max", abs(tab._get_b_max() - 0.040833) < 1e-5, f"{tab._get_b_max():.6f}")

# 2. flat code reproduces convergence_results.json exactly
store = json.load(open(os.path.join(HERE, "..", "rsn_flat", "convergence_results.json")))
ref = store["faithful|r=50|seed=1"]
out = run_JE(toysim3.NET, r=50.0, T=20.0, c1=0.5, c2=10.0, kappa=0.22, seed=1, n_snapshots=1200,
             hhat_fn=lambda W: np.maximum(W[:, 0], W[:, 1]))
check("rsn_flat reproduces sweep entry", out["JE"] == ref["JE"] and out["E_frac"] == ref["E_frac"],
      f"JE {out['JE']!r} vs {ref['JE']!r}")

# 3. general engine == toy engine, bit for bit
toy = gensim.Net("toy", [[1, 0, 1], [0, 1, 1]], [1, 1, 1], [1, 1, 1], [1, 1, 1], [-0.5] * 3)
g = toy.run(300, 10, 0.1, 60, 0.24, 3)
t = toysim3.run(300, 10, (0.1, 60, 0.24), 3, n_grid=10)
check("genkern == toykern3", all(g[k] == t[k] for k in ("JE", "level", "hgi_path", "gap", "E_frac")),
      f"JE {g['JE']!r}")

# 4. flag formula vs simulation (r = 1000, c2 = 30: flags frequent enough to measure)
o = toysim3.run(1000, 40, (0.1, 30, 0.24), 1)
pred = sum(o["f_below"]) * g_flag(o["L"], o["U"], 1 - 0.5 / 1000, toysim3.XTAB[1, 0])
meas = sum(o["p_flag"])
check("flag formula", abs(pred / meas - 1) < 0.25, f"measured {meas:.2e} vs predicted {pred:.2e}")

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
