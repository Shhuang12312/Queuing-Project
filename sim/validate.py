"""Statistical cross-check: compiled toy kernel vs the pure-Python rsn_flat code (r=50, T=20). Writes results/validate.json."""
import json, sys
from multiprocessing import Pool
import numpy as np
import toysim3 as toysim

sys.path.insert(0, toysim.FLAT)
from je import run_JE            # noqa: E402

HH = lambda W: np.maximum(W[:, 0], W[:, 1])


def flat_run(args):
    consts, seed = args
    c1, c2, k = toysim.CONSTS[consts]
    o = run_JE(toysim.NET, r=50.0, T=20.0, c1=c1, c2=c2, kappa=k, seed=seed,
               n_snapshots=1200, hhat_fn=HH)
    return consts, o["JE"], o["ssc_gap"], o["E_frac"]


if __name__ == "__main__":
    jobs = [(c, s) for c in ("reference", "tuned") for s in range(10, 18)]
    with Pool(2) as p:
        flat = p.map(flat_run, jobs)
    rows = {}
    for c in ("reference", "tuned"):
        f = np.array([x[1:] for x in flat if x[0] == c])
        k = np.array([[o["JE"], o["gap"], o["E_frac"]] for o in
                      (toysim.run(50, 20, c, s) for s in range(100, 164))])
        for name, a in (("flat (8 seeds)", f), ("compiled (64 seeds)", k)):
            m, se = a.mean(0), a.std(0, ddof=1) / np.sqrt(len(a))
            print(f"{c:9s} {name:20s} J_E={m[0]:6.2f}±{se[0]:.2f}  gap={m[1]:.4f}±{se[1]:.4f}"
                  f"  E_frac={m[2]:.4f}±{se[2]:.4f}")
        fm, fs = f.mean(0), f.std(0, ddof=1) / np.sqrt(len(f))
        km, ks = k.mean(0), k.std(0, ddof=1) / np.sqrt(len(k))
        z = (fm - km) / np.sqrt(fs ** 2 + ks ** 2)
        print(f"{'':9s} z-scores (J_E, gap, E_frac): {np.round(z, 2)}")
        rows[c] = dict(flat=f.tolist(), compiled_mean=km.tolist(), z=z.tolist())
    json.dump(rows, open("results/validate.json", "w"))
