"""Convergence check on the toy with the same-path HGI (toykern3).
deep = (0.1, 60, 0.24) and tuned = (0.1, 30, 0.24): admissible constants, fixed across r. Writes results/final_runs.json."""
import json, time, warnings
from multiprocessing import Pool
import toysim3 as ts

warnings.filterwarnings("ignore")
DEEP, TUNED = (0.1, 60.0, 0.24), (0.1, 30.0, 0.24)
JOBS = ([(DEEP, 10000, 20, s) for s in (1, 2)]
        + [(DEEP, 4000, 30, s) for s in (1, 2, 3)] + [(TUNED, 4000, 30, s) for s in (1, 2)]
        + [(DEEP, 2000, 40, s) for s in range(1, 5)]
        + [(c, r, 40, s) for c in (DEEP, TUNED) for r in (300, 1000) for s in range(1, 5)])


def go(job):
    c, r, T, s = job
    o = ts.run(r, T, c, s, burn=0.3, n_grid=200)
    o["c2"] = c[1]
    return o


if __name__ == "__main__":
    t0 = time.time(); res = []
    with Pool(2) as p:
        for o in p.imap_unordered(go, JOBS):
            res.append(o); json.dump(res, open("results/final_runs.json", "w"))
            print(f"c2={o['c2']:3.0f} r={o['r']:5d} T={o['T']:3.0f} s={o['seed']} JE={o['JE']:7.3f} "
                  f"level={o['level']:7.3f} hgi_path={o['hgi_path']:.3f} gap={o['gap']:.4f} "
                  f"1st/2nd={o['JE_1st']:.2f}/{o['JE_2nd']:.2f} ({time.time()-t0:.0f}s)", flush=True)
    print("ALL DONE", flush=True)
