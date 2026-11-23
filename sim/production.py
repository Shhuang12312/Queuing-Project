"""Toy network at fixed constants over a range of r (Table 1 and the discounted cost). Writes results/production.json incrementally."""
import json, os, time, warnings
from multiprocessing import Pool
import toysim3 as toysim

warnings.filterwarnings("ignore")
OUT = "results/production.json"

ERG = ([("tuned", r, 160, s) for r in (100, 200, 400, 1000) for s in range(1, 5)]
       + [("tuned", 2000, 80, s) for s in range(1, 5)]
       + [("tuned", 4000, 40, s) for s in range(1, 5)]
       + [("reference", r, 80, s) for r in (100, 400, 1000) for s in range(1, 3)])
DISC = ([("tuned", 100, s) for s in range(1000, 1200)] + [("tuned", 300, s) for s in range(2000, 2100)]
        + [("tuned", 1000, s) for s in range(3000, 3048)] + [("tuned", 3000, s) for s in range(4000, 4012)])


def erg(job):
    c, r, T, s = job
    o = toysim.run(r, T, c, s, burn=0.3, n_grid=400)
    o["kind"] = "ergodic"
    return o


def disc(job):
    c, r, s = job
    o = toysim.run(r, 10.0, c, s, burn=0.0, n_grid=20)
    return dict(kind="discounted", consts=c, r=r, seed=s, JD=o["JD"], JD_level=o["JD_level"])


if __name__ == "__main__":
    t0 = time.time()
    res = []
    with Pool(2) as p:
        for o in p.imap_unordered(disc, DISC):
            res.append(o)
        json.dump(res, open(OUT, "w"))
        print(f"discounted done {time.time()-t0:.0f}s", flush=True)
        for o in p.imap_unordered(erg, sorted(ERG, key=lambda j: j[1] ** 2 * j[2])):
            res.append(o)
            json.dump(res, open(OUT, "w"))
            print(f"{o['consts']:9s} r={o['r']:5d} T={o['T']:4.0f} s={o['seed']} JE={o['JE']:7.3f} "
                  f"level={o['level']:7.3f} 1st/2nd={o['JE_1st']:.2f}/{o['JE_2nd']:.2f} "
                  f"Efrac={o['E_frac']:.5f} idle={o['idle_drift'][0]:.3f},{o['idle_drift'][1]:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print("ALL DONE", flush=True)
