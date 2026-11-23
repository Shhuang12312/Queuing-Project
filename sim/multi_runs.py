"""Convergence sweep across networks. Band depth c2 from networks.rule_c2 (fixed in r);
contrast: c2 = 30 at r = 1000. Writes results/multi_runs.json."""
import json, time, warnings
from multiprocessing import Pool
import networks as N

warnings.filterwarnings("ignore")
BENCH = json.load(open("results/multi_bench.json"))
JOBS = []
for name in N.SPECS:
    c2 = BENCH[name]["c2"]
    plan = ([(4000, 20, 1), (2000, 30, 2), (1000, 40, 2)] if name.startswith("F")
            else [(4000, 30, 2), (2000, 40, 2), (1000, 40, 2)])
    for r, T, ns in plan:
        JOBS += [(name, r, T, c2, s, "rule") for s in range(1, ns + 1)]
    JOBS += [(name, 1000, 40, 30.0, s, "c2=30") for s in (1, 2)]
JOBS.sort(key=lambda j: -(j[1] ** 2 * j[2] * (2 if j[0].startswith("F") else 1)))

_cache = {}


def go(job):
    name, r, T, c2, s, tag = job
    if name not in _cache:
        _cache[name] = N.build(name)
    o = _cache[name].run(r, T, N.C1, c2, N.KAPPA, s)
    o["tag"] = tag
    o["pred_floor"] = N.floor_drift(_cache[name], r, o["U"], N.F_OCC)
    return o


if __name__ == "__main__":
    t0 = time.time(); res = []
    with Pool(2) as p:
        for o in p.imap_unordered(go, JOBS):
            res.append(o); json.dump(res, open("results/multi_runs.json", "w"))
            H = BENCH[o["net"]]["HGI_E"]
            print(f"{o['net']:24s} {o['tag']:6s} r={o['r']:5d} s={o['seed']} JE={o['JE']:7.3f} "
                  f"sameHGI={o['hgi_path']:6.3f} excess={100*(o['JE']-o['hgi_path'])/H:6.1f}% "
                  f"gap={o['gap']:.3f} 1st/2nd={o['JE_1st']:.2f}/{o['JE_2nd']:.2f} "
                  f"floor pred={o['pred_floor']:.3f} Efrac={o['E_frac']:.1e} ({time.time()-t0:.0f}s)", flush=True)
    print("ALL DONE", flush=True)
