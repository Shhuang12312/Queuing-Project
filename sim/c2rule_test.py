"""The model-optimal band c2* (f_meas variant of c2rule_fit.py) on two networks,
D (c2* = 31 vs rule 41) and E (c2* = 57 vs rule 62), at r = 1000 and 2000, 2 seeds each.
Writes results/c2rule_test.json (same fields as multi_runs.json, tag "c2*")."""
import json, time, warnings
from multiprocessing import Pool
import networks as N

warnings.filterwarnings("ignore")
PLAN = {"D unequal rates": 31.0, "E unequal loads, costs": 57.0}
JOBS = [(name, r, T, c2, s) for name, c2 in PLAN.items() for r, T in ((2000, 40), (1000, 40)) for s in (1, 2)]
_cache = {}


def go(job):
    name, r, T, c2, s = job
    if name not in _cache:
        _cache[name] = N.build(name)
    o = _cache[name].run(r, T, N.C1, c2, N.KAPPA, s)
    o["tag"] = "c2*"
    o["pred_floor"] = N.floor_drift(_cache[name], r, o["U"], N.F_OCC)
    return o


if __name__ == "__main__":
    t0 = time.time(); res = []
    B = json.load(open("results/multi_bench.json"))
    with Pool(2) as p:
        for o in p.imap_unordered(go, JOBS):
            res.append(o); json.dump(res, open("results/c2rule_test.json", "w"), indent=1)
            H = B[o["net"]]["HGI_E"]
            print(f"{o['net'][:24]:24s} c2={o['c'][1]:5.1f} r={o['r']:5d} s={o['seed']} JE={o['JE']:7.3f} "
                  f"sameHGI={o['hgi_path']:6.3f} excess={100*(o['JE']-o['hgi_path'])/H:6.1f}% gap={o['gap']:.3f} "
                  f"1st/2nd={o['JE_1st']:.2f}/{o['JE_2nd']:.2f} Efrac={o['E_frac']:.1e} ({time.time()-t0:.0f}s)", flush=True)
    print("ALL DONE", flush=True)
