"""Band scan for network G with the per-pattern tables (OUTSIDE eq. (6)) at r = 1000,
seed 1, c2 in {100, 200, 300, 400} (the fitted error model is miscalibrated on G: it predicts 37-54% flag drift at
c2 = 545-660 while the measured flag fraction is 0). Then r = 2000 x 2 seeds at the best c2 of the scan.
Appends to results/vbcap_runs.json with tag "per_pattern_scan"."""
import json, sys, time, warnings
from multiprocessing import Pool
import numpy as np
import networks as N
from vbcap_runs import go as _go, B

warnings.filterwarnings("ignore")
NAME = "G ranking-map fails (BJ20 Ex. 6.6)"


def go(job):
    o = _go(job); o["tag"] = "per_pattern_scan"; return o


def report(o, t0):
    H = B[NAME]["HGI_E"]
    print(f"G per-pattern c2={o['c'][1]:.0f} r={o['r']:5d} s={o['seed']} JE={o['JE']:7.3f} sameHGI={o['hgi_path']:6.3f} "
          f"excess={100*(o['JE']-o['hgi_path'])/H:6.1f}% 1st/2nd={o['JE_1st']:.2f}/{o['JE_2nd']:.2f} "
          f"floor pred={o['pred_floor']:.3f} Efrac={o['E_frac']:.1e} idle={np.round(o['idle_drift'],2).tolist()} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    res = json.load(open("results/vbcap_runs.json"))
    if "--fresh" in sys.argv:
        res = [o for o in res if o.get("tag") != "per_pattern_scan"]
    sys.argv = [a for a in sys.argv if a != "--fresh"]
    H = B[NAME]["HGI_E"]
    c2s = [float(c) for c in sys.argv[1:]] or [100.0, 200.0, 300.0, 400.0]
    with Pool(4) as p:
        for o in p.imap_unordered(go, [(NAME, 1000, 40, c2, 1) for c2 in c2s]):
            res.append(o); json.dump(res, open("results/vbcap_runs.json", "w")); report(o, t0)
    scan = [o for o in res if o["tag"] == "per_pattern_scan" and o["r"] == 1000]
    best = min(scan, key=lambda o: o["JE"] - o["hgi_path"])["c"][1]
    print(f"best c2 at r = 1000: {best:.0f}", flush=True)
    todo = [(NAME, 2000, 40, best, s) for s in (1, 2)] + [(NAME, 1000, 40, best, 2)]
    todo = [j for j in todo if not any(o["tag"] == "per_pattern_scan" and o["r"] == j[1] and o["c"][1] == j[3]
                                      and o["seed"] == j[4] for o in res)]
    with Pool(4) as p:
        for o in p.imap_unordered(go, todo):
            res.append(o); json.dump(res, open("results/vbcap_runs.json", "w")); report(o, t0)
    print("ALL DONE", flush=True)
