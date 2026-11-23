"""Network G with the per-pattern tables (OUTSIDE eq. (6)) at the same band c2 = 660 the
eq. (6) runs used (run_network_g.py), and at the per-pattern error-model minimizer c2 = 545, r = 1000 and 2000,
2 seeds — isolates the push effect from the band effect (the 5%-rule band c2 = 2262 pins every queue at U/r).
Appends to results/vbcap_runs.json with tags "per_pattern_c2=660" / "per_pattern_c2=545"."""
import json, time, warnings
from multiprocessing import Pool
import networks as N
from vbcap_runs import go as _go, B

warnings.filterwarnings("ignore")
NAME = "G ranking-map fails (BJ20 Ex. 6.6)"


def go(job):
    o = _go(job); o["tag"] = f"per_pattern_c2={job[3]:.0f}"; return o


if __name__ == "__main__":
    t0 = time.time()
    jobs = [(NAME, r, 40, c2, s) for r in (2000, 1000) for c2 in (660.0, 545.0) for s in (1, 2)]
    res = json.load(open("results/vbcap_runs.json"))
    res = [o for o in res if not o.get("tag", "").startswith("per_pattern_c2=")]
    H = B[NAME]["HGI_E"]
    with Pool(4) as p:
        for o in p.imap_unordered(go, jobs):
            res.append(o); json.dump(res, open("results/vbcap_runs.json", "w"))
            print(f"G per-pattern c2={o['c'][1]:.0f} r={o['r']:5d} s={o['seed']} JE={o['JE']:7.3f} sameHGI={o['hgi_path']:6.3f} "
                  f"excess={100*(o['JE']-o['hgi_path'])/H:6.1f}% 1st/2nd={o['JE_1st']:.2f}/{o['JE_2nd']:.2f} "
                  f"floor pred={o['pred_floor']:.3f} Efrac={o['E_frac']:.1e} ({time.time()-t0:.0f}s)", flush=True)
    print("ALL DONE", flush=True)
