"""Network G (Definition 2.3, c2 = 660): stationarity by halves of T1 + T2.

r = 1000, 2000: from results/decomp_runs.json (T = 250, batches of 25, seeds 1-2).
r = 4000: two runs with T = 40, batches of 4 (7 post-burn batches), seeds 1-2.
Writes results/network_g_check.json."""
import json, sys, time, warnings
from multiprocessing import Pool
import numpy as np
import networks as N
from common import log_run, halves, mean_hw

warnings.filterwarnings("ignore")
G = "G ranking-map fails (BJ20 Ex. 6.6)"
_nw = None


def go(seed):
    global _nw
    if _nw is None:
        _nw = N.build(G)
    t0 = time.time(); o = _nw.run(4000, 40.0, 0.1, 660.0, 0.24, seed, burn=0.3, batch=4.0); o["wall_s"] = time.time() - t0
    o["pred_floor"] = N.floor_drift(_nw, 4000.0, o["U"], 0.5); o["label"] = _nw.tab.label
    return o


if __name__ == "__main__":
    t0 = time.time()
    with Pool(2) as p:
        new = p.map(go, (1, 2), chunksize=1)
    H = json.load(open("results/multi_bench.json"))[G]["HGI_E"]; D = json.load(open("results/decomp_runs.json")); nw = N.build(G); n = nw.net
    rows = []
    for r, runs in ((1000, [o for o in D if o["net"] == G and o["cap"] == "uniform" and o["r"] == 1000]), (2000, [o for o in D if o["net"] == G and o["cap"] == "uniform" and o["r"] == 2000]), (4000, new)):
        for o in sorted(runs, key=lambda o: o["seed"]):
            x = np.array(o["batches"]["hq"]) - np.array(o["batches"]["hg"]); a, b = halves(x); pf = np.array(o["p_flag"])
            fd = (r * (n.K @ pf) / np.abs(n.theta)).tolist()
            rows.append(dict(r=r, seed=o["seed"], T=o["T"], T12=float(x.mean()), T12_pct=float(100 * x.mean() / H), half1=a, half2=b, change_pct=float(100 * (b - a) / x.mean()),
                             E_frac=o["E_frac"], idle_drift=o["idle_drift"], flag_drift_measured=fd, pred_floor=float(N.floor_drift(nw, float(r), o["U"], 0.5))))
            print(f"r = {r} seed {o['seed']} T = {o['T']:.0f}: T1+T2 = {x.mean():8.3f} ({100*x.mean()/H:6.1f}% of HGI_E)  halves {a:8.3f} -> {b:8.3f} ({100*(b-a)/x.mean():+.1f}%)  "
                  f"flag drift measured {np.round(fd, 2).tolist()} vs predicted {rows[-1]['pred_floor']:.2f}  total idle {np.round(o['idle_drift'], 2).tolist()}  E_frac {o['E_frac']:.1e}")
    for o in new:
        if "ygrid" in o: del o["ygrid"]
    json.dump(dict(note="G, Definition 2.3, (c1, c2, kappa) = (0.1, 660, 0.24). flag_drift_measured = r sum_j K_ij P(E_j = 1)/|theta_i| (x_j = 1, an upper bound).",
                   spec=dict(K=N.SPECS[G][0], alpha=N.SPECS[G][1], beta=N.SPECS[G][2], h=N.SPECS[G][3], eta=N.SPECS[G][4]), rows=rows, new_runs_r4000=new),
              open("results/network_g_check.json", "w"))
    dt = time.time() - t0
    log_run("python network_g_check.py", "1, 2 (r = 4000, T = 40)", dt, "sim/results/network_g_check.json, network_g_check.log", f"2 runs in parallel; single run {max(o['wall_s'] for o in new):.0f} s")
    print(f"{dt:.0f}s", flush=True)
