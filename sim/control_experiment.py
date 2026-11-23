"""The CONTROL EXPERIMENT of the report (OUTSIDE Definition 2.3: hysteresis flags off).

Toy network. Flags are disabled by c1 = -1 (L < 0: no queue is ever below L); the near-empty pattern Z still uses
U = c2 r^kappa. Three variants of the push: |v^b| = 0.25 (= rho*/4 exactly), 0.24 (the value in rsn_flat/check.py), and
the definition's own gamma_b*b_max = 0.040 ("flags off at b_max"). Constant sets (c2, kappa) = (10, 0.22), (30, 0.24),
(60, 0.24). r = 30, 100 (T = 1600), 300 (T = 320), 1000 (T = 160); 4 seeds (1-4); 30% burn-in; 7 post-burn batches.
Stores per run: J_E, level h^(W), same-path benchmark h^(G), flag fraction and the batch series of h.Q, h^(W), h^(G).
Writes results/control_experiment.json and prints the summary table."""
import json, sys, time, warnings
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, "../rsn_flat")
import gensim
from rsn import RSN
from offline import OfflineTables
from common import log_run, halves, se, HGI_E_TOY

warnings.filterwarnings("ignore")
K2 = [[1, 0, 1], [0, 1, 1]]
VBS = (0.25, 0.24, None)
SETS = ((10.0, 0.22), (30.0, 0.24), (60.0, 0.24))
RT = ((30, 1600.0), (100, 1600.0), (300, 320.0), (1000, 160.0))
SEEDS = (1, 2, 3, 4)
PAPER = {30: 3.90, 100: 3.33, 300: 3.27, 1000: 3.14}
_c = {}


def net(vb):
    if vb not in _c:
        tab = None if vb is None else OfflineTables(RSN(K2, [1, 1, 1], [1, 1, 1], [1, 1, 1], eta=[-0.5] * 3), gamma_c=0.98, gamma_b=0.98, vb_norm=vb)
        _c[vb] = gensim.Net("toy", K2, [1, 1, 1], [1, 1, 1], [1, 1, 1], eta=[-0.5] * 3, tables=tab)
    return _c[vb]


def go(job):
    vb, c2, kap, r, T, seed = job
    nw = net(vb)
    o = nw.run(r, T, -1.0, c2, kap, seed, burn=0.3, batch=T / 10.0)
    b = o["batches"]
    return dict(vb=vb, vb_norm=float(np.linalg.norm(nw.tab.vb((1, 1, 1)))), label=nw.tab.label + "; flags off (c1 = -1)", c2=c2, kappa=kap, r=r, T=T, seed=seed,
                JE=o["JE"], level=o["level"], hgi_path=o["hgi_path"], E_frac=o["E_frac"], JE_1st=o["JE_1st"], JE_2nd=o["JE_2nd"],
                hq=b["hq"], hhat=b["hhat"], hg=b["hg"], events=o["events"])


def summarize(res):
    rows = []
    for vb in VBS:
        for c2, kap in SETS:
            line = []
            for r, T in RT:
                g = [o for o in res if o["vb"] == vb and o["c2"] == c2 and o["r"] == r]
                je = np.array([o["JE"] for o in g]); t12 = np.array([o["JE"] - o["hgi_path"] for o in g]); hg = np.array([o["hgi_path"] for o in g])
                hv = [halves(np.array(o["hq"]) - np.array(o["hg"])) for o in g]; h1, h2 = np.mean([a for a, _ in hv]), np.mean([b for _, b in hv])
                jh = [halves(o["hq"]) for o in g]
                row = dict(vb=vb, c2=c2, kappa=kap, r=r, T=T, n=len(g), JE=float(je.mean()), JE_se=se(je), T12=float(t12.mean()), T12_se=se(t12),
                           T12_half1=float(h1), T12_half2=float(h2), JE_half1=float(np.mean([a for a, _ in jh])), JE_half2=float(np.mean([b for _, b in jh])),
                           hgi_path=float(hg.mean()), hgi_path_se=se(hg), z_vs_paper=float((je.mean() - PAPER[r]) / se(je)))
                rows.append(row); line.append(row)
            name = "b_max (0.040)" if vb is None else f"{vb}"
            print(f"|v^b| = {name:13s} (c2, kappa) = ({c2:.0f}, {kap})")
            for x in line:
                print(f"   r={x['r']:5d} T={x['T']:5.0f}: J_E = {x['JE']:8.3f} ± {x['JE_se']:6.3f} (paper {PAPER[x['r']]:.2f}, z = {x['z_vs_paper']:+6.1f})  T1+T2 = {x['T12']:8.3f} ± {x['T12_se']:6.3f}"
                      f"  halves of T1+T2 {x['T12_half1']:8.3f} -> {x['T12_half2']:8.3f}  halves of J_E {x['JE_half1']:8.2f} -> {x['JE_half2']:8.2f}  h^(G) = {x['hgi_path']:.3f} ± {x['hgi_path_se']:.3f}")
    return rows


if __name__ == "__main__":
    t0 = time.time()
    jobs = [(vb, c2, kap, r, T, s) for vb in VBS for c2, kap in SETS for r, T in RT for s in SEEDS]
    jobs.sort(key=lambda j: -(j[3] ** 2) * j[4])
    with Pool(4) as p:
        res = p.map(go, jobs, chunksize=1)
    rows = summarize(res)
    json.dump(dict(note="CONTROL EXPERIMENT, outside Definition 2.3: hysteresis flags off (c1 = -1). vb = null means the definition's own 0.98*b_max.",
                   HGI_E=HGI_E_TOY, paper_values=PAPER, summary=rows, runs=res), open("results/control_experiment.json", "w"))
    dt = time.time() - t0
    log_run("python control_experiment.py", "1-4 per cell (3 pushes x 3 constant sets x 4 r)", dt, "sim/results/control_experiment.json, control_experiment.log",
            f"{len(jobs)} runs on 4 cores; longest single run about {max(o['events'] for o in res)/1.2e7:.0f} s")
    print(f"{dt:.0f}s", flush=True)
