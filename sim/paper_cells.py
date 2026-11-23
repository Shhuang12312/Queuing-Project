"""Cells for the report tables, run with the one-pass kernel (M^r, batch means, pathwise checks):
  Table 2   — toy, Definition 2.3 verbatim, (0.1, 60, 0.24), r = 300 / 1000 / 2000 / 4000 (T = 250, batches of 25)
              and r = 10^4 (T = 50, batches of 5), 2 seeds each: J_E, T1, T2, T3 with batch-means 95% half-widths;
  network G at r = 4000 under the per-pattern cap (LABELED, outside eq. (6)) at c2 = 30 and 660, T = 40,
              batches of 4, 2 seeds.
Appends to results/paper_cells.json (records carry the tables' label); prints the per-cell table."""
import json, os, subprocess, sys, time
from multiprocessing import Pool
import numpy as np
from scipy.stats import t as tdist
import networks as N

HGI = {"A toy (Ex. 2.2)": 2.8093965016743754, "G ranking-map fails (BJ20 Ex. 6.6)": 24.790698207335083}
TOY, G = "A toy (Ex. 2.2)", "G ranking-map fails (BJ20 Ex. 6.6)"
CELLS = [(TOY, "uniform", 60.0, r, 250.0, 25.0) for r in (300, 1000, 2000, 4000)] + [(TOY, "uniform", 60.0, 10000, 50.0, 5.0)]
CELLS += [(G, "per_pattern", c2, 4000, 40.0, 4.0) for c2 in (30.0, 660.0)]
SEEDS = (1, 2)
_cache = {}


def go(job):
    name, cap, c2, r, T, batch, seed = job
    if (name, cap) not in _cache:
        _cache[(name, cap)] = N.build(name, vb_cap=cap)
    nw = _cache[(name, cap)]
    o = nw.run(r, T, N.C1, c2, N.KAPPA, seed, burn=0.3, batch=batch)
    o["cap"] = cap; o["label"] = nw.tab.label
    return o


def hw(x):
    x = np.asarray(x, float)
    return tdist.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def table(res):
    rows = []
    print(f"{'net':4s} {'cap':11s} {'c2':>5s} {'r':>6s} {'T':>5s} {'nb':>3s} | {'J_E':>14s} | {'T1':>16s} | {'T2':>16s} | {'T1+T2 (%HGI)':>14s} | {'T3':>16s} | Efrac")
    for name, cap, c2, r, T, batch in CELLS:
        runs = [o for o in res if o["net"] == name and o["cap"] == cap and o["c"][1] == c2 and o["r"] == r and o["T"] == T]
        if not runs:
            continue
        H = HGI[name]
        hq = np.concatenate([o["batches"]["hq"] for o in runs]); hh = np.concatenate([o["batches"]["hhat"] for o in runs])
        hg = np.concatenate([o["batches"]["hg"] for o in runs])
        t1, t2, t3 = hq - hh, hh - hg, hg - H
        row = dict(net=name, cap=cap, label=runs[0]["label"], c2=c2, r=r, T=T, batch=batch, n_batches=len(hq), seeds=[o["seed"] for o in runs], HGI_E=H,
                   JE=float(hq.mean()), JE_hw=float(hw(hq)), T1=float(t1.mean()), T1_hw=float(hw(t1)), T2=float(t2.mean()), T2_hw=float(hw(t2)),
                   T12=float((t1 + t2).mean()), T12_hw=float(hw(t1 + t2)), T3=float(t3.mean()), T3_hw=float(hw(t3)),
                   E_frac=float(np.mean([o["E_frac"] for o in runs])), T1_over_c2r=float(t1.mean() / (c2 * r ** (N.KAPPA - 1))))
        rows.append(row)
        print(f"{name[:4]:4s} {cap:11s} {c2:5.0f} {r:6d} {T:5.0f} {len(hq):3d} | {row['JE']:6.3f} ± {row['JE_hw']:5.3f} | {row['T1']:7.4f} ± {row['T1_hw']:6.4f} | "
              f"{row['T2']:7.4f} ± {row['T2_hw']:6.4f} | {100*row['T12']/H:5.1f} ± {100*row['T12_hw']/H:4.1f} | {row['T3']:+7.3f} ± {row['T3_hw']:6.3f} | {row['E_frac']:.1e}")
    return rows


if __name__ == "__main__":
    t0 = time.time()
    res = json.load(open("results/paper_cells.json")) if os.path.exists("results/paper_cells.json") else []
    done = {(o["net"], o["cap"], o["c"][1], o["r"], o["T"], o["seed"]) for o in res}
    jobs = [c + (s,) for c in CELLS for s in SEEDS if (c[0], c[1], c[2], c[3], c[4], s) not in done]
    jobs.sort(key=lambda j: -(j[3] ** 2 * j[4] * (2 if j[0][0] == "G" else 1)))
    with Pool(4) as p:
        for o in p.imap_unordered(go, jobs):
            res.append(o); json.dump(res, open("results/paper_cells.json", "w"))
            print(f"{o['net'][:4]} {o['cap']:11s} c2={o['c'][1]:4.0f} r={o['r']:5d} s={o['seed']} JE={o['JE']:7.3f} h^(W)={o['level']:7.3f} h^(G)={o['hgi_path']:7.3f} "
                  f"Efrac={o['E_frac']:.1e} ({time.time()-t0:.0f}s)", flush=True)
    rows = table(res)
    rows_out = dict(commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd="..").stdout.strip(), rows=rows)
    json.dump(rows_out, open("results/paper_cells_table.json", "w"), indent=1)
    print("ALL DONE", flush=True)
