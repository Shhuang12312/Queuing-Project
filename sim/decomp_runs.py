"""The three-term decomposition of the excess cost, per (network, cap, r), in one pass per run:

    J_E - HGI_E  =  [h.Q^ - h^(W^)]  +  [h^(W^) - h^(G)]  +  [h^(G) - HGI_E],      G = Gamma(w^ + X^),
                       T1 (SSC gap)        T2 (idling gap)       T3 (free-process residual)

with batch means (batches of 25 diffusion time units, 2 seeds, pooled 95% half-widths). T2 is the gap of
eq. (12), Gamma(w^ + X^) <= W^; Prop 3.2 bounds sup_s |W^_i - Gamma(.)_i| by 3 J c2 r^{kappa-1} + 2 r^{-1} C_i
int_0^{r^2 T} sum_j 1{E_j = 1} ds, whose two ingredients are reported per cell (the integral is r^2 T sum_j p_flag_j).
Caps: eq. (6) at the rule bands (results/multi_bench.json) and the per-pattern cap (LABELED, outside eq. (6)) at
the bands of results/vbcap_summary.json (G at c2 = 30, the best of the scan, plus the same-band c2 = 660).
Writes results/decomp_runs.json (every record carries the tables' label) and prints the table."""
import json, sys, time, warnings
from multiprocessing import Pool
import numpy as np
from scipy.stats import t as tdist
import networks as N

warnings.filterwarnings("ignore")
B = json.load(open("results/multi_bench.json"))
S = json.load(open("results/vbcap_summary.json"))
T, BURN, BATCH, SEEDS, RS = 250.0, 0.3, 25.0, (1, 2), (1000, 2000)
PP_C2 = {"A toy (Ex. 2.2)": 13.0, "B cost-trivial (Ex. 2.1)": 13.0, "C shared type costly": 13.0,
         "D unequal rates": S["D unequal rates"]["c2_used"], "E unequal loads, costs": S["E unequal loads, costs"]["c2_used"],
         "F 3x5 network": S["F 3x5 network"]["c2_used"], "G ranking-map fails (BJ20 Ex. 6.6)": 30.0}
CELLS = [(name, "uniform", B[name]["c2"]) for name in N.SPECS] + [(name, "per_pattern", PP_C2[name]) for name in N.SPECS]
CELLS.append(("G ranking-map fails (BJ20 Ex. 6.6)", "per_pattern", 660.0))
_cache = {}


def go(job):
    name, cap, c2, r, seed = job
    key = (name, cap)
    if key not in _cache:
        _cache[key] = N.build(name, vb_cap=cap)
    nw = _cache[key]
    o = nw.run(r, T, N.C1, c2, N.KAPPA, seed, burn=BURN, batch=BATCH)
    o["cap"] = cap; o["label"] = nw.tab.label
    n = nw.net
    o["prop32_layer"] = 3 * n.J * c2 * r ** (N.KAPPA - 1)
    o["prop32_flag"] = [2 * r * float(n.C[i]) * T * (1 - BURN) * float(np.sum(o["p_flag"])) for i in range(n.I)]
    return o


def hw(x):
    x = np.asarray(x, float)
    return tdist.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def table(res):
    print(f"\n{'network':22s} {'cap':11s} {'c2':>6s} {'r':>5s} | {'J_E':>7s} {'T1=SSC gap':>16s} {'T2=idling gap':>16s} {'T3=h^(G)-HGI':>16s} "
          f"| {'Efrac':>8s} {'3Jc2r^(k-1)':>11s} {'2rC_i T sum p_flag':>18s}")
    rows = []
    for name, cap, c2 in CELLS:
        H = B[name]["HGI_E"]
        for r in RS:
            runs = [o for o in res if o["net"] == name and o["cap"] == cap and o["r"] == r and o["c"][1] == c2]
            if not runs:
                continue
            hq = np.concatenate([o["batches"]["hq"] for o in runs]); hh = np.concatenate([o["batches"]["hhat"] for o in runs])
            hg = np.concatenate([o["batches"]["hg"] for o in runs])
            t1, t2, t3 = hq - hh, hh - hg, hg - H
            ef = np.mean([o["E_frac"] for o in runs])
            row = dict(net=name, cap=cap, label=runs[0]["label"], c2=c2, r=r, n_batches=len(hq), HGI_E=H,
                       JE=float(hq.mean()), JE_hw=float(hw(hq)), T1=float(t1.mean()), T1_hw=float(hw(t1)),
                       T2=float(t2.mean()), T2_hw=float(hw(t2)), T3=float(t3.mean()), T3_hw=float(hw(t3)),
                       E_frac=float(ef), prop32_layer=runs[0]["prop32_layer"],
                       prop32_flag=list(np.mean([o["prop32_flag"] for o in runs], axis=0)))
            rows.append(row)
            print(f"{name[:22]:22s} {cap:11s} {c2:6.0f} {r:5d} | {row['JE']:7.3f} {row['T1']:7.3f} ± {row['T1_hw']:5.3f} "
                  f"{row['T2']:7.3f} ± {row['T2_hw']:5.3f} {row['T3']:7.3f} ± {row['T3_hw']:5.3f} | {ef:8.1e} "
                  f"{row['prop32_layer']:11.3f} {np.max(row['prop32_flag']):18.2f}")
    return rows


def parse_cells(argv):
    """--cells 'A toy (Ex. 2.2):per_pattern:68,D unequal rates:uniform:20' -> extra (name, cap, c2) cells,
    appended to results/decomp_runs.json (existing records kept; the table is rebuilt from all records)."""
    out = []
    for a in argv:
        if a.startswith("--cells="):
            for cell in a[len("--cells="):].split(","):
                name, cap, c2 = cell.rsplit(":", 2)
                out.append((name, cap, float(c2)))
    return out


if __name__ == "__main__":
    t0 = time.time()
    extra = parse_cells(sys.argv[1:])
    nproc = int([a for a in sys.argv[1:] if a.startswith("--procs=")][0][8:]) if any(a.startswith("--procs=") for a in sys.argv[1:]) else 4
    if extra:
        CELLS = extra + [c for c in CELLS if c not in extra]          # run the extra cells, keep the rest for the table
        res = json.load(open("results/decomp_runs.json"))
        done = {(o["net"], o["cap"], o["c"][1], o["r"], o["seed"]) for o in res}
        jobs = [(name, cap, c2, r, s) for name, cap, c2 in extra for r in RS for s in SEEDS
                if (name, cap, c2, r, s) not in done]
    else:
        res = []
        jobs = [(name, cap, c2, r, s) for name, cap, c2 in CELLS for r in RS for s in SEEDS]
    jobs.sort(key=lambda j: -(j[3] ** 2 * (2 if j[0][0] in "FG" else 1)))
    with Pool(nproc) as p:
        for o in p.imap_unordered(go, jobs):
            res.append(o); json.dump(res, open("results/decomp_runs.json", "w"))
            H = B[o["net"]]["HGI_E"]
            print(f"{o['net'][:22]:22s} {o['cap']:11s} c2={o['c'][1]:5.0f} r={o['r']:5d} s={o['seed']} JE={o['JE']:7.3f} "
                  f"h^(W)={o['level']:7.3f} h^(G)={o['hgi_path']:7.3f} T3={o['hgi_path']-H:+.3f} Efrac={o['E_frac']:.1e} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    rows = table(res)
    json.dump(rows, open("results/decomp_table.json", "w"), indent=1)
    print("ALL DONE", flush=True)
