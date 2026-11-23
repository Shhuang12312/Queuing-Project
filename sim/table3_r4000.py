"""Table 3 at r = 4000: T = 250, batches of 25 diffusion time units,
30% burn-in, seeds 1-2, both caps (eq. (6) and the pattern-dependent cap, which is OUTSIDE Definition 2.3), all seven
networks, with the same bands as the r = 1000 / 2000 runs of decomp_runs.json. B's two caps give identical tables, so B
runs once (eq. (6), the rule band). 26 runs, four at a time, longest first.

Every run is written to results/table3_r4000/<cell>.json the moment it finishes (atomic rename), so a crash loses only
the runs in flight; a restart skips the files that exist. At the end the runs are merged into results/table3_r4000.json.
Each record carries the per-batch time averages (batches.hq, .hhat, .hg), so T1+T2 by batch is hq - hg."""
import json, os, sys, time, warnings
from multiprocessing import Pool
import numpy as np
import networks as N
from common import log_run

warnings.filterwarnings("ignore")
B = json.load(open("results/multi_bench.json")); S = json.load(open("results/vbcap_summary.json"))
R, T, BURN, BATCH, SEEDS = 4000, 250.0, 0.3, 25.0, (1, 2)
OUT = "results/table3_r4000"
NAMES = list(N.SPECS)
PP_C2 = {"A": 13.0, "C": 13.0, "D": S["D unequal rates"]["c2_used"], "E": S["E unequal loads, costs"]["c2_used"],
         "F": S["F 3x5 network"]["c2_used"], "G": 30.0}
CELLS = [(n, "uniform", B[n]["c2"]) for n in NAMES] + [(n, "per_pattern", PP_C2[n[0]]) for n in NAMES if n[0] != "B"]
WEIGHT = {"A": 35, "B": 35, "C": 35, "D": 35, "E": 29, "F": 68, "G": 104}          # expected minutes per run


def path(name, cap, seed):
    return f"{OUT}/{name[0]}_{cap}_s{seed}.json"


def go(job):
    name, cap, c2, seed = job
    t0 = time.time(); started = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    nw = N.build(name, vb_cap=cap)
    o = nw.run(R, T, N.C1, c2, N.KAPPA, seed, burn=BURN, batch=BATCH)
    o["cap"] = cap; o["label"] = nw.tab.label; o["wall_s"] = time.time() - t0; o["started_utc"] = started
    o["finished_utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    tmp = path(name, cap, seed) + ".tmp"
    json.dump(o, open(tmp, "w")); os.replace(tmp, path(name, cap, seed))
    return o


if __name__ == "__main__":
    t0 = time.time(); os.makedirs(OUT, exist_ok=True)
    jobs = [(n, cap, c2, s) for n, cap, c2 in CELLS for s in SEEDS]
    todo = [j for j in jobs if not os.path.exists(path(j[0], j[1], j[3]))]
    todo.sort(key=lambda j: -WEIGHT[j[0][0]])
    print(f"{len(jobs)} runs, {len(jobs) - len(todo)} already on disk, {len(todo)} to do; r = {R}, T = {T}, batch = {BATCH}, seeds {SEEDS}", flush=True)
    with Pool(4, maxtasksperchild=1) as p:
        for o in p.imap_unordered(go, todo, chunksize=1):
            hq, hg = np.array(o["batches"]["hq"]), np.array(o["batches"]["hg"]); H = B[o["net"]]["HGI_E"]
            print(f"{o['finished_utc']} {o['net'][:22]:22s} {o['cap']:11s} c2={o['c'][1]:7.2f} s={o['seed']} JE={o['JE']:8.3f} "
                  f"T1+T2={np.mean(hq - hg):8.4f} ({100 * np.mean(hq - hg) / H:6.1f}% of stored HGI_E) Efrac={o['E_frac']:.1e} "
                  f"events={o['events']:.3e} wall={o['wall_s'] / 60:.1f} min (elapsed {(time.time() - t0) / 60:.0f} min)", flush=True)
    res = [json.load(open(path(*[j[0], j[1], j[3]]))) for j in jobs]
    json.dump(dict(config=dict(r=R, T=T, burn=BURN, batch=BATCH, seeds=list(SEEDS), c1=N.C1, kappa=N.KAPPA,
                               note="per_pattern = pattern-dependent cap, OUTSIDE Definition 2.3; B runs once (identical tables under both caps)"),
                   runs=res), open("results/table3_r4000.json", "w"))
    dt = time.time() - t0
    walls = ", ".join(f"{o['net'][0]}/{o['cap'][:3]}/s{o['seed']} {o['wall_s'] / 60:.0f}" for o in res)
    log_run("python table3_r4000.py", "1, 2 per cell (13 cells: 7 networks x eq. (6), 6 networks x pattern cap)", dt,
            "sim/results/table3_r4000.json, table3_r4000/<cell>.json, table3_r4000.log",
            f"26 runs, 4 at a time; wall minutes per run (net/cap/seed): {walls}")
    print(f"ALL DONE {dt:.0f}s", flush=True)
