"""Reproducibility check of the stored Table 3 runs at r = 4000 (results/table3_r4000.json).
Runs one cell (network E, Definition 2.3, seed 1; about 23 minutes) and compares it with the stored record:
J_E, the per-batch arrays hq / hhat / hg, E_frac and the event count.
Writes results/table3_r4000_repro_check.json."""
import json, time
import numpy as np
import table3_r4000 as T3
from common import log_run

if __name__ == "__main__":
    t0 = time.time(); name, cap, seed = "E unequal loads, costs", "uniform", 1
    c2 = [c for n, cp, c in T3.CELLS if n == name and cp == cap][0]
    old = [o for o in json.load(open("results/table3_r4000.json"))["runs"] if o["net"] == name and o["cap"] == cap and o["seed"] == seed][0]
    nw = T3.N.build(name, vb_cap=cap)
    new = nw.run(T3.R, T3.T, T3.N.C1, c2, T3.N.KAPPA, seed, burn=T3.BURN, batch=T3.BATCH)
    cmp = {k: bool(new[k] == old[k]) for k in ("JE", "level", "E_frac", "events", "JE_1st", "JE_2nd")}
    for k in ("hq", "hhat", "hg"):
        cmp["batches." + k] = bool(np.array_equal(np.array(new["batches"][k]), np.array(old["batches"][k])))
        cmp["batches." + k + ".max_abs_diff"] = float(np.max(np.abs(np.array(new["batches"][k]) - np.array(old["batches"][k]))))
    out = dict(cell=dict(net=name, cap=cap, c2=c2, r=T3.R, T=T3.T, batch=T3.BATCH, seed=seed), comparison=cmp,
               identical=all(v for k, v in cmp.items() if not k.endswith("max_abs_diff")), JE_new=new["JE"], JE_stored=old["JE"], wall_s=time.time() - t0)
    json.dump(out, open("results/table3_r4000_repro_check.json", "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)
    log_run("python table3_repro_check.py", "1 (network E, Definition 2.3, r = 4000, T = 250)", time.time() - t0,
            "sim/results/table3_r4000_repro_check.json, table3_r4000_repro_check.log", f"one stored cell rerun and compared; identical = {out['identical']}")
