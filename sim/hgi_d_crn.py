"""HGI_D for the toy (w0 = 0, varsigma = 1) with both sqrt(dt) legs on the same Brownian paths, 10 000 paths per set.
rsn_flat/simulation.benchmark_hgi_d pairs the legs (the coarse path is the fine one summed in pairs), so the value
of results/hgi_d_toy.json is a common-path value. This script (i) reruns that configuration (seed 12345), which must
reproduce it, and (ii) draws an independent set of 10 000 paths (seed 20261117). dt = (1e-3, 5e-4), T = 20.
Writes results/hgi_d_toy_crn.json."""
import json, math, sys, time
sys.path.insert(0, "../rsn_flat")
from rsn import make_2lln
from simulation import benchmark_hgi_d
from common import log_run

if __name__ == "__main__":
    t0 = time.time(); old = json.load(open("results/hgi_d_toy.json"))["1.0"]; runs = {}
    for seed in (12345, 20261117):
        t1 = time.time(); m, se, d = benchmark_hgi_d(make_2lln(), seed=seed, return_details=True)
        runs[str(seed)] = {"HGI_D": m, "se": se, "leg_dt_1e-3": d["per_dt"][1e-3][0], "leg_dt_5e-4": d["per_dt"][5e-4][0],
                           "se_difference": d["per_dt"]["se_difference"], "n_paths": d["n_paths"], "T": d["T"], "wall_s": time.time() - t1}
        print(f"seed {seed}: HGI_D = {m:.4f} ± {se:.4f}  legs {d['per_dt'][1e-3][0]:.4f} (dt 1e-3), {d['per_dt'][5e-4][0]:.4f} (dt 5e-4)  ({time.time() - t1:.0f}s)", flush=True)
    a, b = runs["12345"], runs["20261117"]
    pooled = dict(HGI_D=(a["HGI_D"] + b["HGI_D"]) / 2, se=math.sqrt(a["se"] ** 2 + b["se"] ** 2) / 2, n_paths=20000)
    out = dict(method="rsn_flat/simulation.benchmark_hgi_d: sqrt(dt) extrapolation from dt = (1e-3, 5e-4), both legs on the same Brownian paths",
               stored_nov14=dict(HGI_D=old["HGI_D"], se=old["se"]), reproduces_stored=abs(a["HGI_D"] - old["HGI_D"]) < 1e-12,
               runs=runs, pooled=pooled, z_between_sets=(a["HGI_D"] - b["HGI_D"]) / math.hypot(a["se"], b["se"]))
    json.dump(out, open("results/hgi_d_toy_crn.json", "w"), indent=1)
    print(f"reproduces the stored value bit for bit: {out['reproduces_stored']}; pooled {pooled['HGI_D']:.4f} ± {pooled['se']:.4f}; z between the two sets {out['z_between_sets']:+.2f}", flush=True)
    dt = time.time() - t0
    log_run("python hgi_d_crn.py", "12345 (the stored configuration), 20261117 (independent set); 10 000 paths each", dt,
            "sim/results/hgi_d_toy_crn.json, hgi_d_toy_crn.log", "one process")
    print(f"{dt:.0f}s", flush=True)
