"""HGI_D benchmark for the toy from an empty start (paper eq. (11), unnormalized), sqrt(dt)-extrapolated from
dt = (1e-3, 5e-4) with 10000 paths (both legs on the same Brownian paths), for varsigma = 1 (the only value used in any queueing run) and, for the
record, 0.5 and 2. Writes results/hgi_d_toy.json."""
import json, subprocess, sys, time
sys.path.insert(0, "../rsn_flat")
from rsn import make_2lln
from simulation import benchmark_hgi_d

if __name__ == "__main__":
    t0 = time.time(); out = {}
    for vs in (1.0, 0.5, 2.0):
        m, se, d = benchmark_hgi_d(make_2lln(), varsigma=vs, return_details=True)
        out[str(vs)] = dict(HGI_D=m, se=se, per_dt={str(k): (list(v) if isinstance(v, tuple) else v) for k, v in d["per_dt"].items()},
                            T=d["T"], n_paths=d["n_paths"], legs_paired="common random numbers: coarse path = fine increments summed in pairs")
        print(f"varsigma = {vs}: HGI_D = {m:.4f} ± {se:.4f}  legs {d['per_dt']}  ({time.time()-t0:.0f}s)", flush=True)
    out["commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd="..").stdout.strip()
    out["note"] = ("Unnormalized convention of eq. (11); all queueing-side J_D use varsigma = 1 (toykern3.c). "
                   "A check at dt = 2e-3 / 1e-3 / 5e-4 gave 1.2846 / 1.2998 / 1.3107 at varsigma = 1.")
    json.dump(out, open("results/hgi_d_toy.json", "w"), indent=1)
    print("saved results/hgi_d_toy.json", flush=True)
