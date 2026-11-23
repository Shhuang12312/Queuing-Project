"""HGI_E for the seven networks: the benchmark values used in the report.
rsn_flat/simulation.benchmark_hgi_e with its defaults: both sqrt(dt) legs on one Brownian path per seed,
dt = (4e-3, 1e-3), T = 20000, seeds 0-7, exact dual-vertex h^.
Writes results/multi_bench_crn.json (results/multi_bench.json, which holds the bands c2 and b_max, is only read).
`exact` is filled where h^ is linear (one dual vertex y): HGI_E = sum_i y_i Sigma_ii / (2 |theta_i|)."""
import json, sys, time
import numpy as np
from scipy.stats import t as tdist
sys.path.insert(0, "../rsn_flat")
from rsn import RSN
from simulation import benchmark_hgi_e, make_hhat
import networks as N
from common import log_run

if __name__ == "__main__":
    t0 = time.time(); OLD = json.load(open("results/multi_bench.json")); out = {}
    print(f"{'network':36s} {'HGI_E (8 seeds)':>20s} {'95% hw':>8s} {'exact':>7s} {'multi_bench.json':>17s} {'old/new':>8s}", flush=True)
    for name, spec in N.SPECS.items():
        net = RSN(*spec, name=name); hh = make_hhat(net); t1 = time.time()
        m, se, d = benchmark_hgi_e(net, hhat_fn=hh, return_details=True)
        V = np.asarray(hh.vertices); exact = float(V[0] @ (np.diag(net.Sigma) / (2 * np.abs(net.theta)))) if len(V) == 1 else None
        hw = float(tdist.ppf(0.975, len(d["per_seed"]) - 1) * se)
        out[name] = dict(HGI_E=m, se=se, hw95=hw, n_seeds=len(d["seeds"]), seeds=list(d["seeds"]), T=d["T"], dt=[4e-3, 1e-3],
                         per_seed=d["per_seed"].tolist(), leg_coarse=d["per_dt"][4e-3].tolist(), leg_fine=d["per_dt"][1e-3].tolist(),
                         exact=exact, z_vs_exact=(None if exact is None else (m - exact) / se),
                         old_multi_bench=OLD[name]["HGI_E"], old_se=OLD[name]["se"], old_over_new=OLD[name]["HGI_E"] / m,
                         wall_s=time.time() - t1)
        print(f"{name[:36]:36s} {m:10.4f} ± {se:.4f} {hw:8.4f} {('%.3f' % exact) if exact is not None else '-':>7s} {OLD[name]['HGI_E']:17.4f} {OLD[name]['HGI_E'] / m:8.4f}", flush=True)
        json.dump(dict(method="rsn_flat/simulation.benchmark_hgi_e: sqrt(dt) extrapolation, both legs on the same Brownian path per seed; SE across seeds",
                       networks=out), open("results/multi_bench_crn.json", "w"), indent=1)
    dt = time.time() - t0
    log_run("python multi_bench_crn.py", "0-7 per network (default_rng(seed), one path per seed at dt = 1e-3)", dt,
            "sim/results/multi_bench_crn.json, multi_bench_crn.log", "one process")
    print(f"{dt:.0f}s", flush=True)
