"""Error model for the excess cost, fitted on the multi-network runs, and the band c2*(r) that minimizes it.

    excess(c2, r) ~ a * U / r  +  HGI_E * delta / (1 - delta),   U = c2 beta_min r^kappa,
    delta = floor_drift(c2, r) = predicted hysteresis idle drift / |theta| (flagmodel, f = 0.5).

The first term is the threshold boundary layer (the workload is effectively reflected at ~U/r instead
of 0); the second is the flag inflation: an RBM whose drift is reduced by the fraction delta has its
stationary mean, hence its cost, inflated by 1/(1 - delta) (transient at delta -> 1).
`a` is calibrated per network from the rule-band runs (delta small there). Out-of-sample check: the
model's prediction at c2 = 30, r = 1000 on the toy vs the measured runs.
c2*(r) = argmin_c2 of the model. Writes results/c2rule_fit.json; prints a table."""
import json
import numpy as np
import networks as N

B = json.load(open("results/multi_bench.json"))
R = json.load(open("results/multi_runs.json"))
C2_GRID = np.arange(5.0, 400.0, 1.0)


def model(nw, H, a, c2, r, f=N.F_OCC):
    """Predicted excess J_E - same-path HGI at (c2, r); f = below-U occupancy (scalar or per type)."""
    U = c2 * nw.net.beta.min() * r ** N.KAPPA
    d = N.floor_drift(nw, r, U, f)
    if not d < 1.0:
        return np.inf, d
    return a * U / r + H * d / (1.0 - d), d


def measured_occupancy(name):
    """Per-type P(Q_j < U, E_j = 0) averaged over all stored runs of the network (it is
    independent of r and c2 to within a few %: a property of the network's SSC dynamics)."""
    runs = [o for o in R if o["net"] == name]
    return np.mean([o["f_below"] for o in runs], axis=0)


def fit(name, nw=None, occ=None):
    nw = nw or N.build(name)
    H = B[name]["HGI_E"]
    rule = [o for o in R if o["net"] == name and o["tag"] == "rule"]
    variants = {"f=0.5": N.F_OCC, "f_ssc": np.asarray(occ[name]) if occ else N.ssc_occupancy(nw),
                "f_meas": measured_occupancy(name)}
    out = dict(net=name, HGI_E=H, n_rule_runs=len(rule), c2_rule=B[name]["c2"], b_max=B[name]["b_max"],
               f_ssc=[float(v) for v in variants["f_ssc"]], f_meas=[float(v) for v in variants["f_meas"]],
               variants={})
    con = [o for o in R if o["net"] == name and o["tag"] == "c2=30"]
    for lab, f in variants.items():
        # boundary-layer coefficient a: from the rule-band runs, after removing the (small) flag term
        a_runs = []
        for o in rule:
            d = N.floor_drift(nw, o["r"], o["U"], f)
            a_runs.append((o["JE"] - o["hgi_path"] - H * d / (1 - d)) * o["r"] / o["U"])
        a_runs = np.array(a_runs)
        a, a_se = a_runs.mean(), a_runs.std(ddof=1) / np.sqrt(len(a_runs))
        v = dict(a=float(a), a_se=float(a_se), a_over_H=float(a / H))
        if con:
            meas = np.array([o["JE"] - o["hgi_path"] for o in con])
            pred, d30 = model(nw, H, a, 30.0, 1000.0, f)
            v["check_c2_30_r1000"] = dict(measured=float(meas.mean()),
                                         measured_se=float(meas.std(ddof=1) / np.sqrt(len(meas))),
                                         predicted=float(pred), delta=float(d30))
        for r in (1000.0, 2000.0, 4000.0):
            vals = np.array([model(nw, H, a, c2, r, f)[0] for c2 in C2_GRID])
            k = int(np.argmin(vals))
            pr, dr = model(nw, H, a, B[name]["c2"], r, f)
            v[f"c2_star_{int(r)}"] = dict(c2=float(C2_GRID[k]), pred_excess=float(vals[k]), pred_pct=float(100 * vals[k] / H),
                                          delta=float(model(nw, H, a, C2_GRID[k], r, f)[1]),
                                          pred_excess_rule=float(pr), pred_pct_rule=float(100 * pr / H), delta_rule=float(dr))
        v["c2_rule_with_f"] = float(N.rule_c2(nw, f=f))
        out["variants"][lab] = v
    return out


if __name__ == "__main__":
    occ = json.load(open("results/ssc_occupancy.json"))
    res = {}
    for name in N.SPECS:
        if name not in B or not any(o["net"] == name and o["tag"] == "rule" for o in R):
            continue
        res[name] = fit(name, occ=occ if name in occ else None)
    json.dump(res, open("results/c2rule_fit.json", "w"), indent=1)
    for lab in ("f=0.5", "f_ssc", "f_meas"):
        print(f"\n=== occupancy variant {lab} ===")
        print(f"{'network':26s} {'a/H':>5s} {'c2 rule':>7s} {'rule(f)':>7s} | {'c2*(1e3)':>8s} {'pred%':>6s} {'rule%':>6s} | "
              f"{'c2*(2e3)':>8s} {'pred%':>6s} | {'c2*(4e3)':>8s} {'pred%':>6s} | c2=30@1e3 measured vs predicted")
        for name, f in res.items():
            v = f["variants"][lab]; chk = v.get("check_c2_30_r1000")
            chk_s = (f"{chk['measured']:.3f}±{chk['measured_se']:.3f} vs {chk['predicted']:.3f} (delta {chk['delta']:.3f})"
                     if chk else "-")
            print(f"{name[:26]:26s} {v['a_over_H']:5.2f} {f['c2_rule']:7.1f} {v['c2_rule_with_f']:7.1f} | "
                  f"{v['c2_star_1000']['c2']:8.0f} {v['c2_star_1000']['pred_pct']:6.1f} {v['c2_star_1000']['pred_pct_rule']:6.1f} | "
                  f"{v['c2_star_2000']['c2']:8.0f} {v['c2_star_2000']['pred_pct']:6.1f} | "
                  f"{v['c2_star_4000']['c2']:8.0f} {v['c2_star_4000']['pred_pct']:6.1f} | {chk_s}")
    print("\nmeasured below-U occupancy f_meas per type:")
    for name, f in res.items():
        print(f"  {name[:26]:26s} f_meas={np.round(f['f_meas'], 3).tolist()}  f_ssc={np.round(f['f_ssc'], 3).tolist()}")
