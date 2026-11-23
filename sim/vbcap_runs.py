"""The pattern-dependent cap as a LABELED EXPERIMENT (OUTSIDE eq. (6)).

OfflineTables(vb_cap="per_pattern"):  ||v^b(z)|| <= |lambda_c(z)|/(4|beta||h|) on M,  rho*/4 off M
(Definition 2.3 / eq. (6) uses the uniform min(rho*/4, |lambda~|/(4|beta||h|)) everywhere).

Part 1 (all seven networks): verify  h beta.(v^c(z)+v^b(z)) <= -3/4|lambda_c(z)| <= -3/4|lambda~|  on M
(the inequality step (47) of the proof of Prop 3.5 needs), tabulate the push ||v^b(z)|| under both caps,
and compute the rule c2 (5% predicted flag drift at r0 = 1000, f = 0.5) for the per-pattern push.
Part 2 (D, E, F, G): r = 1000, 2000 (2 seeds) with the per-pattern tables and that c2; print
side by side with the eq. (6) runs stored in results/multi_runs.json.
Outputs: results/vbcap_summary.json, results/vbcap_runs.json (every record labeled), results/vbcap_runs.log."""
import json, sys, time, warnings
from multiprocessing import Pool
import numpy as np
import networks as N

warnings.filterwarnings("ignore")
LABEL = "outside eq. (6): per-pattern cap"
RUN_NETS = ["D unequal rates", "E unequal loads, costs", "F 3x5 network", "G ranking-map fails (BJ20 Ex. 6.6)"]
B = json.load(open("results/multi_bench.json"))
R = json.load(open("results/multi_runs.json"))
OCC = json.load(open("results/ssc_occupancy.json"))
_cache = {}


def model_c2(nw, H, a_over_H=0.87, r0=N.R0, f=N.F_OCC, grid=np.arange(5.0, 3000.0, 1.0)):
    """Minimizer of the error model a U/r + H delta/(1-delta) at r0 (used when rule_c2 has no solution)."""
    best, best_c2 = np.inf, np.nan
    for c2 in grid:
        U = c2 * nw.net.beta.min() * r0 ** N.KAPPA
        d = N.floor_drift(nw, r0, U, f)
        if d < 1.0:
            v = a_over_H * H * U / r0 + H * d / (1.0 - d)
            if v < best:
                best, best_c2 = v, float(c2)
    return best_c2, best


def offline_part():
    summ = {}
    for name in N.SPECS:
        nu, npp = N.build(name), N.build(name, vb_cap="per_pattern")
        tu, tp = nu.tab, npp.tab
        c47u, c47p = tu.check_step47(), tp.check_step47()
        su, sp = tu.push_stats(), tp.push_stats()
        H = B[name]["HGI_E"]
        f_ssc = np.asarray(OCC[name]) if name in OCC else N.F_OCC
        row = dict(net=name, label_uniform=tu.label, label_per_pattern=tp.label,
                   lam=tu.lam, lam_tilde=tu.lam_tilde, M=len(tu.M_set), b_max_eq6=tu.b_max,
                   rho_star_over_4=float(nu.net.rho_star / 4),
                   b_cap_in_M=(sorted({round(tp.b_cap(z), 9) for z in tp.M_set}) if tp.M_set else []),
                   step47_uniform=c47u, step47_per_pattern=c47p, push_uniform=su, push_per_pattern=sp,
                   c2_rule_uniform=float(N.rule_c2(nu)), c2_rule_per_pattern=float(N.rule_c2(npp)),
                   c2_rule_fssc_uniform=float(N.rule_c2(nu, f=f_ssc)), c2_rule_fssc_per_pattern=float(N.rule_c2(npp, f=f_ssc)),
                   c2_phase2=B[name]["c2"])
        row["c2_model_uniform"], _ = model_c2(nu, H)          # error model minimizer, a = 0.87 H
        row["c2_model_per_pattern"], _ = model_c2(npp, H)
        # the worst-case below-U service deficit mu_j = log(a_j/d_j) of each local type (drives the flag formula)
        for tag, nw in (("uniform", nu), ("per_pattern", npp)):
            n = nw.net; a = n.alphas_r(N.R0); mu = []
            for j in range(n.J):
                xb = max(nw.xtab[i, j] for i in range(2 ** n.J) if (i >> j) & 1)
                mu.append(float(np.log(a[j] / (n.beta[j] * xb))))
            row[f"mu_worst_{tag}"] = mu
        summ[name] = row
        print(f"{name[:26]:26s} |M|={row['M']:3d} lam~={0 if tu.lam_tilde is None else tu.lam_tilde:8.4f} "
              f"b_max={tu.b_max:.5f} rho*/4={row['rho_star_over_4']:.4f} | step47 margin uni {c47u.get('min_margin_tilde') or 0:.3f} "
              f"per {c47p.get('min_margin') or 0:.3f} | push: eq6 {su['all']['min']:.4f}; per-pattern on M "
              f"{sp['in_M'].get('min', 0):.4f}-{sp['in_M'].get('max', 0):.4f}, off M {sp['off_M'].get('min', 0):.4f} | "
              f"mu_worst eq6 {np.round(row['mu_worst_uniform'], 4).tolist()} per {np.round(row['mu_worst_per_pattern'], 4).tolist()} | "
              f"c2 rule eq6 {row['c2_rule_uniform']:.1f} -> per-pattern {row['c2_rule_per_pattern']:.1f}"
              + f" (model minimizer {row['c2_model_uniform']:.0f} -> {row['c2_model_per_pattern']:.0f})"
              + f"; f_ssc variant {row['c2_rule_fssc_uniform']:.1f} -> {row['c2_rule_fssc_per_pattern']:.1f}", flush=True)
    json.dump(summ, open("results/vbcap_summary.json", "w"), indent=1, default=float)
    return summ


def go(job):
    name, r, T, c2, s = job
    if name not in _cache:
        _cache[name] = N.build(name, vb_cap="per_pattern")
    nw = _cache[name]
    o = nw.run(r, T, N.C1, c2, N.KAPPA, s)
    o["tag"] = "per_pattern"; o["label"] = nw.tab.label
    o["pred_floor"] = N.floor_drift(nw, r, o["U"], N.F_OCC)
    return o


def side_by_side(summ, res):
    print("\n=== side by side: same-path excess (J_E - hgi_path)/HGI_E, mean +- SE over seeds ===")
    print(f"{'network':26s} {'r':>5s} | {'eq. (6) c2':>10s} {'excess':>16s} {'E_frac':>8s} | {'per-pattern c2':>14s} {'excess':>16s} {'E_frac':>8s}  [per-pattern = OUTSIDE eq. (6)]")
    rows = []
    for name in RUN_NETS:
        H = B[name]["HGI_E"]
        for r in (1000, 2000):
            def agg(runs):
                e = np.array([100 * (o["JE"] - o["hgi_path"]) / H for o in runs])
                ef = np.mean([o["E_frac"] for o in runs]) if runs else np.nan
                se = e.std(ddof=1) / np.sqrt(len(e)) if len(e) > 1 else 0.0
                return (e.mean() if len(e) else np.nan), se, ef, len(e)
            u = agg([o for o in R if o["net"] == name and o["tag"] == "rule" and o["r"] == r])
            p = agg([o for o in res if o["net"] == name and o["r"] == r])
            c2p = summ[name].get("c2_used", np.nan)
            rows.append(dict(net=name, r=r, eq6=dict(c2=B[name]["c2"], excess_pct=u[0], se=u[1], E_frac=u[2], n=u[3]),
                             per_pattern=dict(c2=c2p, excess_pct=p[0], se=p[1], E_frac=p[2], n=p[3], label=LABEL)))
            print(f"{name[:26]:26s} {r:5d} | {B[name]['c2']:10.1f} {u[0]:7.1f} +- {u[1]:5.1f}% {u[2]:8.1e} | "
                  f"{c2p:14.1f} {p[0]:7.1f} +- {p[1]:5.1f}% {p[2]:8.1e}")
    return rows


if __name__ == "__main__":
    t0 = time.time()
    summ = offline_part()
    for name in RUN_NETS:
        c2 = summ[name]["c2_rule_per_pattern"]
        if np.isnan(c2):
            c2 = summ[name]["c2_model_per_pattern"]
        summ[name]["c2_used"] = float(c2)
    json.dump(summ, open("results/vbcap_summary.json", "w"), indent=1, default=float)
    if "--offline-only" in sys.argv:
        sys.exit(0)
    jobs = []
    for name in RUN_NETS:
        plan = [(2000, 30, 2), (1000, 40, 2)] if name.startswith("F") else [(2000, 40, 2), (1000, 40, 2)]
        jobs += [(name, r, T, summ[name]["c2_used"], s) for r, T, ns in plan for s in range(1, ns + 1)]
    jobs.sort(key=lambda j: -(j[1] ** 2 * j[2] * (2 if j[0][0] in "FG" else 1)))
    res = []
    with Pool(4) as p:
        for o in p.imap_unordered(go, jobs):
            res.append(o); json.dump(res, open("results/vbcap_runs.json", "w"))
            H = B[o["net"]]["HGI_E"]
            print(f"{o['net'][:24]:24s} per-pattern c2={o['c'][1]:6.1f} r={o['r']:5d} s={o['seed']} JE={o['JE']:7.3f} "
                  f"sameHGI={o['hgi_path']:6.3f} excess={100*(o['JE']-o['hgi_path'])/H:6.1f}% gap={o['gap']:.3f} "
                  f"1st/2nd={o['JE_1st']:.2f}/{o['JE_2nd']:.2f} floor pred={o['pred_floor']:.3f} "
                  f"Efrac={o['E_frac']:.1e} ({time.time()-t0:.0f}s)", flush=True)
    rows = side_by_side(summ, res)
    json.dump(dict(label=LABEL, rows=rows), open("results/vbcap_side_by_side.json", "w"), indent=1, default=float)
    print("ALL DONE", flush=True)
