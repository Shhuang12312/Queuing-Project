"""Does phi = T1/(c2 r^(kappa-1)) = T1 r/U have a constant limit?

T1 = h.Q^ - h^(W^) is nearly deterministic along a path, so every stored run with T1 can be used:
  toy c2 = 60:  paper_cells.json (r = 300..1e4, seeds 1-2) + drift_check.json (r = 1000, 2000, seeds 3-8) + final_runs.json
  toy c2 = 68 and networks B-G at the rule band:  multi_runs.json 'gap' (r = 1000, 2000, 4000) + decomp_runs.json (r = 1000, 2000)
  pattern-dependent cap (outside eq. (2)):  decomp_runs.json (r = 1000, 2000), paper_cells.json (G, r = 4000)
Model with a finite-r correction:  T1 r = phi_inf U - b   <=>   phi(r) = phi_inf - b/U,  U = c2 r^kappa (in jobs).
For the toy there is a mechanism: the smaller local queue sits just below U with an upward drift mu = log(a/d), so its
mean depth below U is about 1/mu, giving phi_inf = 1 and b ~ 1/mu = 44 jobs.
Run from anywhere: python analysis/phi_fit.py  -> analysis/phi_fit.json"""
import json, math, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RES = os.path.join(ROOT, "sim", "results")
J = lambda n: json.load(open(os.path.join(RES, n)))
B = J("multi_bench.json"); R = J("multi_runs.json"); D = J("decomp_runs.json"); PC = J("paper_cells.json"); DC = J("drift_check.json")["runs"]; FIN = J("final_runs.json")
K = 0.24


def pts_from(runs, t1):
    out = {}
    for o in runs:
        out.setdefault((o["r"], round(o["U"], 3)), []).append(t1(o))
    return sorted((r, U, float(np.mean(v)), float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan"), len(v)) for (r, U), v in out.items())


def fit(pts, label, fix_phi=None):
    r = np.array([p[0] for p in pts], float); U = np.array([p[1] for p in pts]); y = np.array([p[2] for p in pts]) * r      # T1 r, in jobs
    phi = y / U
    res = dict(label=label, points=[dict(r=int(a), U=float(b), T1=float(c), T1_se=d, n=e, phi=float(c * a / b)) for a, b, c, d, e in pts])
    if len(pts) >= 2:
        A = np.vstack([U, -np.ones_like(U)]).T; coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef; dof = max(len(y) - 2, 0)
        cov = np.linalg.inv(A.T @ A) * (resid @ resid / dof) if dof > 0 else np.full((2, 2), np.nan)
        res.update(phi_inf=float(coef[0]), phi_inf_se=float(math.sqrt(cov[0, 0])) if dof > 0 else float("nan"), b=float(coef[1]), b_se=float(math.sqrt(cov[1, 1])) if dof > 0 else float("nan"),
                   max_abs_resid_jobs=float(np.max(np.abs(resid))), dof=dof)
        # constant-phi alternative (no correction): phi = mean; and a pure power correction phi = phi_inf - c r^-p is not identifiable with <= 5 points
        res["constant_phi_spread_pct"] = float(100 * (phi.max() / phi.min() - 1))
    return res


out = []
toy = "A toy (Ex. 2.2)"
t1_batches = lambda o: float(np.mean(np.array(o["batches"]["hq"]) - np.array(o["batches"]["hhat"])))
# toy, c2 = 60
runs60 = [o for o in PC if o["net"] == toy and o["c"][1] == 60.0] + [o for o in DC if o["part"] == "b"]
out.append(fit(pts_from(runs60, t1_batches), "A (toy), eq. (2), c2 = 60, r = 300..1e4 [paper_cells + drift_check]"))
old60 = [dict(r=x["r"], U=x["U"], gap=x["gap"]) for x in FIN if x["c2"] == 60]
out.append(fit(pts_from(old60, lambda o: o["gap"]), "A (toy), eq. (2), c2 = 60, October toy-kernel runs [final_runs]"))
for n in B:
    L = n[0]
    rule = [o for o in R if o["net"] == n and o["tag"] == "rule"] 
    dec = [dict(r=o["r"], U=o["U"], gap=t1_batches(o)) for o in D if o["net"] == n and o["cap"] == "uniform" and abs(o["c"][1] - B[n]["c2"]) < 1e-6]
    if L != "B":
        out.append(fit(pts_from([dict(r=o["r"], U=o["U"], gap=o["gap"]) for o in rule] + dec, lambda o: o["gap"]), f"{L}, eq. (2), c2 = {B[n]['c2']:.0f}, r = 1000, 2000, 4000 [multi_runs + decomp_runs]"))
    for c2 in sorted({round(o["c"][1], 3) for o in D if o["net"] == n and o["cap"] == "per_pattern"}):
        pp = [dict(r=o["r"], U=o["U"], gap=t1_batches(o)) for o in D if o["net"] == n and o["cap"] == "per_pattern" and abs(o["c"][1] - c2) < 1e-3]
        pp += [dict(r=o["r"], U=o["U"], gap=t1_batches(o)) for o in PC if o["net"] == n and o["cap"] == "per_pattern" and abs(o["c"][1] - c2) < 1e-3]
        if L != "B" and len({p["r"] for p in pp}) >= 2:
            out.append(fit(pts_from(pp, lambda o: o["gap"]), f"{L}, pattern-dependent cap (outside eq. (2)), c2 = {c2:.0f} [decomp_runs (+ paper_cells)]"))
d = 0.976896
print(f"toy mechanism: mu = log(a/d), 1/mu = {1/math.log((1-0.5/300)/d):.1f} (r = 300), {1/math.log((1-0.5/1000)/d):.1f} (r = 1000), {1/math.log((1-0.5e-4)/d):.1f} (r = 1e4), {-1/math.log(d):.1f} (r = inf)")
for f in out:
    ph = ", ".join(f"{p['r']}: {p['phi']:.4f}" for p in f["points"])
    extra = (f"  ->  phi_inf = {f['phi_inf']:.3f} ± {f['phi_inf_se']:.3f}, b = {f['b']:.1f} ± {f['b_se']:.1f} jobs, max residual {f['max_abs_resid_jobs']:.2f} jobs ({f['dof']} dof)" if "phi_inf" in f else "")
    print(f"{f['label']}\n    phi(r): {ph}  (spread {f.get('constant_phi_spread_pct', float('nan')):.1f}%){extra}")
json.dump(out, open(os.path.join(ROOT, "analysis", "phi_fit.json"), "w"), indent=1)
