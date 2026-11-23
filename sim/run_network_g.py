"""Network G (Example 3 of Budhiraja and Johnson (2020), Math. Oper. Res. 45(3), p. 812 = Example 5.6 of arXiv:1710.09042;
ranking-map condition fails): offline pass, HGI_E, rule c2,
convergence sweep r = 1000/2000/4000 (2 seeds) + the c2 = 30 contrast. Appends to
results/multi_bench.json and results/multi_runs.json (same format as bench.py / multi_runs.py)."""
import json, time, warnings
from multiprocessing import Pool
import numpy as np
import networks as N

warnings.filterwarnings("ignore")
NAME = "G ranking-map fails (BJ20 Ex. 6.6)"
_cache = {}


def go(job):
    name, r, T, c2, s, tag = job
    if name not in _cache:
        _cache[name] = N.build(name)
    o = _cache[name].run(r, T, N.C1, c2, N.KAPPA, s)
    o["tag"] = tag
    o["pred_floor"] = N.floor_drift(_cache[name], r, o["U"], N.F_OCC)
    return o

if __name__ == "__main__":
    t0 = time.time()
    nw = N.build(NAME); t = nw.tab
    print(nw.net); print(t.summary())
    print("dual vertices:", nw.V.tolist(), " vertex-LP err", nw.vertex_err)
    m, se = nw.benchmark(); c2_rule = N.rule_c2(nw)
    # The 5%-flag-drift rule has no solution for G (b_max = 0.0028: predicted flag drift at r0 = 1000 is
    # 0.37 |theta| even at c2 = 1000, 0.11 at c2 = 3000). Fixed c2 chosen instead as the minimizer of the
    # fitted error model  a U/r + H delta/(1 - delta)  with a = 0.87 H (c2rule_fit.py), at r0 = 1000:
    c2 = 660.0
    d0 = N.floor_drift(nw, N.R0, c2 * nw.net.beta.min() * N.R0 ** N.KAPPA, N.F_OCC)
    print(f"HGI_E = {m:.4f} +- {se:.4f}   rule c2 = {c2_rule}   chosen c2 = {c2} (model optimum; "
          f"predicted flag drift {d0:.2f} |theta| at r0)   ({time.time()-t0:.0f}s)", flush=True)
    bench = json.load(open("results/multi_bench.json"))
    bench[NAME] = dict(HGI_E=m, se=se, c2=c2, c2_rule_5pct=None if np.isnan(c2_rule) else c2_rule,
                       c2_choice="argmin of a U/r + H delta/(1-delta), a = 0.87 H, r0 = 1000 (rule_c2 has no solution)",
                       b_max=t.b_max, M=len(t.M_set), trivial=bool(t.trivial),
                       theta=nw.net.theta.tolist(), nV=len(nw.V), u=t.u.tolist() if t.u is not None else None,
                       lam=t.lam, lam_tilde=t.lam_tilde, M_set=[list(z) for z in t.M_set])
    json.dump(bench, open("results/multi_bench.json", "w"), indent=1)
    jobs = [(NAME, r, T, c2, s, "rule") for r, T, ns in [(4000, 30, 2), (2000, 40, 2), (1000, 40, 2)]
            for s in range(1, ns + 1)] + [(NAME, 1000, 40, 30.0, s, "c2=30") for s in (1, 2)]
    res = json.load(open("results/multi_runs.json"))
    res = [o for o in res if o["net"] != NAME]
    with Pool(2) as p:
        for o in p.imap_unordered(go, jobs):
            res.append(o); json.dump(res, open("results/multi_runs.json", "w"))
            print(f"{o['tag']:6s} r={o['r']:5d} s={o['seed']} JE={o['JE']:7.3f} sameHGI={o['hgi_path']:6.3f} "
                  f"excess={100*(o['JE']-o['hgi_path'])/m:6.1f}% gap={o['gap']:.3f} "
                  f"1st/2nd={o['JE_1st']:.2f}/{o['JE_2nd']:.2f} floor pred={o['pred_floor']:.3f} "
                  f"Efrac={o['E_frac']:.1e} ({time.time()-t0:.0f}s)", flush=True)
    print("ALL DONE", flush=True)
