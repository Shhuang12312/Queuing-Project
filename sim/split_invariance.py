"""Split invariance of Condition 2 (the paper's (alpha_bar, beta_bar) form) on the toy, as a statistical test.

Three splits with the same eta = -1/2 (hence the same theta, Sigma and limit):
    all-alpha (canonical): (alpha_bar, beta_bar) = (-0.5, 0)      alpha^r = 1 - 0.5/r, beta^r = 1
    all-beta:              (0, +0.5)                               alpha^r = 1,         beta^r = 1 + 0.5/r
    mixed:                 (-0.25, +0.25)
Toy, (c1, c2, kappa) = (0.1, 60, 0.24), r = 1000, T = 160, 30% burn-in, batches of 16 diffusion units
(7 per seed), the one-pass kernel with M^r in the workload. Per split: J_E, T1 = h.Q^ - h^(W^), T2 = h^(W^) - h^(G),
T1 + T2, T3 = h^(G) - HGI_E, each with the batch-means 95% half-width (t quantile, 28 batch means), and the
pairwise across-split z = (m_a - m_b)/sqrt(se_a^2 + se_b^2).
  (A) independent seeds, different across splits (1-4, 11-14, 21-24): the statistical test;
  (B) the SAME seeds 1-4 for all three splits: a code-consistency check only (one shared clock stream, rates
      differing at O(1/r) -> nearly identical event sequences), reported as split-to-split differences per seed.
Writes results/split_invariance.json (with the git commit hash) and prints the tables.
"""
import json, subprocess, sys, time
from multiprocessing import Pool
import numpy as np
from scipy.stats import t as tdist
import gensim

K2 = [[1, 0, 1], [0, 1, 1]]
R, T, BURN, BATCH, C = 1000, 160.0, 0.3, 16.0, (0.1, 60.0, 0.24)
HGI_E = 2.8094
SPLITS = {"all-alpha": dict(alpha_bar=[-0.5] * 3, beta_bar=[0.0] * 3),
          "all-beta": dict(alpha_bar=[0.0] * 3, beta_bar=[0.5] * 3),
          "mixed": dict(alpha_bar=[-0.25] * 3, beta_bar=[0.25] * 3)}
SEEDS_INDEP = {"all-alpha": (1, 2, 3, 4), "all-beta": (11, 12, 13, 14), "mixed": (21, 22, 23, 24)}
SEEDS_SAME = (1, 2, 3, 4)
_nets = {}


def net(name):
    if name not in _nets:
        _nets[name] = gensim.Net(name, K2, [1, 1, 1], [1, 1, 1], [1, 1, 1], **SPLITS[name])
    return _nets[name]


def run(job):
    name, seed = job
    o = net(name).run(R, T, *C, seed, burn=BURN, batch=BATCH)
    b = o["batches"]
    return dict(split=name, seed=seed, JE=o["JE"], level=o["level"], hgi_path=o["hgi_path"], E_frac=o["E_frac"],
                checks=o["checks"], hq=b["hq"], hhat=b["hhat"], hg=b["hg"], n_batch=b["n"])


def stats(runs):
    hq = np.concatenate([o["hq"] for o in runs]); hh = np.concatenate([o["hhat"] for o in runs]); hg = np.concatenate([o["hg"] for o in runs])
    out = {}
    for key, x in (("JE", hq), ("T1", hq - hh), ("T2", hh - hg), ("T1+T2", hq - hg), ("T3", hg - HGI_E)):
        n = len(x); se = x.std(ddof=1) / np.sqrt(n)
        out[key] = dict(mean=float(x.mean()), se=float(se), hw95=float(tdist.ppf(0.975, n - 1) * se), n=int(n))
    out["E_frac"] = float(np.mean([o["E_frac"] for o in runs]))
    return out


if __name__ == "__main__":
    t0 = time.time()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd="..").stdout.strip()
    jobs = [(name, s) for name, seeds in SEEDS_INDEP.items() for s in seeds]
    jobs += [(name, s) for name in ("all-beta", "mixed") for s in SEEDS_SAME]       # all-alpha seeds 1-4 already in (A)
    res = {}
    with Pool(4) as p:
        for o in p.imap_unordered(run, jobs):
            res[(o["split"], o["seed"])] = o
            print(f"{o['split']:9s} seed {o['seed']:2d}: JE={o['JE']:.4f} h^(W)={o['level']:.4f} h^(G)={o['hgi_path']:.4f} "
                  f"E_frac={o['E_frac']:.1e} ({time.time()-t0:.0f}s)", flush=True)
    for name in SPLITS:
        n = net(name).net
        print(f"{name}: alpha^r={n.alphas_r(R).round(6).tolist()} beta^r={n.betas_r(R).round(6).tolist()} eta={n.eta.tolist()} "
              f"theta={n.theta.tolist()} Sigma={n.Sigma.tolist()}")

    # (A) independent seeds
    A = {name: stats([res[(name, s)] for s in seeds]) for name, seeds in SEEDS_INDEP.items()}
    print(f"\n(A) independent seeds (different across splits), r = {R}, T = {T}, batches of {BATCH} -> 28 batch means per split; ± = 95% half-width")
    print(f"{'split':10s} {'J_E':>16s} {'T1':>16s} {'T2':>16s} {'T1+T2':>16s} {'T3':>16s} {'E_frac':>8s}")
    for name, st in A.items():
        print(f"{name:10s} " + " ".join(f"{st[k]['mean']:7.4f} ± {st[k]['hw95']:6.4f}" for k in ("JE", "T1", "T2", "T1+T2", "T3")) + f" {st['E_frac']:8.1e}")
    pairs = [("all-alpha", "all-beta"), ("all-alpha", "mixed"), ("all-beta", "mixed")]
    zA = {}
    print("pairwise z = (m_a - m_b)/sqrt(se_a^2 + se_b^2):")
    for a, b in pairs:
        zA[f"{a} vs {b}"] = {k: float((A[a][k]["mean"] - A[b][k]["mean"]) / np.hypot(A[a][k]["se"], A[b][k]["se"])) for k in ("T1", "T2", "T1+T2", "JE", "T3")}
        print(f"  {a:9s} vs {b:9s}: " + "  ".join(f"{k} z={zA[f'{a} vs {b}'][k]:+.2f}" for k in ("T1", "T2", "T1+T2", "JE", "T3")))

    # (B) same seeds: code-consistency check
    print(f"\n(B) SAME seeds {SEEDS_SAME} for all three splits — code-consistency check, NOT a statistical test")
    B = {}
    for s in SEEDS_SAME:
        a, b, m = (res[(n_, s)] for n_ in ("all-alpha", "all-beta", "mixed"))
        row = {}
        for k in ("JE", "hgi_path", "level"):
            row[k] = dict(all_alpha=a[k], all_beta=b[k], mixed=m[k], d_beta=b[k] - a[k], d_mixed=m[k] - a[k])
        row["T1+T2"] = dict(all_alpha=a["JE"] - a["hgi_path"], all_beta=b["JE"] - b["hgi_path"], mixed=m["JE"] - m["hgi_path"])
        B[s] = row
        print(f"  seed {s}: J_E {a['JE']:.4f} / {b['JE']:.4f} / {m['JE']:.4f}  (beta - alpha {b['JE']-a['JE']:+.4f}, mixed - alpha {m['JE']-a['JE']:+.4f}); "
              f"T1+T2 {row['T1+T2']['all_alpha']:.4f} / {row['T1+T2']['all_beta']:.4f} / {row['T1+T2']['mixed']:.4f}")
    same = {name: stats([res[(name, s)] for s in SEEDS_SAME]) for name in SPLITS}
    print("  same-seed split means (for reference only): " + "; ".join(f"{n_}: J_E {same[n_]['JE']['mean']:.4f}, T1+T2 {same[n_]['T1+T2']['mean']:.4f}" for n_ in SPLITS))

    json.dump(dict(commit=commit, config=dict(network="toy (Example 2.2)", K=K2, alpha=[1, 1, 1], beta=[1, 1, 1], h=[1, 1, 1], r=R, T=T, burn=BURN,
                                               batch=BATCH, constants=C, HGI_E=HGI_E, kernel="genkern one-pass, M^r in W", splits=SPLITS,
                                               seeds_independent=SEEDS_INDEP, seeds_same=SEEDS_SAME),
                   independent=A, pairwise_z=zA, same_seed=dict(per_seed={str(s): v for s, v in B.items()}, split_means=same),
                   runs=[dict(split=o["split"], seed=o["seed"], JE=o["JE"], level=o["level"], hgi_path=o["hgi_path"], E_frac=o["E_frac"],
                              checks=o["checks"], n_batch=o["n_batch"], hq=o["hq"], hhat=o["hhat"], hg=o["hg"]) for o in res.values()]),
              open("results/split_invariance.json", "w"), indent=1)
    print(f"\nsaved results/split_invariance.json (commit {commit[:7]}), {time.time()-t0:.0f}s", flush=True)
