"""Non-exponential toy: H2 interarrivals (SCV 2) and Erlang-2 service requirements
(SCV 1/2), Definition 2.3 with (0.1, 60, 0.24), r = 1000/2000/4000; HGI_E computed for
Sigma = K M (Sigma^u + Sigma^v R) M^T K^T = 2.5 K K^T. Writes results/gi_runs.json."""
import json, time, warnings
from multiprocessing import Pool
from gensim_gi import GINet

warnings.filterwarnings("ignore")
K2 = [[1, 0, 1], [0, 1, 1]]
SPEC = dict(arrivals=("h2", 2.0), services=("erlang", 2))
C = (0.1, 60.0, 0.24)
JOBS = [(4000, 30, s) for s in (1, 2)] + [(2000, 40, s) for s in (1, 2)] + [(1000, 40, s) for s in (1, 2, 3)]
_nw = None


def build():
    return GINet("toy H2(2) arrivals / Erlang-2 services", K2, [1, 1, 1], [1, 1, 1], [1, 1, 1], [-0.5] * 3, **SPEC)


def go(job):
    global _nw
    if _nw is None:
        _nw = build()
    r, T, s = job
    return _nw.run(r, T, *C, s)


if __name__ == "__main__":
    t0 = time.time()
    nw = build()
    print("Sigma =", nw.net.Sigma.tolist(), " scv_u", nw.net.scv_u.tolist(), " scv_v", nw.net.scv_v.tolist())
    m, se = nw.benchmark()
    print(f"HGI_E(Sigma = 2.5 K K^T) = {m:.4f} +- {se:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    out = dict(spec=dict(arrivals=list(SPEC["arrivals"]), services=list(SPEC["services"])), Sigma=nw.net.Sigma.tolist(),
               HGI_E=m, se=se, c=list(C), runs=[])
    with Pool(2) as p:
        for o in p.imap_unordered(go, JOBS):
            out["runs"].append(o); json.dump(out, open("results/gi_runs.json", "w"), indent=1)
            print(f"r={o['r']:5d} T={o['T']} s={o['seed']} JE={o['JE']:7.3f} sameHGI={o['hgi_path']:6.3f} "
                  f"excess={100*(o['JE']-o['hgi_path'])/m:6.1f}% gap={o['gap']:.3f} "
                  f"1st/2nd={o['JE_1st']:.2f}/{o['JE_2nd']:.2f} Efrac={o['E_frac']:.1e} "
                  f"{o['events']:.2e} ev ({time.time()-t0:.0f}s)", flush=True)
    print("ALL DONE", flush=True)
