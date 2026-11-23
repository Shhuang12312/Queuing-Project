#!/usr/bin/env python3
"""One run of the toy network (Example 2.2 of Budhiraja & Johnson 2024) under Definition 2.3, at any r.

    python run_toy.py --r 20000 --T 50 --seed 1            # one run, result in results/
    python run_toy.py --estimate --r 100000 --T 20         # events and hours only, nothing is run
    python run_toy.py --bench                              # measures events per second on this machine (about 5 s)
    python run_toy.py --selftest                           # r = 300, T = 250 against the stored run (about 10 s)

The policy is Definition 2.3 as written; the only choices are the free constants c1 < c2, kappa in (0, 1/4).
Engine: sim/genkern.c (exact event-by-event simulation, no time discretization), the one behind Table 2 of the report.
Each run writes results/toy_r<r>_c2<c2>_k<kappa>_T<T>_s<seed>.json with J_E, the three terms of
    J_E - HGI_E = T1 + T2 + T3,   T1 = h.Q - h^(W),  T2 = h^(W) - h^(G),  T3 = h^(G) - HGI_E,
per batch (batches.hq, .hhat, .hg), the flag fraction, the event count and the wall time."""
import argparse, json, math, os, socket, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "sim"))
EVENTS_PER_R2T = 6.0                       # toy: measured 5.996 events per unit of r^2 T
TOY = dict(K=[[1, 0, 1], [0, 1, 1]], alpha=[1, 1, 1], beta=[1, 1, 1], h=[1, 1, 1], eta=[-0.5, -0.5, -0.5])
BENCH = json.load(open(os.path.join(HERE, "benchmark.json")))
HGI_E = BENCH["HGI_E"]


def speed_guess():
    p = os.path.join(HERE, "speed.txt")
    return float(open(p).read().split()[0]) if os.path.exists(p) else 1.2e7


def estimate(r, T):
    ev = EVENTS_PER_R2T * r * r * T
    return ev, ev / speed_guess() / 3600.0


def precision_ratio(r, T):
    """Spacing of double-precision numbers at the final clock value r^2 T, relative to the mean time between events."""
    t_end = r * r * T
    return t_end * 2.220446049250313e-16 / (1.0 / EVENTS_PER_R2T)


def halves(x):
    n = len(x); w1 = [1.0] * n
    if n % 2:
        w1[n // 2] = 0.5
        for i in range(n // 2 + 1, n):
            w1[i] = 0.0
    else:
        for i in range(n // 2, n):
            w1[i] = 0.0
    a = sum(v * w for v, w in zip(x, w1)) / sum(w1); b = sum(v * (1 - w) for v, w in zip(x, w1)) / sum(1 - w for w in w1)
    return a, b


def build():
    import gensim
    return gensim.Net("A toy (Ex. 2.2)", TOY["K"], TOY["alpha"], TOY["beta"], TOY["h"], eta=TOY["eta"])


def run(r, T, c1, c2, kappa, seed, burn, batch):
    import numpy as np
    from flagmodel import g_flag
    nw = build()
    nw.net.check_r(r)                                         # |rho^r - rho| < rho*/4
    L, U = nw.thresholds(r, c1, c2, kappa)
    if not U - L > 1:
        raise SystemExit(f"hysteresis band (c2 - c1) r^kappa = {U - L:.3f} is not wider than one job")
    t0 = time.time()
    o = nw.run(r, T, c1, c2, kappa, seed, burn=burn, batch=batch)
    wall = time.time() - t0
    hq, hh, hg = (np.array(o["batches"][k]) for k in ("hq", "hhat", "hg"))
    t12 = hq - hg; a, b = halves(t12.tolist())
    d = float(nw.tab.x((1, 0, 0))[0])                          # service rate of a local queue below U
    o.update(c=[c1, c2, kappa], burn=burn, wall_s=wall, events_per_s=o["events"] / wall, host=socket.gethostname(),
             HGI_E=HGI_E, T1=float((hq - hh).mean()), T2=float((hh - hg).mean()), T12=float(t12.mean()),
             excess_pct=100 * float(t12.mean()) / HGI_E, T3=float(hg.mean()) - HGI_E,
             T12_half1=a, T12_half2=b, T12_halves_change_pct=100 * (b - a) / a,
             phi=float((hq - hh).mean()) * r / U,
             predicted=dict(flag_drift=0.5 * r * d * g_flag(L, U, 1 - 0.5 / r, d), d=d,
                            excess_pct_line=100 * 2.83 * U / r / HGI_E, T1_fit=(1.006 * U - 43.5) / r),
             p_flag=[float(x) for x in o["p_flag"]], f_below=[float(x) for x in o["f_below"]],
             idle_drift=[float(x) for x in o["idle_drift"]], events=float(o["events"]))
    return o


def line(o):
    return (f"r = {o['r']}  T = {o['T']:g}  (c1, c2, kappa) = ({o['c'][0]:g}, {o['c'][1]:g}, {o['c'][2]:g})  seed {o['seed']}:  "
            f"J_E = {o['JE']:.4f}  T1 = {o['T1']:.5f}  T2 = {o['T2']:.5f}  T1+T2 = {o['excess_pct']:.2f}% of HGI_E  "
            f"(halves {o['T12_halves_change_pct']:+.1f}%)  T3 = {o['T3']:+.3f}  flags on {o['E_frac']:.1e} of the time  "
            f"{o['events']:.3e} events in {o['wall_s'] / 60:.1f} min ({o['events_per_s']:.2e}/s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--r", type=float); ap.add_argument("--T", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--c1", type=float, default=0.1); ap.add_argument("--c2", type=float, default=60.0)
    ap.add_argument("--kappa", type=float, default=0.24)
    ap.add_argument("--batch", type=float, default=None, help="batch length in diffusion time units (default T/10)")
    ap.add_argument("--burn", type=float, default=0.3, help="fraction of the window discarded as burn-in (default 0.3)")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--force", action="store_true", help="rerun even if the result file exists")
    ap.add_argument("--allow-low-precision", action="store_true")
    ap.add_argument("--estimate", action="store_true"); ap.add_argument("--bench", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.bench:
        o = run(1000, 10.0, 0.1, 60.0, 0.24, 1, 0.3, 1.0)
        open(os.path.join(HERE, "speed.txt"), "w").write(f"{o['events_per_s']:.4e}  events per second on {o['host']} ({time.strftime('%Y-%m-%d')})\n")
        print(f"{o['events_per_s']:.3e} events per second on {o['host']} (written to speed.txt)")
        return
    if a.selftest:
        ref = json.load(open(os.path.join(HERE, "reference_r300.json")))
        o = run(300, 250.0, 0.1, 60.0, 0.24, 1, 0.3, 25.0)
        same = o["JE"] == ref["JE"] and o["batches"]["hq"] == ref["batches"]["hq"] and o["events"] == ref["events"]
        ok = abs(o["T1"] - ref["T1"]) < 0.01 and abs(o["T12"] - ref["T12"]) < 0.15 and abs(o["events"] / ref["events"] - 1) < 1e-3
        print(line(o))
        print(f"stored run:  J_E = {ref['JE']:.4f}  T1 = {ref['T1']:.5f}  T1+T2 = {100 * ref['T12'] / HGI_E:.2f}%  {ref['events']:.3e} events")
        print("bit-identical to the stored run: " + ("yes" if same else "no (a different C library rounds log() differently; the run is still exact)"))
        print("SELFTEST " + ("PASSED" if ok else "FAILED"))
        sys.exit(0 if ok else 1)
    if a.r is None:
        ap.error("--r is required")
    r = int(a.r) if float(a.r).is_integer() else a.r
    c1, c2, kappa = a.c1, a.c2, a.kappa
    if not (c1 < c2 and 0 < kappa < 0.25):
        ap.error("Definition 2.3 needs c1 < c2 and 0 < kappa < 1/4")
    ev, hours = estimate(r, a.T); pr = precision_ratio(r, a.T)
    print(f"r = {r}, T = {a.T:g}: about {ev:.2e} events, {hours:.1f} h at {speed_guess():.2e} events/s; "
          f"U = {c2 * r ** kappa:.0f} jobs, L = {c1 * r ** kappa:.2f} jobs", flush=True)
    if pr > 0.01:
        msg = (f"clock resolution at r^2 T = {r * r * a.T:.2e} is {100 * pr:.1f}% of the mean time between events "
               f"(double precision). Use a smaller T")
        if pr > 0.1 or not a.allow_low_precision:
            raise SystemExit("REFUSED: " + msg + (" or pass --allow-low-precision (up to 10%)." if pr <= 0.1 else "."))
        print("WARNING: " + msg, flush=True)
    if a.estimate:
        return
    os.makedirs(a.out, exist_ok=True)
    name = f"toy_r{r}_c2{c2:g}_k{kappa:g}_T{a.T:g}_s{a.seed}.json"
    path = os.path.join(a.out, name)
    if os.path.exists(path) and not a.force:
        print(f"{path} exists; skipped (use --force to rerun)"); return
    o = run(r, a.T, c1, c2, kappa, a.seed, a.burn, a.batch if a.batch else a.T / 10.0)
    json.dump(o, open(path + ".tmp", "w"), indent=1); os.replace(path + ".tmp", path)
    print(line(o)); print("written:", path)


if __name__ == "__main__":
    main()
