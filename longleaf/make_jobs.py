#!/usr/bin/env python3
"""Writes jobs.txt (one run per line) and prints the sbatch command with a time limit.

    python make_jobs.py --r 20000 40000 100000 --T 50 --seeds 1-4
    python make_jobs.py --r 100000 --T 20 --seeds 1-8 --c2 60 --append

Each line of jobs.txt is the argument list of one `python run_toy.py ...` call; array task k runs line k.
The expected relative standard error of T1+T2 over n seeds is about 0.62/sqrt(T n): the across-seed spread of the
report's eight-seed runs at r = 1000 and 2000 (T = 250, c2 = 60), scaled to the window length (a planning figure)."""
import argparse, math, os
from run_toy import estimate, precision_ratio, speed_guess, HERE


def seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--r", type=float, nargs="+", required=True); ap.add_argument("--T", type=float, default=50.0)
ap.add_argument("--seeds", default="1-4"); ap.add_argument("--c1", type=float, default=0.1)
ap.add_argument("--c2", type=float, nargs="+", default=[60.0]); ap.add_argument("--kappa", type=float, default=0.24)
ap.add_argument("--append", action="store_true", help="add to jobs.txt instead of replacing it")
ap.add_argument("--safety", type=float, default=1.6, help="time limit = safety x the estimate of the longest run")
a = ap.parse_args()
path = os.path.join(HERE, "jobs.txt")
old = open(path).read().splitlines() if a.append and os.path.exists(path) else []
new, worst = [], 0.0
print(f"speed assumed: {speed_guess():.2e} events/s (speed.txt if present; `python run_toy.py --bench` measures it)")
ns = len(seeds(a.seeds))
print(f"{'r':>9s} {'T':>5s} {'c2':>5s} {'events per run':>15s} {'hours per run':>14s} {'U (jobs)':>9s} {'expected relative SE of T1+T2':>30s}")
for r in a.r:
    r = int(r) if float(r).is_integer() else r
    for c2 in a.c2:
        ev, h = estimate(r, a.T); worst = max(worst, h)
        note = "  <-- clock resolution too coarse: lower T" if precision_ratio(r, a.T) > 0.01 else ""
        print(f"{r:9g} {a.T:5g} {c2:5g} {ev:15.2e} {h:14.2f} {c2 * r ** a.kappa:9.0f} {100 * 0.62 / (a.T * ns) ** 0.5:21.1f}% ({ns} seeds){note}")
        for s in seeds(a.seeds):
            new.append(f"--r {r} --T {a.T:g} --c1 {a.c1:g} --c2 {c2:g} --kappa {a.kappa:g} --seed {s}")
open(path, "w").write("\n".join(old + new) + "\n")
lim = max(1.0, worst * a.safety); d, hh = int(lim // 24), int(math.ceil(lim % 24))
if hh == 24:
    d, hh = d + 1, 0
n0, n1 = len(old) + 1, len(old) + len(new)
print(f"\njobs.txt: {len(old) + len(new)} lines ({len(new)} new, lines {n0}-{n1}); longest run about {worst:.1f} h")
print(f"submit with:\n    sbatch --array={n0}-{n1} --time={d}-{hh:02d}:00:00 submit.sbatch")
if lim > 11 * 24:
    print("WARNING: longer than 11 days (the usual limit of the general partition); lower T or r.")
