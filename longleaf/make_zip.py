#!/usr/bin/env python3
"""Builds dist/hgi_toy_longleaf.zip from the repository (run from anywhere):  python longleaf/make_zip.py

The zip is self-contained: the scripts of this directory, the three rsn_flat modules and the three sim files the toy
run needs, the benchmark value, one stored run for the self-test and the rows of the report's Table 2."""
import json, os, sys, zipfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); RES = os.path.join(ROOT, "sim", "results")
TOP = "hgi_toy_longleaf"
TOY = "A toy (Ex. 2.2)"
bn = json.load(open(os.path.join(RES, "multi_bench_crn.json")))["networks"][TOY]
bench = dict(HGI_E=bn["HGI_E"], se=bn["se"], hw95=bn["hw95"], n_seeds=bn["n_seeds"],
             source="sim/results/multi_bench_crn.json: sqrt(dt) extrapolation, both legs on one Brownian path per seed")
pc = [o for o in json.load(open(os.path.join(RES, "paper_cells.json"))) if o["net"] == TOY and o["cap"] == "uniform" and o["c"][1] == 60.0]
o = [x for x in pc if x["r"] == 300 and x["seed"] == 1][0]
hq, hh, hg = (np.array(o["batches"][k]) for k in ("hq", "hhat", "hg"))
ref300 = dict(r=300, T=o["T"], c=o["c"], seed=1, JE=o["JE"], events=o["events"], batches=o["batches"], T1=float((hq - hh).mean()), T12=float((hq - hg).mean()),
              source="sim/results/paper_cells.json")
from scipy.stats import t as tdist
hw = lambda x: float(tdist.ppf(0.975, len(x) - 1) * np.std(x, ddof=1) / np.sqrt(len(x)))
tab2 = []
for r in (300, 1000, 2000, 4000, 10000):
    g = [x for x in pc if x["r"] == r]
    hq = np.concatenate([x["batches"]["hq"] for x in g]); hh = np.concatenate([x["batches"]["hhat"] for x in g]); hg = np.concatenate([x["batches"]["hg"] for x in g])
    tab2.append(dict(r=r, T=g[0]["T"], seeds=len(g), JE=float(hq.mean()), JE_hw=hw(hq), T1=float((hq - hh).mean()), T2=float((hh - hg).mean()), T2_hw=hw(hh - hg),
                     excess_pct=100 * float((hq - hg).mean()) / bench["HGI_E"], excess_hw=100 * hw(hq - hg) / bench["HGI_E"], phi=float((hq - hh).mean()) * r / g[0]["U"]))
files = {f"{TOP}/{n}": os.path.join(HERE, n) for n in ("setup.sh", "run_toy.py", "make_jobs.py", "collect.py", "submit.sbatch")}
files.update({f"{TOP}/rsn_flat/{n}": os.path.join(ROOT, "rsn_flat", n) for n in ("rsn.py", "offline.py", "simulation.py")})
files.update({f"{TOP}/sim/{n}": os.path.join(ROOT, "sim", n) for n in ("gensim.py", "genkern.c", "flagmodel.py")})
os.makedirs(os.path.join(ROOT, "dist"), exist_ok=True)
out = os.path.join(ROOT, "dist", "hgi_toy_longleaf.zip")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for arc, src in sorted(files.items()):
        info = zipfile.ZipInfo.from_file(src, arc)
        if arc.endswith((".sh", ".sbatch")) or arc.endswith(("run_toy.py", "make_jobs.py", "collect.py")):
            info.external_attr = (0o755 << 16)
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, open(src, "rb").read())
    for arc, obj in ((f"{TOP}/benchmark.json", bench), (f"{TOP}/reference_r300.json", ref300), (f"{TOP}/reference_table2.json", tab2)):
        z.writestr(arc, json.dumps(obj, indent=1))
print(f"wrote {out} ({os.path.getsize(out) / 1024:.0f} kB, {len(files) + 3} files)")
