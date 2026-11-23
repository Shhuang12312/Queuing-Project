"""Offline pass, rule band c2 and b_max for the seven networks. Writes results/multi_bench.json."""
import json, numpy as np, networks as N
from multiprocessing import Pool
def go(name):
    nw = N.build(name); m, se = nw.benchmark()
    t = nw.tab
    return name, dict(HGI_E=m, se=se, c2=N.rule_c2(nw), b_max=t._get_b_max(), M=len(t.M_set),
                      trivial=bool(t.trivial), theta=nw.net.theta.tolist(), nV=len(nw.V))
if __name__ == "__main__":
    with Pool(2) as p: res = dict(p.map(go, list(N.SPECS)))
    json.dump(res, open("results/multi_bench.json", "w"), indent=1)
    for k, v in res.items(): print(f"{k:26s} HGI_E={v['HGI_E']:.3f}±{v['se']:.3f} c2={v['c2']:.1f} b_max={v['b_max']:.4f}")
