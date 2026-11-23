"""Definition 2.3 with a deeper hysteresis band: c1=0.1, kappa=0.24, c2 in {60,90,120}
(all admissible, fixed in r). Tests the flag-model prediction that the floor
vanishes once mu*U >> log r, and whether J_E then reaches HGI_E."""
import json, time, warnings
from multiprocessing import Pool
import toysim3 as ts

warnings.filterwarnings("ignore")
JOBS = ([((0.1, c2, 0.24), 1000, 40, s) for c2 in (60, 90, 120) for s in range(1, 5)]
        + [((0.1, c2, 0.24), 4000, 40, s) for c2 in (60, 90, 120) for s in range(1, 3)])


def go(job):
    c, r, T, s = job
    o = ts.run(r, T, c, s, burn=0.3, n_grid=200)
    o["c2"] = c[1]
    return o


if __name__ == "__main__":
    t0 = time.time(); res = []
    with Pool(2) as p:
        for o in p.imap_unordered(go, JOBS):
            o.pop("grid"); res.append(o); json.dump(res, open("results/c2scan.json", "w"))
            print(f"c2={o['c2']:4.0f} r={o['r']:5d} s={o['seed']} JE={o['JE']:6.3f} level={o['level']:6.3f} "
                  f"gap={o['gap']:.4f} 1st/2nd={o['JE_1st']:.2f}/{o['JE_2nd']:.2f} "
                  f"idle={o['idle_drift'][0]:.3f},{o['idle_drift'][1]:.3f} "
                  f"pflag={o['p_flag'][0]:.2e},{o['p_flag'][1]:.2e} f={o['f_below'][0]:.3f},{o['f_below'][1]:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print("ALL DONE", flush=True)
