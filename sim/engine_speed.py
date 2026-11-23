"""Engine speeds on this machine. Run alone (nothing else on the machine).
  * toy kernel (toykern3): r = 1000, T = 40, 3 repeats;
  * general kernel (genkern) on each of the seven networks: r = 1000, T = 10, 2 repeats;
  * the r = 1e4, T = 20 toy run of Section 4 (toy kernel, seed 1): wall time and event count.
Writes results/engine_speed.json."""
import json, os, platform, sys, time, warnings
import numpy as np
import toysim3, networks as N
from common import log_run

warnings.filterwarnings("ignore")
if __name__ == "__main__":
    t00 = time.time(); B = json.load(open("results/multi_bench.json")); out = dict(machine=dict(platform=platform.platform(), cpus=os.cpu_count(), load_before=os.getloadavg()), toy=[], general=[])
    for rep in range(3):
        t0 = time.time(); o = toysim3.run(1000, 40, (0.1, 60.0, 0.24), seed=1 + rep); dt = time.time() - t0
        out["toy"].append(dict(seed=1 + rep, events=o["events"], wall_s=dt, events_per_s=o["events"] / dt)); print(f"toykern3 r=1000 T=40 seed {1+rep}: {o['events']:.3e} events, {dt:.2f} s, {o['events']/dt:.3e} ev/s", flush=True)
    for name in N.SPECS:
        nw = N.build(name)
        for rep in range(2):
            t0 = time.time(); o = nw.run(1000, 10, 0.1, B[name]["c2"], 0.24, 1 + rep); dt = time.time() - t0
            out["general"].append(dict(net=name, J=nw.net.J, I=nw.net.I, seed=1 + rep, events=o["events"], wall_s=dt, events_per_s=o["events"] / dt))
            print(f"genkern {name[:10]:10s} (I={nw.net.I}, J={nw.net.J}) seed {1+rep}: {o['events']:.3e} events, {dt:.2f} s, {o['events']/dt:.3e} ev/s", flush=True)
    if "--no-big" not in sys.argv:
        t0 = time.time(); o = toysim3.run(10000, 20, (0.1, 60.0, 0.24), seed=1); dt = time.time() - t0
        out["toy_r1e4_T20"] = dict(seed=1, events=o["events"], wall_s=dt, minutes=dt / 60, events_per_s=o["events"] / dt, JE=o["JE"], hgi_path=o["hgi_path"])
        print(f"toykern3 r=10000 T=20 seed 1: {o['events']:.3e} events, {dt:.1f} s = {dt/60:.2f} min, {o['events']/dt:.3e} ev/s, J_E = {o['JE']:.4f}", flush=True)
    out["machine"]["load_after"] = os.getloadavg()
    ts = np.array([x["events_per_s"] for x in out["toy"]]); print(f"toy kernel: {ts.mean():.3e} ev/s (min {ts.min():.3e}, max {ts.max():.3e})")
    for name in N.SPECS:
        g = np.array([x["events_per_s"] for x in out["general"] if x["net"] == name]); print(f"general kernel, {name[:1]}: {g.mean():.3e} ev/s")
    json.dump(out, open("results/engine_speed.json", "w"), indent=1)
    dt_all = time.time() - t00
    log_run("python engine_speed.py", "toy 1-3; general 1-2 per network; r = 1e4 run seed 1", dt_all, "sim/results/engine_speed.json, engine_speed.log", "run alone on the machine (single process)")
    print(f"{dt_all:.0f}s", flush=True)
