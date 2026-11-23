"""queue_trace: what the hysteresis flag does to one local queue of the toy network (Definition 2.3, r = 1000, seed 1).

The compiled kernels integrate costs but do not record a path, so this script carries its
own recording loop: a line-by-line Python copy of the event loop of sim/genkern.c (same xoshiro256** generator and
seeding, same order of operations, same offline tables through gensim.Net). With the same seed it follows the SAME
random path as the compiled kernel, so the check is exact, not statistical: over the recorded window its J_E, flag
fractions and below-U fractions must equal those of gensim.Net.run(r, T, ..., seed, burn=...) to rounding error.

Run (from sim/):   python make_trace_fig.py --draw   # draw from the stored results/queue_trace_data.npz
                   python make_trace_fig.py          # simulate both bands first (about 2 min, two processes), check, draw
For (c1, c2, kappa) = (0.1, 30, 0.24) and (0.1, 60, 0.24): the first 5 diffusion time units are discarded, then the
queue of job type 1 (local to resource 1), in jobs, and its flag are recorded over the next 2 diffusion time units
(2e6 units of physical time). Stored in results/queue_trace_data.npz, per band (suffix _30, _60):
  grid_t (diffusion units from the start of the window, 20 000 cells), grid_q (queue at the start of each cell),
  grid_min (minimum over the cell), flag_on / flag_off (exact switch times, same units), U, L, check_* (recorder and
  compiled-kernel values), and for the first flag episode of the c2 = 30 window: zoom_t (physical time from the switch-on),
  zoom_q (queue after each event of that type), zoom_off (switch-off time).
Writes queue_trace.png (300 dpi) and .pdf to results/final/, or to DIR with --out DIR; without --draw also
results/queue_trace_data.npz."""
import json, math, os, sys, time
from array import array
from multiprocessing import Pool
import numpy as np

R, SEED, T_DISCARD, T_WINDOW, C1, KAPPA, J0, NCELL = 1000, 1, 5.0, 2.0, 0.1, 0.24, 0, 20000
BANDS = (30.0, 60.0)
M64 = (1 << 64) - 1


def record(c2):
    """Python copy of genkern.c's event loop for the toy; returns the recorded path of queue J0 and the window integrals."""
    import gensim
    nw = gensim.Net("A toy (Ex. 2.2)", [[1, 0, 1], [0, 1, 1]], [1, 1, 1], [1, 1, 1], [1, 1, 1], eta=[-0.5] * 3)
    n = nw.net; J = n.J
    L, U = nw.thresholds(R, C1, c2, KAPPA)
    T = T_DISCARD + T_WINDOW; burn = T_DISCARD / T
    T_end = R * R * T; T_burn = burn * T_end                       # exactly as in gensim.Net.run
    alpha_r = [float(v) for v in n.alphas_r(R)]; beta_r = [float(v) for v in n.betas_r(R)]
    xtab = [[float(v) for v in row] for row in nw.xtab]; h = [float(v) for v in n.h]
    tol = 1e-12; Lt, Ut = L - tol, U - tol
    # xoshiro256** seeded by splitmix64 (genkern.c lines 40-55)
    s = []; z = SEED
    for _ in range(4):
        z = (z + 0x9e3779b97f4a7c15) & M64; x = z
        x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & M64; x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & M64
        s.append(x ^ (x >> 31))
    s0, s1, s2, s3 = s
    q = [0.0] * J; E = [0.0] * J; rate = [0.0] * J
    t = 0.0; nev = 0
    out_hq = 0.0; flag_t = [0.0] * J; below_t = [0.0] * J; any_t = 0.0; q0_t = 0.0
    rec_from = T_burn - 12.0 * U - 100.0
    ev_t = array("d"); ev_q = array("d"); on_t = array("d"); off_t = array("d")
    e_prev = 0.0; q_rec_start = None
    log = math.log
    while t < T_end:
        idx = 0
        for j in range(J):
            qj = q[j]
            if qj < Lt:
                E[j] = 1.0
            if qj >= Ut:
                E[j] = 0.0
            if qj < Ut:
                idx |= (1 << j)
        if E[J0] != e_prev:
            (on_t if E[J0] > 0 else off_t).append(t); e_prev = E[J0]
        x = xtab[idx]
        tot = 0.0
        for j in range(J):
            tot += alpha_r[j]
        for j in range(J):
            b = (1.0 - E[j]) * x[j]
            usedj = b if q[j] > 0 else 0.0
            rate[j] = beta_r[j] * usedj
            tot += rate[j]
        r1 = ((((s1 * 5) & M64) << 7 | ((s1 * 5) & M64) >> 57) & M64) * 9 & M64; tt = (s1 << 17) & M64
        s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= tt; s3 = ((s3 << 45) | (s3 >> 19)) & M64
        t_next = t - log(((r1 >> 11) + 0.5) * 1.1102230246251565e-16) / tot
        a = t if t > T_burn else T_burn; e = t_next if t_next < T_end else T_end
        if e > a:
            ov = e - a; hq = 0.0; anyE = 0.0
            for j in range(J):
                hq += h[j] * q[j]; anyE += E[j]
                if E[j] > 0:
                    flag_t[j] += ov
                elif q[j] < Ut:
                    below_t[j] += ov
            out_hq += ov * hq; q0_t += ov * q[J0]
            if anyE > 0:
                any_t += ov
        t = t_next
        if t >= T_end:
            break
        nev += 1
        r2 = ((((s1 * 5) & M64) << 7 | ((s1 * 5) & M64) >> 57) & M64) * 9 & M64; tt = (s1 << 17) & M64
        s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= tt; s3 = ((s3 << 45) | (s3 >> 19)) & M64
        u = ((r2 >> 11) + 0.5) * 1.1102230246251565e-16 * tot; c = 0.0; k = -1
        for j in range(J):
            c += alpha_r[j]
            if u < c:
                q[j] += 1.0; k = j; break
        if k < 0:
            for j in range(J):
                c += rate[j]
                if u < c:
                    if q[j] > 0:
                        q[j] -= 1.0; k = j
                    break
        if t >= rec_from:
            if q_rec_start is None:
                q_rec_start = q[J0]; ev_t.append(t); ev_q.append(q[J0])
            elif k == J0:
                ev_t.append(t); ev_q.append(q[J0])
    win = T_end - T_burn
    return dict(c2=c2, L=L, U=U, T_end=T_end, T_burn=T_burn, events=nev, ev_t=np.frombuffer(ev_t, float).copy(), ev_q=np.frombuffer(ev_q, float).copy(),
                on=np.frombuffer(on_t, float).copy(), off=np.frombuffer(off_t, float).copy(),
                JE=out_hq / win / R, p_flag=[v / win for v in flag_t], f_below=[v / win for v in below_t], E_frac=any_t / win, mean_q0=q0_t / win)


def production(c2):
    import gensim
    nw = gensim.Net("A toy (Ex. 2.2)", [[1, 0, 1], [0, 1, 1]], [1, 1, 1], [1, 1, 1], [1, 1, 1], eta=[-0.5] * 3)
    T = T_DISCARD + T_WINDOW
    return nw.run(R, T, C1, c2, KAPPA, SEED, burn=T_DISCARD / T)


def reduce(rec):
    """Grid of NCELL cells over the window: queue at the start of each cell and its minimum over the cell; flag episodes."""
    t0, t1 = rec["T_burn"], rec["T_end"]; et, eq = rec["ev_t"], rec["ev_q"]
    edges = np.linspace(t0, t1, NCELL + 1)
    idx = np.searchsorted(et, edges, side="right") - 1               # last event at or before each edge
    start = eq[idx[:-1]]
    mins = start.copy()
    inside = (et > t0) & (et <= t1); cell = np.minimum(((et[inside] - t0) / (t1 - t0) * NCELL).astype(int), NCELL - 1)
    np.minimum.at(mins, cell, eq[inside])
    on = rec["on"][(rec["on"] >= t0) & (rec["on"] < t1)]
    off = np.array([rec["off"][rec["off"] > a][0] if np.any(rec["off"] > a) else t1 for a in on])
    in_window = float(np.sum(np.minimum(off, t1) - on)) + (float(min(rec["off"][rec["off"] > t0][0], t1) - t0) if (len(rec["on"][rec["on"] < t0]) > len(rec["off"][rec["off"] <= t0])) else 0.0)
    return dict(grid_t=(edges[:-1] - t0) / R / R, grid_q=start, grid_min=mins, flag_on=(on - t0) / R / R, flag_off=(off - t0) / R / R,
                on_phys=on, off_phys=off, flag_fraction_from_episodes=in_window / (t1 - t0))


def simulate():
    t0 = time.time()
    with Pool(2) as p:
        recs = p.map(record, BANDS)
    wall = time.time() - t0
    out = dict(r=R, seed=SEED, discard=T_DISCARD, window=T_WINDOW, job_type=J0 + 1, wall_s=wall)
    ok = True
    for rec in recs:
        tag = f"_{rec['c2']:.0f}"; red = reduce(rec); prod = production(rec["c2"])
        chk = dict(JE=(rec["JE"], prod["JE"]), p_flag=(rec["p_flag"][J0], prod["p_flag"][J0]), f_below=(rec["f_below"][J0], prod["f_below"][J0]),
                   E_frac=(rec["E_frac"], prod["E_frac"]), events=(float(rec["events"]), float(prod["events"])))
        same = all(abs(a - b) <= 1e-9 * max(1.0, abs(b)) for a, b in chk.values())
        ok &= same
        print(f"c2 = {rec['c2']:.0f}: U = {rec['U']:.2f}, L = {rec['L']:.3f}; {rec['events']} events; window = [{T_DISCARD:g}, {T_DISCARD + T_WINDOW:g}] diffusion units")
        for k, (a, b) in chk.items():
            print(f"   {k:8s} recorder {a!r:24}  production kernel {b!r:24}  difference {a - b:+.2e}")
        print(f"   mean of queue {J0 + 1} over the window: {rec['mean_q0']:.3f} jobs; flag episodes starting in the window: {len(red['flag_on'])}; "
              f"flag fraction {rec['p_flag'][J0]:.3e} (from the episode times: {red['flag_fraction_from_episodes']:.3e});  check {'PASSED' if same else 'FAILED'}")
        for k in ("grid_t", "grid_q", "grid_min", "flag_on", "flag_off"):
            out[k + tag] = red[k]
        out["U" + tag] = rec["U"]; out["L" + tag] = rec["L"]; out["mean_q" + tag] = rec["mean_q0"]; out["events" + tag] = rec["events"]
        for k, (a, b) in chk.items():
            out[f"check_{k}{tag}"] = np.array([a, b])
        if rec["c2"] == BANDS[0] and len(red["on_phys"]):
            on, off = red["on_phys"][0], red["off_phys"][0]; U = rec["U"]
            m = (rec["ev_t"] >= on - 10 * U) & (rec["ev_t"] <= off + 5 * U)
            i0 = max(int(np.argmax(m)) - 1, 0)
            out["zoom_t"] = np.concatenate([[-10 * U], rec["ev_t"][m] - on]); out["zoom_q"] = np.concatenate([[rec["ev_q"][i0]], rec["ev_q"][m]])
            out["zoom_off"] = off - on; out["zoom_on_diffusion"] = (on - rec["T_burn"]) / R / R
    if not ok:
        print("STOPPED: the recorder does not reproduce the production kernel; nothing was written."); sys.exit(1)
    np.savez_compressed("results/queue_trace_data.npz", **out)
    print(f"saved results/queue_trace_data.npz ({wall:.0f} s)")


def draw():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load("results/queue_trace_data.npz")
    INK, INK2, SURF, GREY, VERM = "#0b0b0b", "#52514e", "#ffffff", "#7a7a7a", "#D55E00"
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.3), facecolor=SURF, gridspec_kw=dict(width_ratios=[1.25, 0.95, 1.25]))
    ytop = 1.15 * float(d["U_60"])
    BOX = dict(facecolor=SURF, edgecolor="none", alpha=0.85, pad=1.2)
    for ax in axes:
        ax.set_facecolor(SURF); ax.grid(True, alpha=0.25, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_linewidth(0.6)
        ax.tick_params(colors=INK2, labelsize=7, width=0.6, length=3)
    for ax, tag, lab in ((axes[0], "_30", "(a)"), (axes[2], "_60", "(c)")):
        t, q, mn, U = d["grid_t" + tag], d["grid_q" + tag], d["grid_min" + tag], float(d["U" + tag])
        ax.fill_between(t, mn, q, color=GREY, lw=0, alpha=0.55, zorder=2); ax.plot(t, q, color=GREY, lw=0.35, zorder=2)
        ax.axhline(U, color=INK, ls="--", lw=0.8, zorder=3)
        ax.text(1.97, U + 0.03 * ytop, "U", color=INK, fontsize=7, ha="right", va="bottom", bbox=BOX, zorder=6)
        on = d["flag_on" + tag]
        if len(on):
            ax.plot(on, np.full(len(on), ytop), "|", color=VERM, ms=7, mew=1.0, clip_on=False, zorder=5)
        ax.set_xlim(0, T_WINDOW); ax.set_ylim(0, ytop); ax.set_xlabel("time (diffusion units)", color=INK2)
        ax.text(0.035, 0.955, lab, transform=ax.transAxes, ha="left", va="top", fontsize=8, color=INK, bbox=BOX, zorder=6)
    axes[0].set_ylabel("local queue (jobs)", color=INK2)
    ax = axes[1]; U = float(d["U_30"])
    if "zoom_t" in d:
        off = float(d["zoom_off"])
        ax.axvspan(0.0, off, color=VERM, alpha=0.18, lw=0, zorder=1)
        ax.step(np.append(d["zoom_t"], off + 5 * U), np.append(d["zoom_q"], d["zoom_q"][-1]), where="post", color=GREY, lw=0.6, zorder=2)
        ax.set_xlim(-10 * U, off + 5 * U)
    ax.axhline(U, color=INK, ls="--", lw=0.8, zorder=3)
    ax.text(-9.6 * U, U + 0.03 * 1.3 * U, "U", color=INK, fontsize=7, ha="left", va="bottom", bbox=BOX, zorder=6)
    ax.set_ylim(0, 1.3 * U); ax.set_xlabel("time since flag on (time units)", color=INK2)
    ax.text(0.035, 0.955, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=8, color=INK, bbox=BOX, zorder=6)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.19, top=0.955, wspace=0.27)
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "results/final"
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"queue_trace.{ext}"), dpi=300, facecolor=SURF)
    print(f"saved {out}/queue_trace.png, {out}/queue_trace.pdf")


if __name__ == "__main__":
    if "--draw" not in sys.argv:
        simulate()
    draw()
