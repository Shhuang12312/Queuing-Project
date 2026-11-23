/* toykern3.c -- exact Gillespie simulation of Definition 2.3 on the 2LLN toy.
 *
 * Mirrors rsn_flat: simulation.gillespie_ergodic + policy.
 * GeneralThresholdPolicy.allocation, event for event:
 *   Z_j = 1{q_j < U}, E_j <- 1 if q_j < L, E_j <- 0 if q_j >= U,
 *   b = (1-E) * x(Z) with x(Z) from the offline table (passed in, 8x3),
 *   departure rate of type j = beta_j * b_j * 1{q_j > 0}.
 * K = [[1,0,1],[0,1,1]], beta = 1 (so M = I and W = K q).
 *
 * Accumulators (all exact time integrals, no snapshots):
 *   ergodic window [T_burn, T_end]: h.q, hhat(W)=max(W1,W2), any-flag time,
 *     lost capacity C_i - used_i (used_i counts b_j only if q_j > 0),
 *     lost capacity weighted by W_i, lost capacity by W_i/r bin,
 *     second-half h.q and hhat (stationarity check);
 *   discounted (from t = 0, discount 1 per diffusion unit):
 *     int e^{-tau} h.Q/r dtau and int e^{-tau} hhat(W)/r dtau, tau = t/r^2;
 *   trajectory: hhat(W)/r on an even grid of n_grid times over [0, T_end].
 */
#include <math.h>
#include <stdint.h>

static uint64_t s[4];
static inline uint64_t rotl(const uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static inline uint64_t next(void) {                     /* xoshiro256** */
    const uint64_t result = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
    return result;
}
static inline double unif(void) { return ((next() >> 11) + 0.5) * 0x1.0p-53; } /* (0,1) */
static void seed_rng(uint64_t seed) {                    /* splitmix64 */
    uint64_t z = seed;
    for (int i = 0; i < 4; i++) {
        z += 0x9e3779b97f4a7c15ULL; uint64_t x = z;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL; x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        s[i] = x ^ (x >> 31);
    }
}

#define NBIN 6
#define XOFF (13 + 2 * NBIN)
static const double BIN_EDGES[NBIN] = {0.05, 0.25, 0.5, 1.0, 2.0, 1e300};

/* out layout (doubles):
 0 cost  1 hhat  2 Eon  3 lost1  4 lost2  5 lostW1  6 lostW2
 7 cost_2nd 8 hhat_2nd  9 disc_cost 10 disc_hhat 11 n_events 12 window
 13..13+2*NBIN-1  lost by W/r bin: resource 1 bins then resource 2 bins */
void simulate(const double *xtab, double L, double U, const double *alpha_r,
              const double *h, double r, double T_end, double T_burn,
              uint64_t seed, int n_grid, double *grid_hhat, double *out)
{
    seed_rng(seed);
    double q[3] = {0, 0, 0}, E[3] = {0, 0, 0};
    double t = 0.0, r2 = r * r, T_mid = 0.5 * (T_burn + T_end);
    for (int k = 0; k < XOFF + 6; k++) out[k] = 0.0;
    double Y1 = 0, Y2 = 0, m1 = 0, m2 = 0;
    double dt_grid = T_end / (double)n_grid;
    int gi = 0;
    const double tol = 1e-12, C = 2.0;
    double nev = 0.0;

    while (t < T_end) {
        int idx = 0;
        for (int j = 0; j < 3; j++) {
            if (q[j] < L - tol) E[j] = 1.0;
            if (q[j] >= U - tol) E[j] = 0.0;
            if (q[j] < U - tol) idx |= (1 << j);
        }
        const double *x = xtab + 3 * idx;
        double b[3], dep[3], tot = alpha_r[0] + alpha_r[1] + alpha_r[2];
        for (int j = 0; j < 3; j++) {
            b[j] = (1.0 - E[j]) * x[j];
            dep[j] = (q[j] > 0) ? b[j] : 0.0;           /* beta_r = 1 */
            tot += dep[j];
        }
        double t_next = t - log(unif()) / tot;

        double W1 = q[0] + q[2], W2 = q[1] + q[2];
        double G1 = Y1 - m1, G2 = Y2 - m2, hg = G1 > G2 ? G1 : G2;
        double hh = W1 > W2 ? W1 : W2;
        double hq = h[0] * q[0] + h[1] * q[1] + h[2] * q[2];

        while (gi < n_grid && gi * dt_grid < t_next) { grid_hhat[gi] = hh / r; gi++; }

        double a = t > T_burn ? t : T_burn, e = t_next < T_end ? t_next : T_end;
        if (e > a) {
            double ov = e - a;
            out[0] += ov * hq; out[1] += ov * hh;
            if (E[0] + E[1] + E[2] > 0) out[2] += ov;
            double used1 = dep[0] + dep[2], used2 = dep[1] + dep[2];
            double l1 = C - used1, l2 = C - used2;
            out[3] += ov * l1; out[4] += ov * l2;
            out[5] += ov * l1 * W1; out[6] += ov * l2 * W2;
            int b1 = 0, b2 = 0;
            while (W1 / r >= BIN_EDGES[b1]) b1++;
            while (W2 / r >= BIN_EDGES[b2]) b2++;
            out[13 + b1] += ov * l1; out[13 + NBIN + b2] += ov * l2;
            for (int j = 0; j < 2; j++) {
                if (E[j] > 0) out[XOFF + 2 + j] += ov;
                else if (q[j] < U - tol) out[XOFF + j] += ov;
            }
            out[XOFF + 4] += ov * hg;
            double a2 = a > T_mid ? a : T_mid;
            if (e > a2) { out[7] += (e - a2) * hq; out[8] += (e - a2) * hh; }
        }
        /* discounted integrals from 0, tau = t / r^2, discount rate 1 */
        double tend = t_next < T_end ? t_next : T_end;
        double dsc = exp(-t / r2) - exp(-tend / r2);    /* = int e^{-tau} dtau */
        out[9] += dsc * hq / r; out[10] += dsc * hh / r;

        {
            double hold = (t_next < T_end ? t_next : T_end) - t;
            Y1 -= hold * (C - dep[0] - dep[2]); Y2 -= hold * (C - dep[1] - dep[2]);
            if (Y1 < m1) m1 = Y1;
            if (Y2 < m2) m2 = Y2;
        }
        t = t_next;
        if (t >= T_end) break;
        nev += 1.0;
        double u = unif() * tot, c = 0.0;
        int k;
        for (k = 0; k < 3; k++) { c += alpha_r[k]; if (u < c) { q[k] += 1.0; if (k != 1) Y1 += 1.0; if (k != 0) Y2 += 1.0; break; } }
        if (k == 3) {
            for (k = 0; k < 3; k++) { c += dep[k]; if (u < c) { if (q[k] > 0) { q[k] -= 1.0; if (k != 1) Y1 -= 1.0; if (k != 0) Y2 -= 1.0; if (Y1 < m1) m1 = Y1; if (Y2 < m2) m2 = Y2; } break; } }
        }
    }
    while (gi < n_grid) { grid_hhat[gi] = (q[0] + q[2] > q[1] + q[2] ? q[0] + q[2] : q[1] + q[2]) / r; gi++; }
    out[11] = nev;
    out[12] = T_end - T_burn;
}
