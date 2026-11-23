/* genkern.c -- exact Gillespie simulation of Definition 2.3 on a general small RSN.
 *
 * Same event logic as rsn_flat (simulation.gillespie_ergodic + policy.allocation):
 *   Z_j = 1{q_j < U}; E_j <- 1 if q_j < L, E_j <- 0 if q_j >= U;
 *   b = (1-E) * x(Z) from the offline table (2^J x J, bit j of the row index = z_j);
 *   departure rate of type j = beta^r_j * b_j * 1{q_j > 0}.
 * Workload (unscaled) W = K M q, M = diag(1/beta). h_hat(W) = max_k V_k . W over the
 * vertices V of the dual polyhedron {y : (KM)^T y <= h} (exact for W >= 0).
 * Same-path HGI: Y_i = W_i - int (C_i - used_i) dt, used_i = sum_j K_ij b_j 1{q_j>0};
 * G_i = Y_i - min(0, inf Y_i) (exact 1-d Skorokhod map); accumulate h_hat(G).
 * Arithmetic order mirrors toykern3.c so the toy reproduces it bit for bit.
 *
 * out: 0 cost 1 hhat 2 hgipath 3 Eon 4 cost_2nd 5 hhat_2nd 6 n_events 7 window
 *      8..8+J-1 flagged time per j; 8+J..8+2J-1 below-U unflagged time per j;
 *      8+2J..8+2J+I-1 lost capacity per resource
 *
 * Batched integrals: if n_batch > 0, bout[3*k + 0/1/2] receive the integrals of
 * h.q, hhat(W), hhat(G) over the batch [T_burn + k*batch_len, T_burn + (k+1)*batch_len), k < n_batch,
 * accumulated in the same pass with the same left-endpoint rule as out[0..2]. The totals out[0..2]
 * do not depend on n_batch (bit for bit).
 *
 * Pathwise assertions, checked at EVERY event (cost is negligible next to the hhat dot products):
 *   (i)  h.q - hhat(W) >= -tol            always (hhat is the minimum over {q' : K M q' = W});
 *   (ii) G_i <= W_i + tol for every i      always (eq. (12): Gamma(w + X) <= W, the regulator is minimal);
 *        and, if check_hg != 0 (caller sets it when hhat is monotone, i.e. all dual vertices >= 0),
 *        hhat(G) <= hhat(W) + tol.
 * On a violation the loop stops, diag[0..3] = (code, t, lhs, rhs) with code 1 / 2 / 3 for the three
 * checks, and the return value is the code; 0 = clean run. invbeta is 1/beta^r (M^r), set by the caller.
 *
 * Free-process grid (increment test): if n_grid > 0, ygrid[k*I + i] = Y_i(T_burn + k*grid_dt), k < n_grid,
 * Y = W - int (C - used) dt = r (w^ + X^) in unscaled units, evaluated exactly inside the holding interval
 * (Y is linear there with slope -(C_i - used_i)).
 */
#include <math.h>
#include <stdint.h>

#define MAXJ 20
#define MAXI 8

static uint64_t s[4];
static inline uint64_t rotl(const uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static inline uint64_t next(void) {
    const uint64_t result = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
    return result;
}
static inline double unif(void) { return ((next() >> 11) + 0.5) * 0x1.0p-53; }
static void seed_rng(uint64_t seed) {
    uint64_t z = seed;
    for (int i = 0; i < 4; i++) {
        z += 0x9e3779b97f4a7c15ULL; uint64_t x = z;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL; x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        s[i] = x ^ (x >> 31);
    }
}

static inline double hhat(int I, int nv, const double *V, const double *w) {
    double best = -1e300;
    for (int k = 0; k < nv; k++) {
        double v = 0.0;
        for (int i = 0; i < I; i++) v += V[k * I + i] * w[i];
        if (v > best) best = v;
    }
    return best;
}

int simulate(int I, int J, const double *K, const double *invbeta, const double *beta_r,
             const double *alpha_r, const double *h, const double *C, const double *xtab,
             int nv, const double *V, double L, double U, double r, double T_end,
             double T_burn, uint64_t seed, double *out,
             double batch_len, int n_batch, double *bout, int check_hg, double *diag,
             double grid_dt, int n_grid, double *ygrid)
{
    int gi = 0;
    seed_rng(seed);
    for (int k = 0; k < 3 * n_batch; k++) bout[k] = 0.0;
    for (int k = 0; k < 4; k++) diag[k] = 0.0;
    const double atol = 1e-9;
    double q[MAXJ] = {0}, E[MAXJ] = {0}, b[MAXJ], usedj[MAXJ], rate[MAXJ];
    double W[MAXI], Y[MAXI] = {0}, m[MAXI] = {0}, G[MAXI];
    double t = 0.0, T_mid = 0.5 * (T_burn + T_end), nev = 0.0;
    const double tol = 1e-12;
    int nout = 8 + 2 * J + I;
    for (int k = 0; k < nout; k++) out[k] = 0.0;

    while (t < T_end) {
        int idx = 0;
        for (int j = 0; j < J; j++) {
            if (q[j] < L - tol) E[j] = 1.0;
            if (q[j] >= U - tol) E[j] = 0.0;
            if (q[j] < U - tol) idx |= (1 << j);
        }
        const double *x = xtab + J * idx;
        double tot = 0.0;
        for (int j = 0; j < J; j++) tot += alpha_r[j];
        for (int j = 0; j < J; j++) {
            b[j] = (1.0 - E[j]) * x[j];
            usedj[j] = (q[j] > 0) ? b[j] : 0.0;
            rate[j] = beta_r[j] * usedj[j];
            tot += rate[j];
        }
        double t_next = t - log(unif()) / tot;

        for (int i = 0; i < I; i++) {
            double w = 0.0;
            for (int j = 0; j < J; j++) w += K[i * J + j] * q[j] * invbeta[j];
            W[i] = w; G[i] = Y[i] - m[i];
        }
        double hh = hhat(I, nv, V, W), hg = hhat(I, nv, V, G), hq = 0.0, anyE = 0.0;
        for (int j = 0; j < J; j++) { hq += h[j] * q[j]; anyE += E[j]; }

        /* pathwise assertions */
        if (hq < hh - atol * (1.0 + hh)) {
            diag[0] = 1; diag[1] = t; diag[2] = hq; diag[3] = hh; out[6] = nev; out[7] = T_end - T_burn; return 1;
        }
        for (int i = 0; i < I; i++) if (G[i] > W[i] + atol * (1.0 + W[i])) {
            diag[0] = 2; diag[1] = t; diag[2] = G[i]; diag[3] = W[i]; out[6] = nev; out[7] = T_end - T_burn; return 2;
        }
        if (check_hg && hg > hh + atol * (1.0 + hh)) {
            diag[0] = 3; diag[1] = t; diag[2] = hg; diag[3] = hh; out[6] = nev; out[7] = T_end - T_burn; return 3;
        }

        double a = t > T_burn ? t : T_burn, e = t_next < T_end ? t_next : T_end;
        if (e > a) {
            double ov = e - a;
            out[0] += ov * hq; out[1] += ov * hh; out[2] += ov * hg;
            if (anyE > 0) out[3] += ov;
            for (int j = 0; j < J; j++) {
                if (E[j] > 0) out[8 + j] += ov;
                else if (q[j] < U - tol) out[8 + J + j] += ov;
            }
            for (int i = 0; i < I; i++) {
                double used = 0.0;
                for (int j = 0; j < J; j++) used += K[i * J + j] * usedj[j];
                out[8 + 2 * J + i] += ov * (C[i] - used);
            }
            double a2 = a > T_mid ? a : T_mid;
            if (e > a2) { out[4] += (e - a2) * hq; out[5] += (e - a2) * hh; }
            if (n_batch > 0) {                        /* split [a, e) over the batches it touches */
                int kb = (int)((a - T_burn) / batch_len), ke = (int)((e - T_burn) / batch_len);
                if (ke >= n_batch) ke = n_batch - 1;
                for (int k = kb; k <= ke; k++) {
                    double lo = T_burn + k * batch_len, hi = lo + batch_len;
                    if (a > lo) lo = a;
                    if (e < hi) hi = e;
                    if (hi > lo) { bout[3 * k] += (hi - lo) * hq; bout[3 * k + 1] += (hi - lo) * hh; bout[3 * k + 2] += (hi - lo) * hg; }
                }
            }
        }
        {
            double hold = (t_next < T_end ? t_next : T_end) - t;
            double lvec[MAXI];
            for (int i = 0; i < I; i++) {
                double l = C[i];
                for (int j = 0; j < J; j++) if (K[i * J + j] != 0.0) l -= usedj[j];
                lvec[i] = l;
            }
            while (gi < n_grid) {                       /* grid samples inside [t, t_next) */
                double sg = T_burn + gi * grid_dt;
                if (sg >= t_next || sg > T_end) break;
                double ds = sg > t ? sg - t : 0.0;
                for (int i = 0; i < I; i++) ygrid[gi * I + i] = Y[i] - ds * lvec[i];
                gi++;
            }
            for (int i = 0; i < I; i++) {
                Y[i] -= hold * lvec[i];
                if (Y[i] < m[i]) m[i] = Y[i];
            }
        }
        t = t_next;
        if (t >= T_end) break;
        nev += 1.0;
        double u = unif() * tot, c = 0.0;
        int k, done = 0;
        for (k = 0; k < J; k++) {
            c += alpha_r[k];
            if (u < c) {
                q[k] += 1.0;
                for (int i = 0; i < I; i++) if (K[i * J + k] != 0.0) Y[i] += invbeta[k];
                done = 1; break;
            }
        }
        if (!done) {
            for (k = 0; k < J; k++) {
                c += rate[k];
                if (u < c) {
                    if (q[k] > 0) {
                        q[k] -= 1.0;
                        for (int i = 0; i < I; i++) if (K[i * J + k] != 0.0) Y[i] -= invbeta[k];
                        for (int i = 0; i < I; i++) if (Y[i] < m[i]) m[i] = Y[i];
                    }
                    break;
                }
            }
        }
    }
    out[6] = nev;
    out[7] = T_end - T_burn;
    return 0;
}
