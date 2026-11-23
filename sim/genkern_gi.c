/* genkern_gi.c -- event-driven simulation of Definition 2.3 with renewal (GI) arrivals and
 * general service requirements, FIFO head-of-line service within each job type.
 *
 * Primitives per type j, Condition-1 families only (exponential moments), coded as 4 doubles
 * [code, p1, p2, p3]:
 *     code 0  exponential, rate p1
 *     code 1  Erlang-k, k = p1 phases each of rate p2            (SCV 1/k)
 *     code 2  hyperexponential H2: rate p2 w.p. p1, rate p3 otherwise  (SCV > 1)
 * Interarrival times of type j have mean 1/alpha^r_j; service REQUIREMENTS have mean
 * 1/beta^r_j. The head-of-line job of type j is worked off at the bandwidth b_j(t) of the
 * policy, so it completes when the cumulative bandwidth since it reached the head equals its
 * requirement; with exponential requirements this is a completion rate of beta^r_j b_j, i.e.
 * exactly the CTMC of genkern.c (equal in law, not bit for bit: the event mechanism differs).
 *
 * Policy, accumulators and output layout are those of genkern.c:
 *   Z_j = 1{q_j < U}; E_j <- 1 if q_j < L, E_j <- 0 if q_j >= U; b = (1-E) * x(Z) from xtab.
 *   W = K M q (M = diag(1/beta), limit means), h_hat(W) = max_k V_k . W over dual vertices V,
 *   same-path HGI from the netput Y_i = W_i - int (C_i - used_i) dt by the exact 1-d Skorokhod map.
 * out: 0 cost 1 hhat 2 hgipath 3 Eon 4 cost_2nd 5 hhat_2nd 6 n_events 7 window
 *      8..8+J-1 flagged time per j; 8+J..8+2J-1 below-U unflagged time per j;
 *      8+2J..8+2J+I-1 lost capacity per resource
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

/* one draw from the coded distribution d = [code, p1, p2, p3] */
static inline double draw(const double *d) {
    int code = (int)(d[0] + 0.5);
    if (code == 0) return -log(unif()) / d[1];
    if (code == 1) {
        int k = (int)(d[1] + 0.5);
        double prod = 1.0;
        for (int i = 0; i < k; i++) prod *= unif();
        return -log(prod) / d[2];
    }
    double u = unif();
    return -log(unif()) / (u < d[1] ? d[2] : d[3]);
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

void simulate_gi(int I, int J, const double *K, const double *invbeta, const double *h,
                 const double *C, const double *xtab, int nv, const double *V,
                 const double *arr, const double *srv,
                 double L, double U, double r, double T_end, double T_burn, uint64_t seed,
                 double *out)
{
    seed_rng(seed);
    double q[MAXJ] = {0}, E[MAXJ] = {0}, b[MAXJ], usedj[MAXJ];
    double ra[MAXJ], rs[MAXJ] = {0};                 /* residual interarrival time, residual work */
    double W[MAXI], Y[MAXI] = {0}, m[MAXI] = {0}, G[MAXI];
    double t = 0.0, T_mid = 0.5 * (T_burn + T_end), nev = 0.0;
    const double tol = 1e-12;
    int nout = 8 + 2 * J + I;
    for (int k = 0; k < nout; k++) out[k] = 0.0;
    for (int j = 0; j < J; j++) ra[j] = draw(arr + 4 * j);

    while (t < T_end) {
        int idx = 0;
        for (int j = 0; j < J; j++) {
            if (q[j] < L - tol) E[j] = 1.0;
            if (q[j] >= U - tol) E[j] = 0.0;
            if (q[j] < U - tol) idx |= (1 << j);
        }
        const double *x = xtab + J * idx;
        for (int j = 0; j < J; j++) {
            b[j] = (1.0 - E[j]) * x[j];
            usedj[j] = (q[j] > 0) ? b[j] : 0.0;
        }
        /* next event: the smallest residual clock */
        double dt = INFINITY; int ev = -1;
        for (int j = 0; j < J; j++) if (ra[j] < dt) { dt = ra[j]; ev = j; }
        for (int j = 0; j < J; j++) if (q[j] > 0 && b[j] > 0.0) {
            double td = rs[j] / b[j];
            if (td < dt) { dt = td; ev = J + j; }
        }
        if (dt < 0.0) dt = 0.0;
        double t_next = t + dt;

        for (int i = 0; i < I; i++) {
            double w = 0.0;
            for (int j = 0; j < J; j++) w += K[i * J + j] * q[j] * invbeta[j];
            W[i] = w; G[i] = Y[i] - m[i];
        }
        double hh = hhat(I, nv, V, W), hg = hhat(I, nv, V, G), hq = 0.0, anyE = 0.0;
        for (int j = 0; j < J; j++) { hq += h[j] * q[j]; anyE += E[j]; }

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
        }
        {
            double hold = (t_next < T_end ? t_next : T_end) - t;
            for (int i = 0; i < I; i++) {
                double l = C[i];
                for (int j = 0; j < J; j++) if (K[i * J + j] != 0.0) l -= usedj[j];
                Y[i] -= hold * l;
                if (Y[i] < m[i]) m[i] = Y[i];
            }
        }
        /* advance the residual clocks over the holding interval */
        for (int j = 0; j < J; j++) {
            ra[j] -= dt;
            if (q[j] > 0) rs[j] -= b[j] * dt;
        }
        t = t_next;
        if (t >= T_end) break;
        nev += 1.0;
        if (ev < J) {                                   /* arrival of type ev */
            int k = ev;
            q[k] += 1.0;
            ra[k] = draw(arr + 4 * k);
            if (q[k] == 1.0) rs[k] = draw(srv + 4 * k); /* new head-of-line job */
            for (int i = 0; i < I; i++) if (K[i * J + k] != 0.0) Y[i] += invbeta[k];
        } else {                                        /* departure of type ev - J */
            int k = ev - J;
            q[k] -= 1.0;
            rs[k] = (q[k] > 0) ? draw(srv + 4 * k) : 0.0;
            for (int i = 0; i < I; i++) if (K[i * J + k] != 0.0) Y[i] -= invbeta[k];
            for (int i = 0; i < I; i++) if (Y[i] < m[i]) m[i] = Y[i];
        }
    }
    out[6] = nev;
    out[7] = T_end - T_burn;
}
