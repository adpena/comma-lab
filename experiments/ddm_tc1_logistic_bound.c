/* Scorer-free full-categorical logistic objective, gradient, and Hessian.
 * Analysis only: no arithmetic encoder or receiver implementation here. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define K 5
#define F 7
#define D (K * F)

double tc1_objective(size_t n, const double *logp, const int16_t *phi,
                     const uint8_t *truth, const uint8_t *group,
                     const double *weights, int derivatives,
                     double *gradient, double *hessian) {
    const double scale = 0.693147180559945309417232121458176568 / 1024.0;
    double loss = 0.0;
    size_t i;
    memset(gradient, 0, D * sizeof(double));
    memset(hessian, 0, K * F * F * sizeof(double));
    for (i = 0; i < n; ++i) {
        double z[K], p[K], x[K][F], mean[F] = {0};
        double maximum = -INFINITY, denominator = 0.0;
        size_t bank = group[i];
        int k, a, b;
        if (bank >= K || truth[i] >= K) return NAN;
        for (k = 0; k < K; ++k) {
            z[k] = logp[i*K+k];
            for (a = 0; a < F; ++a) {
                x[k][a] = (double)phi[(i*K+k)*F+a] * scale;
                z[k] += weights[bank*F+a] * x[k][a];
            }
            if (z[k] > maximum) maximum = z[k];
        }
        for (k = 0; k < K; ++k) {
            p[k] = exp(z[k] - maximum);
            denominator += p[k];
        }
        loss += log(denominator) + maximum - z[truth[i]];
        if (!derivatives) continue;
        for (k = 0; k < K; ++k) {
            p[k] /= denominator;
            for (a = 0; a < F; ++a) mean[a] += p[k] * x[k][a];
        }
        for (a = 0; a < F; ++a) {
            gradient[bank*F+a] += mean[a] - x[truth[i]][a];
            for (b = 0; b < F; ++b) {
                double v = 0.0;
                for (k = 0; k < K; ++k)
                    v += p[k] * (x[k][a]-mean[a]) * (x[k][b]-mean[b]);
                hessian[(bank*F+a)*F+b] += v;
            }
        }
    }
    return loss;
}
