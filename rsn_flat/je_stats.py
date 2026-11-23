# Standalone module with plain imports; pure Python. The compiled kernels are in sim/.

"""
je_stats.py — statistical utilities for ergodic simulation output.

batch_means_ci: nonoverlapping batch-means confidence interval for the mean
of a stationary, autocorrelated series (the standard method for steady-state
simulation output analysis). The series here is the evenly-time-spaced
snapshot read-out of an ergodic run; batches are valid when the batch span
well exceeds the process mixing time (for the workload RBM, mixing ~
Sigma_ii/theta_i^2 diffusion units — with default settings each batch spans
(1-burn)*T/n_batches diffusion units; check batch_span_ok in the output).
"""

import numpy as np
from scipy import stats as _st


def batch_means_ci(series, n_batches=20, conf=0.95):
    """
    Returns dict(mean, half_width, se, n_batches). The half-width uses the
    Student-t quantile on n_batches-1 degrees of freedom applied to the
    batch means — the standard steady-state output-analysis estimator.
    """
    x = np.asarray(series, dtype=float).ravel()
    n = x.size
    nb = int(n_batches)
    if n < 2 * nb:
        raise ValueError(f"need >= {2*nb} points for {nb} batches, got {n}")
    m = n // nb
    b = x[: m * nb].reshape(nb, m).mean(axis=1)
    mean = float(b.mean())
    se = float(b.std(ddof=1) / np.sqrt(nb))
    t = float(_st.t.ppf(0.5 + conf / 2.0, nb - 1))
    return dict(mean=mean, half_width=t * se, se=se, n_batches=nb)
