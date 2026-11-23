# Standalone module with plain imports; pure Python. The compiled kernels are in sim/.

"""
je.py -- estimate the ergodic cost J_E (and the SSC gap) on simulation data.

Built on exact Gillespie simulation, time-weighted averages and batch-means
confidence intervals.

    from rsn import make_2lln
    from je import run_JE
    out = run_JE(make_2lln(), r=50, T=10)
    print(out["JE"], out["JE_ci"], out["ssc_gap"])
"""
import numpy as np

from offline import OfflineTables
from policy import GeneralThresholdPolicy
from simulation import gillespie_ergodic
from je_stats import batch_means_ci


def run_JE(rsn, r, T=10.0, c1=1.0, c2=4.0, kappa=0.15, seed=0,
           n_snapshots=400, burn_frac=0.3, E_mask=None, vb_norm=None,
           gamma_c=0.98, gamma_b=0.98, hhat_fn=None, n_batches=16,
           tables=None):
    """
    One ergodic run of the threshold policy on the network `rsn` at
    heavy-traffic scale r over T diffusion time units (pure-Python exact
    Gillespie). Defaults instantiate Definition 2.3 verbatim. E_mask=0 (flags
    off) and vb_norm above b_max are the CONTROL EXPERIMENT of the report:
    labeled departures OUTSIDE Definition 2.3, never a recommended setting.

    Returns dict(JE, JE_ci, ssc_gap, gap_ci, E_frac, snapshots).
    """
    tables = (OfflineTables(rsn, gamma_c=gamma_c, gamma_b=gamma_b,
                            vb_norm=vb_norm) if tables is None else tables)
    pol = GeneralThresholdPolicy(rsn, c1=c1, c2=c2, kappa=kappa,
                                 tables=tables, E_mask=E_mask)
    out = gillespie_ergodic(rsn, pol, float(r), T_diff=T,
                            burn_frac=burn_frac, n_snapshots=n_snapshots,
                            rng=np.random.default_rng(seed))
    snaps = out["snapshots"]
    je_series = (snaps @ rsn.h) / float(r)
    if hhat_fn is not None:
        hg = np.asarray(hhat_fn((snaps @ (rsn.K @ rsn.M).T) / float(r)))
    else:
        from simulation import hhat_batch
        hg = hhat_batch((snaps @ (rsn.K @ rsn.M).T) / float(r), rsn)
    ci_je = batch_means_ci(je_series, n_batches)
    ci_gap = batch_means_ci(je_series - hg, n_batches)
    return dict(JE=out["JE"], JE_ci=ci_je["half_width"],
                ssc_gap=float(np.mean(je_series - hg)),
                gap_ci=ci_gap["half_width"], E_frac=out["E_frac"],
                snapshots=snaps)
