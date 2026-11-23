# Standalone module with plain imports; pure Python. The compiled kernels are in sim/.

"""
check.py -- end-to-end verification driver.

Builds the paper toy (Example 2.2), runs the offline pass with all
admissibility assertions, estimates J_E and the SSC gap on simulation
data (je.py) under Definition 2.3 verbatim, computes the RBM benchmark
HGI_E (simulation.py, exact reflection map, sqrt(dt)-extrapolated), and
then runs the CONTROL EXPERIMENT of the report (hysteresis flags off,
||v^b|| = 0.24 > b_max) for comparison. The control experiment is OUTSIDE
Definition 2.3; it is printed as such and is not a recommended setting.
Runs in ~1 minute in Colab; raise r and T for report-grade numbers.

    python check.py
"""
import numpy as np

from rsn import make_2lln
from offline import OfflineTables
from je import run_JE
from simulation import benchmark_hgi_e

if __name__ == "__main__":
    net = make_2lln()
    hhat = lambda W: np.asarray(W).max(axis=1)   # analytic for the 2LLN

    print(net)
    tab = OfflineTables(net, gamma_c=0.98, gamma_b=0.98)
    print()
    print(tab.summary())

    print("\n-- Definition 2.3 verbatim (SSC gap is the finite-r test) --")
    out = run_JE(net, r=50, T=10, c1=1.0, c2=4.0, kappa=0.15, seed=1,
                 hhat_fn=hhat, tables=tab)
    print(f"J_E     = {out['JE']:.3f} +- {out['JE_ci']:.3f}")
    print(f"SSC gap = {out['ssc_gap']:.4f} +- {out['gap_ci']:.4f}"
          f"   (decays like c2*r^(kappa-1))")
    print(f"E-idle  = {out['E_frac']:.3f}")

    print("\n-- RBM benchmark HGI_E (exact reflection map, sqrt(dt)-extrapolated) --")
    bench, se = benchmark_hgi_e(net, hhat_fn=hhat, T=20000.0, seeds=(0, 1))
    print(f"HGI_E   = {bench:.3f} +- {se:.3f}   (2.81 +- 0.02 with 4 seeds)")

    print("\n-- CONTROL EXPERIMENT (outside Definition 2.3): E_mask=0, vb_norm=0.24 --")
    print("   flags disabled and ||v^b|| above b_max; a labeled departure, NOT a recommendation")
    out2 = run_JE(net, r=50, T=20, c1=1.0, c2=10.0, kappa=0.22, seed=1,
                  E_mask=0, vb_norm=0.24, hhat_fn=hhat)
    print(f"J_E     = {out2['JE']:.3f} +- {out2['JE_ci']:.3f}")
    ratio = out2["JE"] / bench
    print(f"control level / benchmark = {ratio:.2f}")
    verdict = "PASS" if ratio < 1.6 else "CHECK SETTINGS"
    print(f"RESULT: {verdict}")
