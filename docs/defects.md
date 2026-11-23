# Defects of an earlier version of the code

| # | what was wrong | where it is fixed | covering test |
|---|---|---|---|
| 1 | The sign of eta was flipped: (alpha beta_bar - alpha_bar beta)/beta^2 instead of (alpha_bar beta - alpha beta_bar)/beta^2. | `rsn_flat/rsn.py`, `RSN._condition2`; theta = K eta < 0 is checked at construction and raises. | `test_condition2.py::test_splits_share_eta_theta_sigma`, `test_condition2.py::test_inconsistent_or_unstable_inputs_raise` |
| 2 | A term R = diag(rho) in the queue update cancelled all services. | `rsn_flat/simulation.py`, `gillespie_ergodic`: departures at rate beta^r_j b_j 1{q_j > 0}. | `test_paper_reproduction.py::test_reproduces_colab_sweep_entry_bit_for_bit`, `test_engine_crosscheck.py::test_rsn_flat_vs_genkern_r50`; script `sim/increment_test.py` (mean increment equals theta) |
| 3 | Time was scaled twice: rates multiplied by r and a horizon of r^2 T. | `rsn_flat/simulation.py`: rates are O(1), the horizon is r^2 T, read-outs are Q/r. | the two tests of defect 2; script `sim/increment_test.py` (covariance equals Sigma per unit time) |
| 4 | The cost routine ignored the random generator passed to it, so replications were identical. | `rsn_flat/je.py`, `run_JE(..., seed)` uses `default_rng(seed)`. | `test_defect_coverage.py::test_seed_sensitivity_pure_python`, `test_defect_coverage.py::test_seed_sensitivity_compiled` |
| 5 | The allocation b was clipped to [0, 1], which disabled v^c. | `rsn_flat/policy.py`, `allocation`: no projection; K x <= C and x >= 0 are asserted. | `test_policy_checks.py::test_allocation_is_table_times_flag_and_feasible`, `test_policy_checks.py::test_tampered_table_raises_and_counts` |
| 6 | The drift of the queueing side did not match the drift of the reflected Brownian motion. | `rsn_flat/rsn.py`: one `RSN` object supplies alpha^r, beta^r, theta and Sigma to both sides. | `test_condition2.py::test_pure_python_simulator_uses_beta_r`; script `sim/increment_test.py` (toy and network D) |
| 7 | The covariance used squared standard deviations instead of renewal variance rates. | `rsn_flat/rsn.py`: Sigma = K M (diag(alpha scv_u) + diag(beta scv_v) R) M^T K^T. | `test_defect_coverage.py::test_sigma_uses_renewal_variance_rates_on_heterogeneous_network` |
| 8 | The tau-leap step was too coarse for the thresholds. | `rsn_flat/simulation.py`: exact event-by-event simulation; no tau-leap code is present. | `test_policy_checks.py::test_hysteresis_flag_transitions` (threshold logic) |
| 9 | The constants of the check script gave an empty hysteresis band. | `rsn_flat/policy.py`: (c2~ - c1~) r^kappa > 1 is required and raises otherwise. | `test_policy_checks.py::test_narrow_hysteresis_band_raises_not_warns` |

Test files are in `rsn_flat/tests/`.
