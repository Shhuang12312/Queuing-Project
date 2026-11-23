# An Exact Implementation and Finite-Scale Study of the Simple-Form Control Policy with HGI Performance on the Two-Resource Sharing Network

Code and result files of the research report with the title above.

The code implements the control policy of Definition 2.3 of Budhiraja and Johnson, Ann. Appl. Probab. 34(1B):851-916,
2024, doi:10.1214/23-AAP1979, for any resource sharing network that satisfies Conditions 1 to 3 of that paper.
It computes the policy's vectors offline, simulates the controlled network event by event, computes the HGI benchmarks
from the limiting reflected Brownian motion, and runs the simulations of the report.

Two experiments are outside Definition 2.3 and are labeled so wherever they appear: the control experiment with the
hysteresis flags switched off, and the pattern-dependent cap on the vector v^b (`vb_cap="per_pattern"`).

## Requirements and build

Python 3.9 or newer with numpy, scipy, matplotlib and pytest; a C compiler (gcc or clang).

```bash
cd sim
./build.sh            # compiles toykern3.c, genkern.c, genkern_gi.c into shared libraries
python selftest.py    # about 30 s
```

The compiled libraries are not part of the repository. `rsn_flat/` alone needs no compiler.

## Layout

```
rsn_flat/     the policy in pure Python
  rsn.py          network, Conditions 2 and 3, drift and covariance of the limit
  offline.py      offline pass: the vectors and the set of Definition 2.3, with every condition asserted
  policy.py       the online policy (thresholds, hysteresis flags, allocation)
  simulation.py   event-by-event simulation, the cost function h-hat, the benchmarks HGI_E and HGI_D
  je.py, je_stats.py, check.py   ergodic cost with confidence intervals; end-to-end check
  tests/          pytest suite
  notebooks/      two Colab notebooks and the script that refreshes the module cells of one of them
  convergence_results.json   output of the notebook "Queueing Project.ipynb"
sim/          compiled kernels, their wrappers, experiment scripts, figure scripts
  toykern3.c, toysim3.py         kernel and wrapper for the toy network
  genkern.c, gensim.py           kernel and wrapper for a general network (J <= 20, I <= 8)
  genkern_gi.c, gensim_gi.py     kernel and wrapper for non-exponential primitives
  networks.py                    the seven networks A to G and the rule for the band c2
  flagmodel.py                   the flag formula
  common.py                      helpers shared by the scripts
  make_*_fig.py                  one script per figure of the report
  other *.py                     one script per experiment; each writes the result file named in its docstring
  results/                       every result file (JSON, NPZ, run logs)
  results/final/                 the four figures of the report (PNG and PDF)
  results/large_r/               the toy runs at r = 20 000 and 40 000
analysis/     tables and in-text numbers computed from the stored result files, with their output files
longleaf/     package for a SLURM cluster: the toy network at large r
docs/
  report_facts.md   every number of the report with its stored value, result file, key and script
  defects.md        nine defects of an earlier version of the code, the fix and the covering test
```

Network G is Example 3 of Budhiraja and Johnson, Math. Oper. Res. 45(3):797-832, 2020 (Example 5.6 of
arXiv:1710.09042). The result files identify it by the dictionary key `G ranking-map fails (BJ20 Ex. 6.6)` of
`sim/networks.py`; the key is a label only.

## Tests

```bash
cd sim && ./build.sh && cd ..
python -m pytest rsn_flat/tests       # 152 tests, about 6 minutes
```

The tests that need a compiled kernel skip themselves if `sim/build.sh` was not run.

## Regenerating the tables and figures

Run the commands from `sim/` unless a path is given. Result files are under `sim/results/`. Times are for the
four-core machine on which the stored runs were made, unless stated. A core-hour figure marked "estimated" is the
stored event count divided by the kernel speed of `engine_speed.json`.

Scripts that simulate write the result files they are listed with. The figure scripts and
`analysis/make_paper_numbers.py` read stored files only.

| item | command | result files read | run time |
|---|---|---|---|
| Table 1, rows | `python ../analysis/make_paper_numbers.py` | `production.json`, `table1_drift.json` | 10 s |
| Table 1, runs | `python production.py`, then `python table1_rerun.py` | writes `production.json`, `table1_drift.json` | 2.6 core-hours for the ergodic rows, two processes; then 8 minutes on four cores |
| Table 2, rows | `python ../analysis/make_paper_numbers.py` | `paper_cells.json`, `multi_bench_crn.json` | 10 s |
| Table 2, runs | `python paper_cells.py` | writes `paper_cells.json` | about 3 core-hours (estimated), four processes |
| Table 2, seeds 3 to 8 at r = 1000, 2000 | `python drift_check.py` | writes `drift_check.json` | 15 minutes on four cores |
| Table 3, rows | `python ../analysis/make_paper_numbers.py` | `decomp_runs.json`, `table3_r4000.json`, `multi_bench.json`, `multi_bench_crn.json` | 10 s |
| Table 3, bands | `python bench.py`, `python run_network_g.py`, `python vbcap_runs.py` | write `multi_bench.json`, `vbcap_summary.json` | `bench.py`: not recorded; `run_network_g.py`: 2 core-hours, two processes; `vbcap_runs.py`: 0.9 core-hours, four processes |
| Table 3, runs at r = 1000, 2000 | `python decomp_runs.py` | writes `decomp_runs.json` | about 7 core-hours (estimated), four processes |
| Table 3, runs at r = 4000 | `python table3_r4000.py` | writes `table3_r4000.json`, `table3_r4000/` | 4 hours 40 minutes on four cores (18 core-hours) |
| Benchmarks HGI_E | `python multi_bench_crn.py` | writes `multi_bench_crn.json` | 10 minutes, one process |
| Benchmark HGI_D | `python hgi_d_bench.py` | writes `hgi_d_toy.json` | minutes, one process (not recorded) |
| In-text numbers | `python ../analysis/make_paper_numbers.py`, `python ../analysis/phi_fit.py` | the files listed in the two docstrings | 10 s and 1 s |
| Figure `toy_convergence_v5` | `python make_toy_fig.py` | `production.json`, `paper_cells.json`, `large_r/*.json`, `control_experiment.json`, `multi_bench_crn.json` | 3 s |
| Figure `multi_convergence_v2` | `python make_multi_fig.py` | `decomp_runs.json`, `table3_r4000.json`, `multi_bench.json`, `multi_bench_crn.json` | 2 s |
| Figure `benchmark_sqrt_dt` | `python make_benchmark_fig.py` | `benchmark_bias.json`, `multi_bench_crn.json` | 1 s |
| Figure `queue_trace` | `python make_trace_fig.py --draw` | `queue_trace_data.npz` | 3 s |

Each figure script writes its PNG and PDF to `results/final/`, or to another folder with `--out DIR`. It checks the
plotted values against the values printed in the report and stops without drawing if one does not match.
`make_trace_fig.py` without `--draw` first simulates the two traces again (about 2 minutes, two processes) and
overwrites `queue_trace_data.npz`.

Runs behind the figures and the in-text numbers:

| result file | command | run time |
|---|---|---|
| `large_r/*.json` (toy, r = 20 000 and 40 000, seeds 1 to 4) | `longleaf/run_toy.py` on a SLURM cluster, see below | one core per run; 3.3 to 3.7 hours per run at r = 20 000, 14.0 to 15.7 hours per run at r = 40 000 |
| `control_experiment.json` | `python control_experiment.py` | 13 minutes on four cores |
| `benchmark_bias.json` | `python benchmark_bias.py` | 6 minutes on four cores |
| `screen_r300.json` | `python screen_r300.py` | 1 minute on four cores |
| `split_r100.json` | `python split_r100.py` | 10 s on four cores |
| `network_g_check.json` | `python network_g_check.py` | 13 minutes, two processes |
| `engine_speed.json` | `python engine_speed.py` | 11 minutes, one process |
| `increment_test.json`, `increment_test_D.json` | `python increment_test.py`, `python increment_test.py "D unequal rates" 41 results/increment_test_D.json` | minutes (not recorded) |
| `jd_seeds.json` | `python jd_seeds.py` | 21 minutes on four cores |
| `vbcap_runs.json` | `python vbcap_runs.py`, `python vbcap_g660.py`, `python vbcap_gscan.py` | 0.9 core-hours for the first, 0.9 core-hours for the other two together |

`docs/report_facts.md` gives the script behind every result file used in the report.

## Runs on a cluster

`python longleaf/make_zip.py` writes `dist/hgi_toy_longleaf.zip`, a self-contained package for a SLURM cluster.
After unzipping it on the cluster:

```bash
bash setup.sh                                              # finds Python, compiles the kernel, self-test, speed
python3 make_jobs.py --r 20000 40000 --T 50 --seeds 1-4    # writes jobs.txt and prints the sbatch command
sbatch --array=1-8 --time=0-16:00:00 submit.sbatch         # use the line that make_jobs.py printed
python3 collect.py                                         # after the jobs finish: table and summary.csv
```

One run uses one core and under 1 GB of memory and simulates about 6 r^2 T events. A run that reaches its time
limit is lost, because the kernel has no checkpoint. The eight files in `sim/results/large_r/` are the output of
these commands.

## Seeds

Every simulation takes an integer seed, and every script names its seeds in its source and stores the seed in each
result record.

- The compiled kernels use the generator xoshiro256**, initialized from the seed by splitmix64. All event times of a
  run come from this one stream.
- On one machine a kernel run is reproduced bit for bit from its seed. On another machine the C library may round
  `log()` differently, which changes the path but not its law.
- The pure-Python simulation and the benchmark routines use `numpy.random.default_rng(seed)`.
- The benchmark HGI_E uses seeds 0 to 7 and evaluates both step sizes of its extrapolation on the same Brownian path
  of each seed.
- Two policies run from the same seed do not share arrivals and job sizes.
