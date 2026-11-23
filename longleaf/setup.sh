#!/bin/bash
# One-time setup on the cluster (run on a login node from this directory):  bash setup.sh
#   1. finds a Python >= 3.8 with numpy and scipy (tries the current one, then `module load`, then a virtual environment),
#   2. writes env.sh (sourced by submit.sbatch) with whatever made it work,
#   3. compiles sim/genkern.c, 4. runs the self-test and measures the speed.
# If your site needs a specific module, set it first:   PYMODULE=python/3.12.4 bash setup.sh
set -eo pipefail
cd "$(dirname "$0")"
mkdir -p logs results
ok_py() { python3 - <<'PY' 2>/dev/null
import sys
assert sys.version_info >= (3, 8)
import numpy, scipy.optimize
PY
}
ENVLINES=""
if ! ok_py; then
  if type module >/dev/null 2>&1; then
    for m in ${PYMODULE:-} python anaconda; do
      [ -n "$m" ] || continue
      if module load "$m" 2>/dev/null && ok_py; then ENVLINES="module load $m"; break; fi
    done
  fi
fi
if ! ok_py; then
  echo "no Python >= 3.8 with numpy and scipy found; creating a virtual environment in .venv (needs internet for pip)"
  python3 -c 'import sys; assert sys.version_info >= (3, 8), "python3 is older than 3.8: load a newer Python module first (module avail python)"'
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install --quiet --upgrade pip && pip install --quiet numpy scipy
  ENVLINES="${ENVLINES:+$ENVLINES
}source \"$(pwd)/.venv/bin/activate\""
  ok_py || { echo "SETUP FAILED: numpy/scipy still not importable"; exit 1; }
fi
{ echo "# written by setup.sh on $(hostname), $(date)"; echo "$ENVLINES"; } > env.sh
echo "python: $(python3 --version 2>&1) at $(command -v python3)"
CC=${CC:-gcc}
# no -march=native: login and compute nodes can have different CPUs
$CC -O3 -shared -fPIC -o sim/genkern.so sim/genkern.c -lm
echo "compiled sim/genkern.so with $($CC --version | head -1)"
python3 run_toy.py --selftest
python3 run_toy.py --bench
echo
echo "Setup done. Next:  python3 make_jobs.py --r 20000 40000 --T 50 --seeds 1-4   and then the sbatch line it prints."
