#!/bin/bash
# Array job: task k runs line k of jobs.txt (written by make_jobs.py, which also prints the sbatch command).
#   sbatch --array=1-12 --time=1-00:00:00 submit.sbatch
# Each run uses one core and well under 1 GB. A run that hits the time limit is lost (the kernel has no
# checkpoint), so keep --time above the estimate printed by make_jobs.py. Finished runs are skipped on resubmission.
#SBATCH --job-name=hgi_toy
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2g
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
set +u; source ./env.sh; set -u                       # written by setup.sh (modules / virtual environment)
TASK=${SLURM_ARRAY_TASK_ID:?submit with --array=1-N}
ARGS=$(sed -n "${TASK}p" jobs.txt)
[ -n "$ARGS" ] || { echo "jobs.txt has no line $TASK"; exit 1; }
echo "$(date)  host $(hostname)  task $TASK: python3 run_toy.py $ARGS"
python3 run_toy.py $ARGS
echo "$(date)  done"
