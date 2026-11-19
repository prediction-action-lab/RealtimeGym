#!/bin/bash
# Non-interactive evaluation. Run from the chtc directory:
#   bash submit.sh <run_name>
# Results land in results/<run_name>/, logs in logs/<run_name>/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

name="${1:?usage: bash submit.sh <run_name>}"
results_dir="results/${name}"
log_dir="logs/${name}"
mkdir -p "${results_dir}" "${log_dir}"

condor_submit job.sub results_dir="${results_dir}" log_dir="${log_dir}"
