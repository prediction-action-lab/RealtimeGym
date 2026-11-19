#!/bin/bash
source .env

results_dir=results/interactive
mkdir -p ${results_dir}
mkdir -p ${results_dir}/logs
condor_submit -i job_i.sub
