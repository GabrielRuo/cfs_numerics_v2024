#!/usr/bin/env bash

# Get the directory of the currently executing script
script_dir="$(dirname "$0")"

# Launching jobs for all experiments
python "${script_dir}/src/cluster_submit.py" --experiment_name=n1_testing --config=gabs_test --n_runs 1 --runpath="${script_dir}/src/run.py"
