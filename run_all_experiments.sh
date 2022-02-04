#!/usr/bin/env bash

# Launching jobs for all experiments
python src/cluster_submit.py --experiment_name=n1_large_f --config=n1_large_f --n_runs 1
python src/cluster_submit.py --experiment_name=n2_large_f --config=n2_large_f --n_runs 1

python src/cluster_submit.py --experiment_name=n1_f2 --config=n1_f2 --n_runs 10
python src/cluster_submit.py --experiment_name=n1_f3 --config=n1_f3 --n_runs 10
python src/cluster_submit.py --experiment_name=n2_f4 --config=n2_f4 --n_runs 10
