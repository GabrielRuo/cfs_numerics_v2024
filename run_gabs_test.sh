#!/usr/bin/env bash


# Load Anaconda module (adjust the version or path as necessary)
module load Anaconda/2023.09-0-hpc1

# Activate Conda environment (assuming conda is properly initialized)
# It's good practice to include error handling if conda.sh cannot be found
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh
else
    echo "Error: conda.sh not found. Make sure Conda is properly installed and initialized."
    exit 1
fi

# Activate the specific Conda environment
conda activate cfs_v2024_env

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate Conda environment 'cfs_v2024_env'."
    exit 1
fi


# Get the directory of the currently executing script
script_dir="$(dirname "$0")"

# Launching jobs for all experiments

python "${script_dir}/src/cluster_submit.py" --experiment_name=n1_testing_0712 --config=gabs_test --n_runs 1 --runpath=/home/x_rojon/cfs_numerics_v2024/src/run.py --result_dir=/home/x_rojon/results/

python "${script_dir}/src/cluster_submit.py" --experiment_name=n1_testing_0712 --config=gabs_test --n_runs 1 --runpath=/home/x_rojon/cfs_numerics_v2024/src/run_new.py --result_dir=/home/x_rojon/results_new/


