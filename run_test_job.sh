#!/bin/bash

# This script sets up the environment and runs a Python script

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

# Run the Python script with specified arguments
python ./src/run.py --f=2 --n=1 --m=16 --seed=4

# Deactivate the Conda environment (optional)
# conda deactivate  # Uncomment if you want to deactivate after script execution
