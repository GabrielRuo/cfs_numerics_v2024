#!/bin/bash

# Load Anaconda module
module load Anaconda

# Check if the conda environment exists
if conda info --envs | grep -q '^cfs_v2024_env\s'; then
    echo "Found existing conda environment 'cfs_v2024_env'. Resetting..."
    # Deactivate environment before removing
    conda deactivate
    # Remove existing environment
    conda env remove -n cfs_v2024_env -y
fi

# Create conda environment with Python 3.10 and pip
conda create -n cfs_v2024_env python=3.10 pip -y

# Activate the newly created environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cfs_v2024_env

# Install packages from requirements.txt
pip install -r requirements.txt

# Final message
echo "Environment 'cfs_v2024_env' has been created and set up."
