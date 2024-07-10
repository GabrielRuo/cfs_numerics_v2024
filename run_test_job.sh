#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cfs_v2024_env
python ./src/run.py --f=2 --n=1 --m=16 --seed=4
