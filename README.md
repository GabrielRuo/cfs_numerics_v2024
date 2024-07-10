# Changes in v2024

## New Requirements and Conda Environment setup

Currently, work is underway which should smoothly set up a consistent Conda environment. This uses the shell script "setup_env.sh".

* Before running the numerics, once run "setup_env.sh", this creates a Conda environment with the name cfs_v2024_env.
* The shell script "run_test_job.sh" then calls a quick test run of the old code.

# (Old version of README below) Numerical Analysis of The Causal Action Principle in Low Dimensions

This is code accompanying the paper [Numerical Analysis of The Causal Action Principle in Low Dimensions](https://arxiv.org/abs/2201.06382).

## Installation

We have tested this code with Python 3.9 and rely on the following packages

* `jax`
* `jaxlib`
* `numpy`
* `absl-py`
* `scipy`

Simply install the dependencies in the `requirements.txt` file via `pip`:

```pip install -r requirements.txt```

Note that due to some breaking changes, our code is not compatible with the latest versions of `jax` and `jaxlib`.


## Running the code

### Individual local runs

A single optimization can be performed using the main `run.py` driver file.
All arguments are described in the file directly. The most important ones are
the number of particles `f`, the spin dimension `n`, and the number of
spacetime points `m`. For example, a single optimization can be performed via

```python run.py --f=4 --n=2 --m=128 --seed=42```

All other options are also described in the paper.

### Sweeps

To reproduce the results in the paper, larger sweeps over many settings and
multiple random seeds are required. We provide a driver script to automatically
perform all required runs on a slurm managed compute cluster. The script
`cluster_submit.py` automatically schedules all the runs required to reproduce
any given plot in the paper, where the sweeps required for the different 
settings are stored in `configs.py`. To reproduce all results of the paper at
once, and to see examples of how to call `cluster_submit.py`, see the 
`run_all_experiments.sh` script.

## Reading and interpreting results

Here we describe how to interpret and use the output of a single optimization.
The 3 most important files placed in the `output_dir` of a `run.py` run are:
`flags.json`, `results.npz`, `parameters_last.npz`

1. The `flags.json` file is a simple json file containing the settings for this run.
This file will also contain a whole host of other specific settings for this run, which are not really important for our purposes here.
2. The `results.npz` file contains the results that were collected throughout the optimization such as weights (m x 1), spectra (m x 2 n), hamiltonians (m x f x f), xs (m x f x f), n, f, m and should be self-explanatory from the keys as well as shapes.
4. The `parameters_last.npz` is an additional file containing the raw optimization parameters at the final step of the optimization. Even though there is redundancy, the raw optimization parameters are not as easily interpretable as the higher level results in `results.npz`. These are mostly useful to continue optimization later form the final results of a prior run or to reconstruct some details missing in `results.npz`. It contains the following entries:
    * 'weights': the final weights for the m space-time points (WARNING: these are not the actual weights, but the corresponding optimization parameters, i.e., not normalized) (dimensions: m x 1)
    * 'pos_spectrum': the logs of the positive part of the spectra (dimensions: m x n)
    * 'neg_spectrum': the logs of the negative part of the spectra (dimensions: m x n)
    * 'block_ul': (roughly) the parameters for the upper left block of the Hamiltonians (dimensions: m x 2n x 2n)
    * 'block_ur': (roughly) the parameters for the upper right (and lower left) block of the Hamiltonians (dimensions: m x 2n x f)

The results are stored via the `numpy` library in python. In Python we can easily read them via

```py
import numpy as np

data = np.load('path/to/file/results.npz')

n = data['n']
f = data['f']
m = data['m']
action = data['action']
weights = data['weights']

# and so on for the other entries
```
