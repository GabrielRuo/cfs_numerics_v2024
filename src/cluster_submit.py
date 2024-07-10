"""Launch parallel runs on a slurm-managed cluster for larger sweeps."""
import json
import os
from copy import deepcopy
from itertools import product
from typing import Text, Sequence, Dict, Any, List

from absl import app
from absl import flags

import configs

Sweep = Dict[Text, Sequence[Any]]

flags.DEFINE_string("experiment_name", None,
                    "The name of the experiment (used for output folder).")
flags.DEFINE_string("config", None,
                    "The name of the desired config from the `config.py` file."
                    "If `None` use config specified here in `main`.")
flags.DEFINE_string("result_dir", "/home/results/",
                    "Base directory for all results.")
# don't need a user name
# flags.DEFINE_string("username", None,
#                     "Username on the cluster.")
flags.DEFINE_string("pythonpath", None,
                    "Absolute path to the python executable (e.g. conda env).")
flags.DEFINE_string("runpath", None,
                    "Absolute path to the `run.py` file of this project.")
flags.DEFINE_integer("n_runs", 10,
                     "The number of runs per setting (different random seeds).")
flags.DEFINE_bool("check_existing", True,
                  "Check whether results for runs already exist and skip. "
                  "Set to False for re-running everything.")
flags.DEFINE_bool("gpu", False, "Whether to use GPUs.")
flags.mark_flag_as_required("experiment_name")
# flags.mark_flag_as_required("username")
# flags.mark_flag_as_required("pythonpath")
# flags.mark_flag_as_required("runpath")
FLAGS = flags.FLAGS

# Some values and paths to be set
# user = FLAGS.username
project = "cfs"
# executable = FLAGS.pythonpath #previous version
executable = "python" # we are trying to just run python from the environment
run_file = "/home/x_rojon/cfs_numerics_v2024/src/run.py"
# run_file = FLAGS.runpath

num_gpus = 0
num_cpus = 2
mem_mb = 12000
# mem_mb = 64000
# max_runtime = "02-23:59:00"
max_runtime = "00-00:59:00"


def get_output_name(value_dict: Dict[Text, Any]) -> Text:
  """Get the name of the output directory."""
  name = ""
  for k, v in value_dict.items():
    name += f"-{k}_{v}"
  return name[1:]


def get_args(sweep: Sweep) -> List[Dict[Text, Any]]:
  """Convert a sweep dictionary into a list of dictionaries of arg settings."""
  values = list(sweep.values())
  args = list(product(*values))
  keys = list(sweep.keys())
  args = [{keys[i]: arg[i] for i in range(len(keys))} for arg in args]
  return args


def get_flag(key: Text, value: Any) -> Text:
  if isinstance(value, bool):
    return f' --{key}' if value else f' --no{key}'
  else:
    return f' --{key} {value}'


def submit_all_jobs(args: Sequence[Dict[Text, Any]], config) -> None:
  """Generate submit scripts and launch them."""
  # Base of the submit file
  base = list()
  base.append(f"#!/bin/bash")
  base.append("")
  base.append(f"#SBATCH -J {project}{'_gpu' if FLAGS.gpu else ''}")
  base.append(f"#SBATCH -c {num_cpus}")
  base.append(f"#SBATCH --mem={mem_mb}")
  base.append(f"#SBATCH -t {max_runtime}")
  base.append(f"#SBATCH --nice=10000")
  if FLAGS.gpu:
    base.append(f"#SBATCH -p gpu_p")
    base.append(f"#SBATCH --gres=gpu:{num_gpus}")
    base.append(f"#SBATCH --exclude=icb-gpusrv0[1-2]")  # keep for interactive
  else:
    # base.append(f"#SBATCH -p cpu_p")
  print(f"Run array of {FLAGS.n_runs} tasks for each setting...")
  base.append(f"#SBATCH --array=0-{FLAGS.n_runs - 1}")

  skipped_runs = 0
  for i, arg in enumerate(args):
    lines = deepcopy(base)
    output_name = get_output_name(arg)

    # Directory for slurm logs
    result_dir = os.path.join(FLAGS.result_dir, FLAGS.experiment_name)
    logs_dir = os.path.join(result_dir, output_name)

    # Create directories if non-existent (may be created by the program itself)
    run_setting = False
    if FLAGS.check_existing:
      for seed in range(FLAGS.n_runs):
        curdir = logs_dir + f'_{seed}'
        if os.path.exists(curdir):
          if not os.path.exists(os.path.join(curdir, "parameters_last.npz")):
            run_setting = True
        else:
          os.makedirs(curdir)
          run_setting = True
    else:
      run_setting = True
      for seed in range(FLAGS.n_runs):
        curdir = logs_dir + f'_{seed}'
        os.makedirs(curdir)

    if not run_setting:
      skipped_runs += 1
      print(f"Skipping {i} (total skipped: {skipped_runs})...")
      continue

    # The output, logs, and errors from running the scripts
    lines.append(f"#SBATCH -o {logs_dir}_%a/slurm.out")
    lines.append(f"#SBATCH -e {logs_dir}_%a/slurm.err")

    # Queue job
    lines.append("")
    runcmd = executable
    runcmd += " "
    runcmd += run_file
    runcmd += f' --output_dir {result_dir}'
    runcmd += f' --output_name {output_name}'
    runcmd += f' --try_seed_from_slurm'
    # Sweep arguments
    for k, v in arg.items():
      runcmd += get_flag(k, v)
    # Adaptive arguments (depending on sweep value)
    for adaptive_k, func in config["adaptive"].items():
      runcmd += get_flag(adaptive_k, func(arg))
    # Fixed arguments
    for k, v in config["fixed"].items():
      runcmd += get_flag(k, v)

    lines.append(runcmd)
    lines.append("")

    # Now dump the string into the `run_all.sub` file.
    with open("run_job.cmd", "w") as file:
      file.write("\n".join(lines))

    print(f"Submitting {i}, id: {output_name}...")
    os.system("sbatch run_job.cmd")

  print(f"\nSubmitted {len(args) - skipped_runs} / {len(args)}. "
        f"(Skipped: {skipped_runs})")


def main(_):
  """Initiate multiple runs."""
  if FLAGS.config is None:
    # No predefined config specified. One example config for a sweep below.
    config = {
      "sweep": {
        "m": [4, 8, 16],
        "seed": [1, 2, 3]
      },
      "fixed": {
        "n": 1,
        "f": 2,
        "jax_enable_x64": True,
        "checkpoint_freq": 1000,
        "sigma_weights": 0.01,
        "sigma_spectrum": 0.01,
        "opt_steps": 5000,
        "bfgs_maxiter": 3000,
        "lbfgs_maxiter": 10000,
        "lbfgs_maxcor": 70,
      },
      "adaptive": {
      }
    }
  else:
    config = configs.configurations[FLAGS.config]

  args = get_args(config["sweep"])
  n_jobs = len(args) * FLAGS.n_runs
  sweep_dir = os.path.join(FLAGS.result_dir, FLAGS.experiment_name)

  # Create directories if non-existent
  if not os.path.exists(sweep_dir):
    os.makedirs(sweep_dir)
  print(f"Store sweep dictionary to {sweep_dir}...")
  with open(os.path.join(sweep_dir, "sweep.json"), 'w') as fp:
    json.dump(config["sweep"], fp, indent=2)

  print(f"Generate all {len(args)} submit scripts a {FLAGS.n_runs} runs "
        f"for a total of {n_jobs} runs...")
  submit_all_jobs(args, config)

  print(f"DONE")


if __name__ == "__main__":
  app.run(main)
