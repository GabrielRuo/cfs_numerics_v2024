"""The main driver file for a single optimization run."""
import json
import os
from collections import defaultdict
from datetime import datetime

import jax.numpy as jnp
from absl import app
from absl import flags
from absl import logging
from jax import random
# from jax.config import config
from jax import config

import utils

# ---------------------------- PHYSICAL PARAMETERS ----------------------------
flags.DEFINE_integer("n", 2, "The spin dimension.")
flags.DEFINE_integer("f", 4, "The number of particles.")
flags.DEFINE_integer("m", 128,
                     "The number of spacetime points in the support of the "
                     "minimizing measure.")
# ---------------------------- INITIALIZATION AND METHODS ---------------------
flags.DEFINE_float("sigma_weights", 0.01,
                   "Standard deviation of random noise added to initialization "
                   "of weights.")
flags.DEFINE_float("sigma_spectrum", 0.01,
                   "Standard deviation of random noise added to initialization "
                   "of positive and negative eigenvalues.")
flags.DEFINE_float("init_spectrum", 1,
                   "Initial absolute value of the mean of negative spectra."
                   "Will add 1/n to each of the positive one for trace = 1.")
# ---------------------------- (L)BFGS PARAMETERS -----------------------------
flags.DEFINE_integer("lbfgs_maxiter", 10_000,
                     "Maximum number of iterations for LBFGS.")
flags.DEFINE_float("lbfgs_gtol", 1e-7, "Gradient value tolerance for LBFGS.")
flags.DEFINE_float("lbfgs_ftol", 1e-9, "Function value tolerance for LBFGS.")
flags.DEFINE_integer("lbfgs_maxcor", 70,
                     "Max number of corrections for limited memory in LBFGS.")
flags.DEFINE_integer("lbfgs_maxls", 20, "Maximum linesearch steps for LBFGS.")
flags.DEFINE_integer("bfgs_maxiter", 5000,
                     "Maximum number of iterations for BFGS.")
flags.DEFINE_float("bfgs_gtol", 1e-7, "Gradient value tolerance for BFGS.")
flags.DEFINE_integer("max_m_bfgs", 1025,
                     "Only run BFGS when m is smaller than this number to "
                     "avoid out of memory errors. Full BFGS requires a lot of "
                     "memory.")
# ---------------------------- OUTPUT -----------------------------------------
flags.DEFINE_string("output_dir", "../results/",
                    "Path to the output directory (for results).")
flags.DEFINE_string("output_name", "",
                    "Name for result folder. Use timestamp if empty.")
flags.DEFINE_integer("checkpoint_freq", 1_000,
                     "Write out checkpoints with this frequency. "
                     "No checkpointing if this value is smaller or equal zero.")
# ---------------------------- MISC ------------------------------------------
flags.DEFINE_bool("try_seed_from_slurm", False,
                  "Try to get the seed from a slurm job array. If doesn't "
                  "exist, use `seed` flag.")
flags.DEFINE_integer("seed", 3421, "The random seed.")
FLAGS = flags.FLAGS


# =============================================================================
# MAIN
# =============================================================================

def main(_):
  # ---------------------------------------------------------------------------
  # Setup
  # ---------------------------------------------------------------------------
  FLAGS.alsologtostderr = True

  #measure time
  start_time = datetime.now()
  
  # Manage random seed
  output_name = FLAGS.output_name
  seed = FLAGS.seed
  if FLAGS.try_seed_from_slurm:
    logging.info("Trying to fetch seed from slurm environment variable...")
    slurm_seed = os.getenv("SLURM_ARRAY_TASK_ID")
    if slurm_seed is not None:
      logging.info(f"Found task id {slurm_seed}")
      seed = int(slurm_seed)
      output_name = f'{output_name}_{seed}'
      logging.info(f"Set output directory {output_name}")
      FLAGS.output_name = output_name
    else:
      raise RuntimeError("Did not find slurm environment variable for seed.")
  FLAGS.seed = seed
  logging.info(f"Set random seed {FLAGS.seed}...")
  key = random.PRNGKey(FLAGS.seed)

  # Setup output directory
  if FLAGS.output_name == "":
    dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  else:
    dir_name = FLAGS.output_name
  out_dir = os.path.join(os.path.abspath(FLAGS.output_dir), dir_name)
  logging.info(f"Save all output to {out_dir}...")
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  # Setup logging
  FLAGS.log_dir = out_dir
  logging.get_absl_handler().use_absl_log_file(program_name="run")

  logging.info("Save FLAGS (arguments)...")
  with open(os.path.join(out_dir, 'flags.json'), 'w') as fp:
    json.dump(FLAGS.flag_values_dict(), fp, sort_keys=True, indent=2)

  # Setup jax precision
  if FLAGS.jax_enable_x64:
    logging.info(f"Running with 64 bit floats.")
    dfloat = jnp.float64
  else:
    logging.info(f"Running with 32 bit floats.")
    dfloat = jnp.float32

  # ---------------------------------------------------------------------------
  # Parameter and result initialization
  # ---------------------------------------------------------------------------
  key, subkey = random.split(key)
  logging.info("Randomly initializing optimization parameters...")
  params_0 = utils.init_params(subkey, FLAGS.n, FLAGS.f, FLAGS.m,
                               FLAGS.sigma_weights, FLAGS.init_spectrum,
                               FLAGS.sigma_spectrum)

  # Count and log number of parameters
  param_names = [
    'weights', 'pos spectrum', 'neg spectrum', 'block ul', 'block_ur'
  ]
  for pname, pval in zip(param_names, params_0):
    logging.info(f'{pname}: {pval.shape}, {pval.dtype}')

  num_params = sum([param.size if param.dtype == dfloat else 2 * param.size
                    for param in params_0])
  logging.info(f'Total number of real optimization parameters: {num_params}')

  logging.info(f'Initialize results container...')
  results = defaultdict(list)
  results['n'] = FLAGS.n
  results['f'] = FLAGS.f
  results['m'] = FLAGS.m

  # ---------------------------------------------------------------------------
  # Setup (L)BFGS options
  # ---------------------------------------------------------------------------
  lbfgs_options = {
    "maxiter": FLAGS.lbfgs_maxiter,
    "maxfun": 2 * FLAGS.lbfgs_maxiter,
    "disp": 50,
    "gtol": FLAGS.lbfgs_gtol,
    "ftol": FLAGS.lbfgs_ftol,
    "maxcor": FLAGS.lbfgs_maxcor,
    "maxls": FLAGS.lbfgs_maxls,
  }
  logging.info(f"Run LBFGS for at most {FLAGS.lbfgs_maxiter} iterations.")
  logging.info(f"   Settings: {lbfgs_options}")

  if FLAGS.max_m_bfgs < FLAGS.m:
    bfgs_options = None
  else:
    bfgs_options = {
      "maxiter": FLAGS.bfgs_maxiter,
      "disp": True,
      "gtol": FLAGS.bfgs_gtol,
    }
    logging.info(f"Add BFGS for at most {FLAGS.bfgs_maxiter} iterations.")
    logging.info(f"   Settings: {bfgs_options}")

  # ---------------------------------------------------------------------------
  # Optimization and writing results
  # ---------------------------------------------------------------------------
  final_params, bfgs_res = utils.optimize(
    params_0, FLAGS.n, FLAGS.f, FLAGS.m, lbfgs_options, bfgs_options, out_dir,
    FLAGS.checkpoint_freq)
  logging.info(f"Store final results and parameters...")
  results.update(bfgs_res)
  utils.write_checkpoint(final_params, 'parameters_last', out_dir, results)

  elapsed_time = datetime.now() - start_time
  logging.info(f"DONE. The script took {elapsed_time} to run.")


if __name__ == "__main__":
  config.config_with_absl()
  app.run(main)
