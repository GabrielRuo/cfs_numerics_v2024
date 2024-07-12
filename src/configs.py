"""Configurations to reproduce sweeps and results in paper."""
import analytical_results

configurations = {
  "gabs_test": {
    "sweep": {
      "m": [256],
      "f": [6, 8, 12, 16, 24, 32]
    },
    "fixed": {
      "n": 1,
      "jax_enable_x64": True,
      "checkpoint_freq": 1000,
      "sigma_weights": 0.01,
      "sigma_spectrum": 0.01,
      "lbfgs_maxiter": 10000,
      "lbfgs_maxcor": 70,
      "lbfgs_gtol": 1e-7,
      "lbfgs_ftol": 1e-9,
      "bfgs_maxiter": 5000,
      "bfgs_gtol": 1e-7,
      "max_m_bfgs": 1000,
      "seed": 543,
    },
    "adaptive": {}
  },
  "n1_large_f": {
    "sweep": {
      "m": [6, 8, 10, 12, 16, 24, 32, 64, 96, 128],
      "f": [2, 3, 4, 6, 8, 12, 16, 24, 32, 40]
    },
    "fixed": {
      "n": 1,
      "jax_enable_x64": True,
      "checkpoint_freq": 1000,
      "sigma_weights": 0.01,
      "sigma_spectrum": 0.01,
      "lbfgs_maxiter": 10000,
      "lbfgs_maxcor": 70,
      "lbfgs_gtol": 1e-7,
      "lbfgs_ftol": 1e-9,
      "bfgs_maxiter": 5000,
      "bfgs_gtol": 1e-7,
      "max_m_bfgs": 1000,
      "seed": 543,
    },
    "adaptive": {}
  },
  "n2_large_f": {
    "sweep": {
      "m": [6, 8, 10, 12, 16, 24, 32, 40],
      "f": [4, 8, 12, 16, 24, 32, 64, 128]
    },
    "fixed": {
      "n": 2,
      "jax_enable_x64": True,
      "checkpoint_freq": 1000,
      "sigma_weights": 0.01,
      "sigma_spectrum": 0.01,
      "lbfgs_maxiter": 10000,
      "lbfgs_maxcor": 70,
      "lbfgs_gtol": 1e-7,
      "lbfgs_ftol": 1e-9,
      "bfgs_maxiter": 5000,
      "bfgs_gtol": 1e-7,
      "max_m_bfgs": 1000,
      "seed": 543,
    },
    "adaptive": {}
  },
  "n1_f2": {
    "sweep": {
      "m": [4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024],
    },
    "fixed": {
      "n": 1,
      "f": 2,
      "jax_enable_x64": True,
      "checkpoint_freq": 1000,
      "sigma_weights": 0.01,
      "sigma_spectrum": 0.01,
      "lbfgs_maxiter": 10000,
      "lbfgs_maxcor": 70,
      "lbfgs_gtol": 1e-7,
      "lbfgs_ftol": 1e-9,
      "bfgs_maxiter": 5000,
      "bfgs_gtol": 1e-7,
      "max_m_bfgs": 1000,
      "seed": 543,
    },
    "adaptive": {
      "init_spectrum": lambda arg: 5 * analytical_results.n1_f2_neg_spectrum_from_m(arg['m']),
    }
  },
  "n1_f3": {
    "sweep": {
      "m": [8, 16, 32, 64, 128, 256, 384, 512, 768, 1024],
      "init_spectrum": [0.1, 1., 10., 50., 100., 500.],
    },
    "fixed": {
      "n": 1,
      "f": 3,
      "jax_enable_x64": True,
      "checkpoint_freq": 1000,
      "sigma_weights": 0.01,
      "sigma_spectrum": 0.01,
      "lbfgs_maxiter": 10000,
      "lbfgs_maxcor": 70,
      "lbfgs_gtol": 1e-7,
      "lbfgs_ftol": 1e-9,
      "bfgs_maxiter": 5000,
      "bfgs_gtol": 1e-7,
      "max_m_bfgs": 1000,
      "seed": 543,
    },
    "adaptive": {}
  },
  "n1_f4": {
    "sweep": {
      "m": [8, 16, 32, 64, 128, 256, 384, 512, 768, 1024],
    },
    "fixed": {
      "n": 1,
      "f": 4,
      "jax_enable_x64": True,
      "checkpoint_freq": 1000,
      "sigma_weights": 0.01,
      "sigma_spectrum": 0.01,
      "lbfgs_maxiter": 10000,
      "lbfgs_maxcor": 70,
      "lbfgs_gtol": 1e-7,
      "lbfgs_ftol": 1e-9,
      "bfgs_maxiter": 5000,
      "bfgs_gtol": 1e-7,
      "max_m_bfgs": 1000,
      "seed": 543,
      "init_spectrum": 15.,
    },
    "adaptive": {}
  },
  "n2_f4": {
    "sweep": {
      "m": [4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024],
    },
    "fixed": {
      "n": 2,
      "f": 4,
      "jax_enable_x64": True,
      "checkpoint_freq": 1000,
      "sigma_weights": 0.01,
      "sigma_spectrum": 0.01,
      "lbfgs_maxiter": 10000,
      "lbfgs_maxcor": 70,
      "lbfgs_gtol": 1e-7,
      "lbfgs_ftol": 1e-9,
      "bfgs_maxiter": 5000,
      "bfgs_gtol": 1e-7,
      "max_m_bfgs": 1000,
      "seed": 543,
    },
    "adaptive": {
      "init_spectrum": lambda arg: 5 * analytical_results.n2_f4_neg_spectrum_from_m(arg['m']),
    }
  }
}
