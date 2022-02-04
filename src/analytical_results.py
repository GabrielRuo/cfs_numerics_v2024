"""Some analytical results for checks and comparisons."""
from jax import numpy as jnp


def asymptotic_minimum(n):
  """Asymptotic minimum where only self-contributions are left."""
  if n == 1:
    return 1 / 6
  elif n == 2:
    return 5 / 256
  else:
    raise NotImplementedError("Analytic values only known for n=1,2.")


def n1_large_f_action_from_m(m):
  """The minimum action for the f to infinity limit for n=1."""
  return 1 / (2 * m)


def n2_large_f_action_from_m(m):
  """The minimum action for the f to infinity limit for n=2."""
  return 1 / (16 * m)


def large_f_action_from_m(m, n):
  return n1_large_f_action_from_m(m) if n == 1 else n2_large_f_action_from_m(m)


def n1_f2_neg_spectrum_from_m(m):
  """In the n=1, f=2 scenario, get the abs value of the neg eigenvalue at opt
  for 2-dim sphere.

  For the case n=1, f=2 *without* the boundedness constraint, the optimal
  solution is to send pos and neg eigenvalues to pos / neg infinity as m goes to
  infinity.
  We can compute these values for the positive and negative eigenvalues
  analytically.

  Returns:
    the absolute value of the negative eigenvalue at the optimum.
  """
  return (3 ** (1. / 4.) * jnp.sqrt(m / (2 * jnp.pi)) - 1.) / 4


def n2_f4_neg_spectrum_from_m(m):
  """In the n=2, f=4 scenario, get the abs value of the neg eigenvalue at opt
  for 4-dim sphere.

  For the case n=2, f=4 *without* the boundedness constraint, the optimal
  solution is to send pos and neg eigenvalues to pos / neg infinity as m goes to
  infinity.
  We can compute these values for the positive and negative eigenvalues
  analytically.

  Returns:
    the absolute value of the negative eigenvalue at the optimum.
  """
  return ((3. * m) ** (1. / 4.) / jnp.sqrt(jnp.pi) - 1.) / 4


def n1_f2_min_action_from_m(m):
  """In the n=1, f=2 scenario, get the min action.

  For the case n=1, f=2 *without* the boundedness constraint, the optimal
  solution is also a fixed constant value.
  We can compute the minimum action analytically.

  Returns:
    the optimal action
  """
  return jnp.ones_like(m) * jnp.sqrt(3) / (4 * jnp.pi)


def n2_f4_min_action_from_m(m):
  """In the n=2, f=4 scenario, get the min action.

  For the case n=2, f=4 *without* the boundedness constraint, the optimal
  solution is to send pos and neg eigenvalues to pos / neg infinity as m goes to
  infinity.
  We can compute the minimum action analytically.

  Returns:
    the optimal action
  """
  return jnp.sqrt(3) / (16 * jnp.pi * jnp.sqrt(m))


def n1_f2_max_bnd_from_m(m):
  """In the n=1, f=2 scenario, get the the boundedness constraint at the opt.

  For the case n=1, f=2 *without* the boundedness constraint, the optimal
  solution is to send pos and neg eigenvalues to pos / neg infinity as m goes to
  infinity.
  We can compute the boundedness constraint val at minimum action analytically.

  Returns:
    the boundedness value
  """
  return (3 * m ** 2 / (16 * jnp.pi ** 2) -
          jnp.sqrt(3) * m / (4 * jnp.pi) +
          1 / 4 + jnp.sqrt(3) / (2 * jnp.pi))


def n2_f4_max_bnd_from_m(m):
  """In the n=2, f=4 scenario, get the the boundedness constraint at the opt.

  For the case n=2, f=4 *without* the boundedness constraint, the optimal
  solution is to send pos and neg eigenvalues to pos / neg infinity as m goes to
  infinity.
  We can compute the boundedness constraint val at minimum action analytically.

  Returns:
    the boundedness value
  """
  return 3 * m / (16 * jnp.pi ** 2)


def total_num_params(n: int, f: int, m: int, fix_weights: bool) -> int:
  """Compute the total number of real optimization parameters."""
  num_params = m * (4 * f * n + 2 * n)
  return num_params if fix_weights else num_params + m
