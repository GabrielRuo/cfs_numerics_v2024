"""Collection of general utility functions."""
import os
from typing import Tuple, Text, Dict, Union, Any

import numpy as np
from absl import logging
from jax import random, vmap, jit, grad, numpy as jnp
from jax.nn import softmax, relu
# from jax.ops import index_update, index
#from jax.scipy.linalg import expm
from scipy.optimize import minimize
import optimistix

Params = Tuple[jnp.ndarray, ...]
Results = Dict[Text, Union[jnp.ndarray, list, float]]
Tup = Tuple[float, float]


# =============================================================================
# RESULT COLLECTION AND CHECKPOINTING
# =============================================================================

def collect_results(params: Params) -> Results:
  """Create results dictionary from parameters at a given step.

  Args:
    params: The parameters to add to the results dictionary.

  Returns:
    results dictionary
  """
  results = {}
  weights, pos_spectrum, neg_spectrum, alphas, betas = params
  n = pos_spectrum.shape[1]
  f = ((alphas.shape[1])//n+1+2*n)//2
  
  results['weights'] = softmax(weights)
  spectra = make_spectra(pos_spectrum, neg_spectrum)
  results['spectra'] = spectra
  results['alphas'] = alphas
  results['betas'] = betas
  results['avg_neg_spec'] = jnp.mean(spectra[:, 1])
  results['avg_pos_spec'] = jnp.mean(spectra[:, 0])
  results['eigenvectors'] = make_eigenvectors(results['alphas'], results['betas'], f, n)
  return {k: np.array(v) for k, v in results.items()}


def write_checkpoint(params: Params,
                     name: Text,
                     out_dir: Text,
                     results: Dict = None) -> None:
  """Checkpoint current optimization parameters.

  Args:
    params: The current parameter tuple.
    name: The name under which to store the parameter checkpoint.
    out_dir: The directory where to store the parameters.
    results: If this result dictionary is not `None`, add the parameters (in
        their interpretable form) to the result dictionary and store this as
        in the same directory as `results.npz`.
  """
  result_path = os.path.join(out_dir, f"{name}.npz")
  weights, pos_spectrum, neg_spectrum, alphas, betas = params
  np.savez(result_path, weights=weights, pos_spectrum=pos_spectrum,
           neg_spectrum=neg_spectrum, alphas=alphas, betas=betas)
  if results is not None:
    results.update(collect_results(params))
    result_path = os.path.join(out_dir, "results.npz")
    np.savez(result_path, **results)


# =============================================================================
# INITIALIZATION OF OPTIMIZATION PARAMETERS
# =============================================================================

def init_params(key: jnp.ndarray, n: int, f: int, m: int,
                sigma_weights: float = 0, init_spectrum: float = 1,
                sigma_spectrum: float = 0) -> Params:
  """Initialize optimization parameters.
  Args:
    key: The random key.
    n: The spin dimension.
    f: The number of particles.
    m: The cardinality of the support of the discrete measure.
    sigma_weights: The standard deviation of the noise to be added to the fixed
      initialization of weights.
    init_spectrum: The initial absolute value of the mean of the negative
      eigenvalues. We will add 1/n to this value for each of the positive
      eigenvalues to ensure trace = 1.
    sigma_spectrum: The standard deviation of the noise to be added to the fixed
      initialization of the spectra.

  Returns:
    a 5-tuple of initial parameters:
        weights: The weights of the discrete measure (m,); real
        pos_spectrum: initial log values of pos spectrum (m, n); real
        neg_spectrum: initial log values of pos spectrum (m, n); real
        alphas: initial values of the alpha angles used to define the unitary (m,n(2f-2n-1)): real
        betas: initial values of the beta angles used to define the unitary (m,n(2f-2n-1)): real
  """
  subkeys = random.split(key, 5)
  weights = sigma_weights * random.normal(subkeys[0], (m,))
  # Initial pos spectra are all (init_spectrum + 1 / n) and
  # negative ones are all (init_spectrum)
  # We work with logs of desired spectra to ensure the correct signs later
  pos_spectrum = jnp.log(init_spectrum + 1. / n)
  pos_spectrum += sigma_spectrum * random.normal(subkeys[1], (m, n))
  neg_spectrum = jnp.log(init_spectrum)
  neg_spectrum += sigma_spectrum * random.normal(subkeys[2], (m, n))
  alphas = random.uniform(subkeys[3],(m,n*(2*f-2*n-1)), minval = 0, maxval = 4*jnp.pi)
  betas = random.uniform(subkeys[4],(m,n*(2*f-2*n-1)), minval = 0, maxval = jnp.pi/2)
  return weights, pos_spectrum, neg_spectrum, alphas, betas


# =============================================================================
# STEPS TO COMPOSE ACTION (AND BOUNDEDNESS FUNCTIONAL) FROM PARAMETERS
# =============================================================================


def make_spectra(pos_spectrum: jnp.ndarray,
                 neg_spectrum: jnp.ndarray) -> jnp.ndarray:
  """Compute actual spectra from optimization parameters.

  The spectra have to have n positive and n negative eigenvalues
  and satisfy the trace constraint, which we ensure here.

  Args:
    pos_spectrum: Optimization parameters for positive eigenvalues.
    neg_spectrum: Optimization parameters for negative eigenvalues.

  Returns:
    Full (m, 2 n) array of the m spectra
  """
  spectra = jnp.concatenate((jnp.exp(pos_spectrum), - jnp.exp(neg_spectrum)), 1)
  return spectra / jnp.sum(spectra, axis=1)[..., jnp.newaxis]

def get_building_blocks(alphas:jnp.ndarray, betas:jnp.ndarray):
  """convert the arrays of angles into arrays of trigonometric functions which are "building blocks" of the band unitaries
    These"building blocks" are: the +-sin(beta) terms, the cos(beta)exp(ialpha) and the cos(beta)exp(-ialpha) terms

  Args:
    alphas: Values of the alpha angles used to define the unitary (m,n(2f-2n-1)): real
    betas: Values of the beta angles used to define the unitary (m,n(2f-2n-1)): real
  Returns:
  3 "building blocks" terms, basic trigonometric functions of the alphas and betas of the same size
  (m,2n(f-n)-n)
  """
  exp_alphas = jnp.exp(1J*alphas)
  cos_betas = jnp.cos(betas)
  sin_betas = jnp.sin(betas)
  cos_betas_exp_pos_alphas = exp_alphas*cos_betas
  cos_betas_exp_neg_alphas = jnp.conj(exp_alphas)* cos_betas

  return cos_betas_exp_pos_alphas, cos_betas_exp_neg_alphas, sin_betas

def make_masks(f:int,n_col:int, band_number:int):

  """construct the masks which will be used to build the band unitaries. 
  The masks are arrays of Booleans which will be used to select the correct terms in the building blocks.
  Each building block is associated with an array of masks: for example,  mask_sin is
  an array of masks for all the sin(beta_j)s: sin(beta_j) has a corresponding mask matrix with True where sin(beta_j) 
  is and False elsewhere. 
  The masks are constructed in such a way that the multiplication of the building blocks with the masks will give the correct terms in the band unitary



  Args:
    f: the number of particles
    n_col: the number of columns of the masks we want to build. To build a full matrix we will want n_col = f
    band_number: index of the band in the unitary decomposition (ranges from 1 to number of nonzero eigenvalues)
  Returns:
  3 arrays of masks of Booleans (True for 1, False for 0) of shape (f-band_number,f,n_col)
  """

  term_index = jnp.arange(f-band_number)[:,jnp.newaxis, jnp.newaxis]
  row_index = jnp.arange(f)[jnp.newaxis,:, jnp.newaxis]
  col_index = jnp.arange(n_col)[jnp.newaxis,jnp.newaxis,:]

  #conditions
  mask_cos_exp_pos = (row_index == term_index + band_number-1) & (band_number-1 <= col_index) & (col_index <= row_index)
  mask_cos_exp_neg = (col_index == term_index + band_number) & (row_index >= col_index)
  mask_sin = ((band_number-1 <= col_index) & (col_index <= term_index + band_number-1) & (row_index >= term_index + band_number)|
((row_index == term_index + band_number-1) & (col_index == row_index +1)))

  return mask_cos_exp_pos, mask_cos_exp_neg,mask_sin

def make_single_band_unitary(alphas_band: jnp.ndarray,betas_band: jnp.ndarray,f:int,n_col:int):
  """Use the angle parameters and the masks to generate a unitary band matrix

    Args:

    alphas_band: shape (f-band_number,)
    betas_band: shape (f-band_number,)
    f: dimension of the matrix (number of  particles)
    n_col: number of columns of the unitary which we want to build

  Returns:
    n_col first columns of band  unitary matrix (f, n_col)
    """
  band_number = f - len(alphas_band)

  #extract the building blocks and masks
  building_blocks = get_building_blocks(alphas_band, betas_band)
  masks = make_masks(f,n_col,band_number)

  #initialise the band matrix with a matrix with ones on lower triangle and superdiagonal
  ones_tril= jnp.tril(jnp.ones((f,f))) + jnp.eye(f,k = 1)
  ones_tril = ones_tril[:,:n_col]
  band_matrix= ones_tril.copy()

  #iterate over the different masks for each building block
  num_masks = len(masks)
  for building_block_index in range(num_masks):
    mask = masks[building_block_index] #shape (f-band_number,f,f)
    building_block = building_blocks[building_block_index]
    band_matrix_building_block = mask*building_block[:,jnp.newaxis,jnp.newaxis]

    #add ones in the lower triangle and superdiagonal before multiplying  the matrices
    band_matrix_building_block += ones_tril - mask

    #multiply the matrices together
    band_matrix_building_block = jnp.prod(band_matrix_building_block, axis =0)

    #multiply the different  building blocks together
    band_matrix *=band_matrix_building_block

  #multiply by a final mask
  final_mask = jnp.tril(jnp.concatenate((jnp.zeros((f, band_number-1)), jnp.ones((f, f - band_number+1))), axis=1), k=-1)# zeros in first band_number-1  columns
  final_mask += jnp.eye(f) #add ones on the diagonal
  super_diagonal_terms = jnp.concatenate((jnp.zeros(band_number-1),jnp.ones(f-band_number)))
  final_mask -= jnp.diag(super_diagonal_terms, k = 1) #-1 on the superdiagonal to represent  the negative  sines
  band_matrix *= final_mask[:,:n_col] #shape (f,n_col)

  return band_matrix

def make_single_eigenvectors(alphas: jnp.ndarray ,betas:jnp.ndarray,f:int, n:int):
  """Use the angle parameters to build the 2n first eigenvectors of a spacetime point  x

    Args:
    f: dimension of the matrix (number of  particles)
    n: spin number hence  2n is total number of eigenvalues
    alphas: shape n(2f-2n-1)
    betas: shape n(2f-2n-1)

  Returns:
    matrix (f,2n) of the 2n first eigenvectors of x
    """
  #extract all the parameters
  num_alphas = len(alphas)
  building_blocks = get_building_blocks(alphas, betas)

  #initialise the eigenvectors:
  start_index = len(alphas)-(f-2*n)
  end_index = len(alphas)
  alphas_band, betas_band = alphas[start_index:end_index],betas[start_index:end_index]
  eigenvectors = make_single_band_unitary(alphas_band,betas_band,f,n_col = 2*n)

  #iterate over the remaining 2n-1 bands
  for band_number in range(2*n-1,0,-1):
    #extract the correct alphas and betas
    end_index = start_index
    start_index = end_index - (f - band_number)
    alphas_band, betas_band = alphas[start_index:end_index],betas[start_index:end_index]

    #build the band unitary
    band_matrix = make_single_band_unitary(alphas_band,betas_band,f,n_col = f)

    #multiply the unitaries and the vectors
    eigenvectors = jnp.dot(band_matrix,eigenvectors)

  return eigenvectors
#vectorize
make_eigenvectors = vmap(make_single_eigenvectors, in_axes=(0, 0, None,None))

def make_lagrangian_1(spectra: jnp.ndarray,eigenvectors:jnp.ndarray, i: int, j: int) -> float:
  """The Lagrangian for a single pair of spacetime points for n = 1.

  Args:
    spectra: (m,2n)
    eigenvectors: (m,f,2n)
    i: Index for first point.
    j: Index for second point.

  Returns:
    value of the Lagrangian
  """
  gram = jnp.dot(jnp.conj(eigenvectors[i].T),eigenvectors[j])
  eigenvalue_products = jnp.outer(spectra[i],spectra[j])
  xy_product = jnp.dot(gram *eigenvalue_products, jnp.conj(gram.T)) #not exactly xy but an isospectral matrix, M in write up
  D = 0.5*(jnp.real(xy_product[0,0])-jnp.real(xy_product[1,1]))**2 + 2*jnp.real(xy_product[0,1]*xy_product[1,0])
  return relu(D)

def make_lagrangian_n(spectra: jnp.ndarray,eigenvectors: jnp.ndarray, i: int, j: int) -> float:
  """The Lagrangian for a single pair of spacetime points for any n

  Args:
    spectra: (m,2n)
    eigenvectors: (m,f,2n)
    i: Index for first point.
    j: Index for second point.

  Returns:
    value of the Lagrangian
  """
  gram = jnp.dot(jnp.conj(eigenvectors[i].T),eigenvectors[j])
  two_n = gram.shape[0]
  eigenvalue_products = jnp.outer(spectra[i],spectra[j])
  xy_product = jnp.dot(gram *eigenvalue_products, jnp.conj(gram.T)) #not exactly xy but an isospectral matrix, M in write up

  spec = jnp.sort(jnp.abs(jnp.linalg.eigvals(xy_product)))
  bnd = jnp.sum(spec) ** 2

  return jnp.sum(spec ** 2) - bnd / (two_n)


def action(params: Params) -> float:
  """Constructs the  action functional from the parameters. 
  Considers contributions from all spacetime  points which are defined by the params

  Args:
    params: The 5-tuple of parameters (weights, positive spectrum,
        negative spectrum, alphas, betas).

  Returns:
    single float for the value of the action
  """
  weights, pos_spectrum, neg_spectrum, alphas, betas = params
  weights = softmax(weights)
  spectra = make_spectra(pos_spectrum, neg_spectrum)
  m, n = pos_spectrum.shape
  f = ((alphas.shape[1])//n+1+2*n)//2

  eigenvectors = make_eigenvectors(alphas, betas, f, n)
  
  # weighted sum of Lagrangian for pairs

  if n == 1:
    make_lag = vmap(make_lagrangian_1, (None, None, 0, 0))
  else:
    make_lag = vmap(make_lagrangian_n, (None, None, 0, 0))

  # Only looking at upper triangle (without diagonal)
  rows, cols = jnp.triu_indices(m, k=1)
  lag_ij = make_lag(spectra,eigenvectors, rows, cols)
  act = 2 * jnp.sum(weights[rows] * weights[cols] * lag_ij)
  # Add diagonal
  diag = jnp.arange(m)
  lag_ij = make_lag(spectra,eigenvectors, diag, diag)
  act += jnp.sum(weights ** 2 * lag_ij)
  return act

def boundedness_summand(spectra: jnp.ndarray,eigenvectors: jnp.ndarray, i: int, j: int) -> float:
  """The summand involved in the computation of the boundary functional, for a given single pair of spacetime points
  Args:
    spectra: (m,2n)
    eigenvectors: (m,f,2n)
    i: Index for first point.
    j: Index for second point.

  Returns:
    value of the summand
  """
  gram = jnp.dot(jnp.conj(eigenvectors[i].T),eigenvectors[j])
  eigenvalue_products = jnp.outer(spectra[i],spectra[j])
  xy_product = jnp.dot(gram *eigenvalue_products, jnp.conj(gram.T)) #not exactly xy but an isospectral matrix, M in write up
  spec = jnp.sort(jnp.abs(jnp.linalg.eigvals(xy_product)))
  _bnd = jnp.sum(spec) ** 2
  return _bnd

def boundedness(params: Params) -> float:
  """The boundedness functional. Computes the weighted sum of the summands for all pairs of spacetime points

  Args:
    params: The 5-tuple of parameters (weights, positive spectrum,
        negative spectrum, alphas, betas).

  Returns:
    single float for the value of the action
  """
  weights, pos_spectrum, neg_spectrum, alphas, betas = params
  weights = softmax(weights)
  spectra = make_spectra(pos_spectrum, neg_spectrum)
  m, n = pos_spectrum.shape
  f = ((alphas.shape[1])//n+1+2*n)//2

  eigenvectors = make_eigenvectors(alphas, betas, f, n)

  make_bnd = vmap(boundedness_summand, (None, None, 0, 0))

  # Only looking at upper triangle (without diagonal)
  rows, cols = jnp.triu_indices(m, k=1)
  bnd_ij = make_bnd(spectra,eigenvectors, rows, cols)
  bnd = 2 * jnp.sum(weights[rows] * weights[cols] * bnd_ij)
  # Add diagonal
  diag = jnp.arange(m)
  bnd_ij = make_bnd(spectra,eigenvectors, diag, diag)
  bnd += jnp.sum(weights ** 2 * bnd_ij)
  return bnd

# =============================================================================
# OPTIMIZATION
# =============================================================================

def _flatten_params(params: Params) -> jnp.ndarray:
  """Flatten all (complex) optimization parameters into single (real) vector."""
  all_params = []
  for p in params:
      all_params.append(p.ravel())
  return jnp.concatenate(all_params)


def _reconstruct_params(params: jnp.ndarray, n: int, f: int, m: int) -> Params:
  """Rearrange parameters into original shape from flat (real) vector.

  Returns:
    a 5-tuple of initial parameters:
        weights: The weights of the discrete measure (m,); real
        pos_spectrum: initial log values of pos spectrum (m, n); real
        neg_spectrum: initial log values of pos spectrum (m, n); real
        alphas: angle parameters for band unitaries (m, n(2f-2n-1)); real
        betas: angle parameters for band unitaries (m, n (2f-2n-1)); real
  """
  n_w = m
  n_pos = n_w + n * m
  n_neg = n_pos + n * m
  n_alphas = n_neg + m * (n*(2*f-2*n-1))
  splits = [n_w, n_pos, n_neg, n_alphas]
  (weights,
   pos_spectrum,
   neg_spectrum,
   alphas,
   betas) = jnp.split(params, splits)
  pos_spectrum = pos_spectrum.reshape(m, n)
  neg_spectrum = neg_spectrum.reshape(m, n)
  alphas = alphas.reshape(m,n*(2*f-2*n-1))
  betas = betas.reshape(m,n*(2*f-2*n-1))
  return weights, pos_spectrum, neg_spectrum, alphas, betas


def _action_flat_params(params: jnp.ndarray, n: int, f: int, m: int) -> float:
  """Action computation for bfgs optimization."""
  params = _reconstruct_params(params, n, f, m)
  return action(params)


def optimize_scipy(params: Params,
             n: int,
             f: int,
             m: int,
             lbfgs_options: Dict[Text, Any],
             bfgs_options: Dict[Text, Any],
             out_dir: Text,
             checkpoint_freq: int) -> Tuple[Params, Results]:
  """Wrapper around the scipy BFGS minimizer that also collects results.

  Args:
    params: The tuple of optimization parameters.
    n: The desired spin dimension.
    f: The desired number of particles.
    m: The desired cardinality of the support of the discrete measure.
    lbfgs_options: The options to pass to L-BFGS. If the argument is None or the
        maxiter option is 0, don't run L-BFGS at all.
    bfgs_options: The options to pass to BFGS. If the argument is None or the
        maxiter option is 0, don't run BFGS at all.
    out_dir: Where to write results.
    checkpoint_freq: Frequency of parameter and result checkpoints.
  """

  if ((bfgs_options is None or bfgs_options['maxiter'] <= 0) and
      (lbfgs_options is None or lbfgs_options['maxiter'] <= 0)):
    raise ValueError("Run either bfgs or lbfgs for at least 1 iteration.")

  results = {
    'action': [],
    'boundedness': [],
    'n_iterations': [],
    'n': n,
    'f': f,
    'm': m
  }

  # ---------------------------------------------------------------------------
  # Callback wrapper class to keep track of number of iterations
  # ---------------------------------------------------------------------------
  class Callback:

    def __init__(self):
      self.n_iter = 1

    def callback(self, xk):
      if self.n_iter % checkpoint_freq == 0:
        cur_params = _reconstruct_params(xk, n, f, m)
        cur_act = action(cur_params)
        cur_bnd = boundedness(cur_params)
        logging.info(f"{self.n_iter}: action: {cur_act:.5f}    "
                     f"bound.: {cur_bnd:.5f}")
        results['action'].append(cur_act)
        results['boundedness'].append(cur_bnd)
        results['n_iterations'].append(self.n_iter)
        write_checkpoint(
          cur_params, f'parameters_{self.n_iter}', out_dir, results)
      self.n_iter += 1

  # ---------------------------------------------------------------------------
  # Logging and result update helper
  # ---------------------------------------------------------------------------
  def _log_and_update_results(method: Text,
                              cur_res: Any,
                              existing_res: Results) -> Results:
    logging.info(f"{method}: Success: {cur_res.success}")
    logging.info(f"{method}: Status: {cur_res.status}")
    logging.info(f"{method}: Message: {cur_res.message}")
    logging.info(f"{method}: Iterations: {cur_res.nit}")
    logging.info(f"{method}: func_eval: {cur_res.nfev}")
    logging.info(f"{method}: jac_eval: {cur_res.njev}")
    logging.info(f"{method}: Achieved action: {cur_res.fun}")

    method = method.lower()
    return dict(existing_res, **{f"{method}_iterations": cur_res.nit,
                                 f"{method}_func_eval": cur_res.nfev,
                                 f"{method}_jac_eval": cur_res.njev,
                                 f"{method}_status": cur_res.status,
                                 f"{method}_success": cur_res.success})

  # ---------------------------------------------------------------------------
  # Initialization and setup
  # ---------------------------------------------------------------------------
  params0 = _flatten_params(params)

  # ---------------------------------------------------------------------------
  # L-BFGS (scipy)
  # ---------------------------------------------------------------------------
  f_act = jit(_action_flat_params, static_argnums=(1, 2, 3))
  func_grad = jit(grad(_action_flat_params), static_argnums=(1, 2, 3))
  g_act = lambda x, _n, _f, _m: np.array(func_grad(x, _n, _f, _m))
  callback = Callback()
  res = None
  if lbfgs_options is not None and lbfgs_options['maxiter'] > 0:
    logging.info(f"Starting scipy L-BFGS")
    res = minimize(f_act, params0, args=(n, f, m), method="l-bfgs-b", jac=g_act,
                   callback=callback.callback, options=lbfgs_options)
    results = _log_and_update_results("LBFGS", res, results)
  else:
    logging.info(f"Skipping L-BFGS...")

  # ---------------------------------------------------------------------------
  # BFGS scipy
  # ---------------------------------------------------------------------------
  if bfgs_options is not None and bfgs_options['maxiter'] > 0:
    logging.info(f"Starting BFGS")
    if res is not None:
      params0 = res.x
    res = minimize(f_act, params0, args=(n, f, m), method="bfgs", jac=g_act,
                   callback=callback.callback, options=bfgs_options)
    results = _log_and_update_results("BFGS", res, results)
  else:
    logging.info(f"Skipping BFGS...")

  # ---------------------------------------------------------------------------
  # Final results reporting
  # ---------------------------------------------------------------------------
  final_params = _reconstruct_params(res.x, n, f, m)
  act = action(final_params)
  bnd = boundedness(final_params)
  logging.info(f"Achieved boundedness: {bnd}")
  if not np.isclose(act, res.fun):
    logging.warning("Final action from minimize and manual do not agree ",
                    f'minimize: {res.fun}, manually: {act}')
    logging.warning(f"Using manually compute action {act}")
  results['action'].append(act)
  results['boundedness'].append(bnd)
  return final_params, {k: np.array(v) for k, v in results.items()}

#--------------------------------------------
#Optimization with optimistix, both unconstrained and constrained 
#--------------------------------------------

#Constrained optimization

def action_with_barrier_bnd_constraint(params:Params, bnd_constraint, k):
  """
  params are the original params without kkt multiplier or slack variable
  """
  action = action(params)
  bnd = boundedness(params)
  barrier = -(1/k)*jnp.log(-(bnd-bnd_constraint))

  return action + barrier

def action_with_barrier_flat_params(params: jnp.ndarray, n: int, f: int, m: int, bnd_constraint, k) -> float:
  """Action computation for bfgs optimization."""
  params = _reconstruct_params(params, n, f, m)
  return action_with_barrier_bnd_constraint(params, bnd_constraint, k)

def feasibility_cost(params,bnd_constraint):
  bnd = boundedness(params)
  return relu(bnd-bnd_constraint)

def _feasibility_cost_flat_params(params,n,f,m,bnd_constraint):
  params = _reconstruct_params(params, n, f, m)
  return feasibility_cost(params,bnd_constraint)

class Optimistix_BFGS_Solver():
  """
  Define 3 solvers, one to solve the initial feasibility, one for unconstrained optimisation, one for constrained optimisation

  """

  def __init__(self, max_iter, rtol, atol):
        self.max_iter = max_iter
        self.rtol = rtol
        self.atol = atol

  @staticmethod
  def _feasibility_cost_with_args(params, args):
      n,f,m,bnd_constraint = args
      return _feasibility_cost_flat_params(params, n, f, m, bnd_constraint)

  @staticmethod
  def _action_with_args(params, args):
      n,f,m = args
      return _action_flat_params(params, n, f, m)
  @staticmethod

  def _action_with_barrier_and_args(params, args):
      n,f,m, bnd_constraint, k = args
      return action_with_barrier_flat_params(params, n, f, m, bnd_constraint, k)

  def _optimize(self, objective_function, params_0, args):
      """
      Helper method to set up and run the optimizer.

      Parameters:
      objective_function: The function to minimize
      params_0: Initial parameters
      args: Arguments to pass to the objective function

      Returns:
      Optimized parameters after minimization
      """
      f_act = jit(objective_function, static_argnums=1)

      solver = optimistix.BFGS(rtol=self.rtol, atol=self.atol)
      solution = optimistix.minimise(
          fn=f_act,
          solver=solver,
          y0=params_0,
          args=args,
          max_steps=self.max_iter,
          throw=True
      )
      return solution

  def satisfy_feasibility(self, params_0, n, f, m, bnd_constraint):
      """
      Find initial feasible parameters.

      Parameters:
      params_0: Initial parameters
      n, f, m, bnd_constraint: Parameters required for the feasibility function

      Returns:
      Feasible parameters after optimization
      """
      return self._optimize(
          self._feasibility_cost_with_args,
          params_0,
          (n, f, m, bnd_constraint)
      )

  def minimise_unconstrained_action(self, params_0, n, f, m):
      """
      Minimize the action function.

      Parameters:
      params_0: Initial parameters
      n, f, m: Parameters required for the action function

      Returns:
      Optimized parameters after minimization
      """
      return self._optimize(
          self._action_with_args,
          params_0,
          (n, f, m)
      )
  def minimise_constrained_action(self, params_0, n, f, m, bnd_constraint, k):
      """
      Minimize the action function under the boundedness constraint

      Parameters:
      params_0: Initial parameters
      n, f, m: Parameters required for the action function

      Returns:
      Optimized parameters after minimization
      """
      return self._optimize(
          self._action_with_barrier_and_args,
          params_0,
          (n, f, m, bnd_constraint, k)
      )
  

def optimize_optimistix(params: Params,
            n: int,
            f: int,
            m: int,
            max_iter: int, 
            rtol: int, 
            atol:int,
            out_dir: Text,
            bnd_constraint = None) -> Tuple[Params, Results]:
  """Wrapper around the scipy BFGS minimizer that also collects results.

  Args:
    params: The tuple of optimization parameters.
    n: The desired spin dimension.
    f: The desired number of particles.
    m: The desired cardinality of the support of the discrete measure.
    lbfgs_options: The options to pass to L-BFGS. If the argument is None or the
        maxiter option is 0, don't run L-BFGS at all.
    bfgs_options: The options to pass to BFGS. If the argument is None or the
        maxiter option is 0, don't run BFGS at all.
    out_dir: Where to write results.
    checkpoint_freq: Frequency of parameter and result checkpoints.
  """
  solver = Optimistix_BFGS_Solver(max_iter, rtol, atol)

  k = -10**3*jnp.log(0.1)*m*n**3
  results = {
    'action': [],
    'boundedness': [],
    'n_iterations': [],
    'n': n,
    'f': f,
    'm': m
  }

  def run_constrained_optimisation(n,f,m,bnd_constraint,k):
    # satisfy feasibility
    solution = solver.satisfy_feasibility(params, n, f, m, bnd_constraint)
    params = solution.value
    # minimise constrained action
    solution = solver.minimise_constrained_action(params,n,f,m,bnd_constraint,k)
    return solution
  
  def run_unconstrained_optimisation_optimistix(n,f,m):
    solution = solver.minimise_unconstrained_action(params, n, f, m)
    return solution

  if bnd_constraint == None: 
    solution = run_unconstrained_optimisation_optimistix(params,n,f,m)
  else:
    solution = run_constrained_optimisation(params,n,f,m,bnd_constraint)

  final_params = _reconstruct_params(solution.value, n, f, m)
  act = action(final_params)
  bnd = boundedness(final_params)
  num_iter = int(solution.stats['num_steps'])

  results['action'].append(act)
  results['boundedness'].append(bnd)
  results['n_iterations'].append(num_iter)

  return final_params, {k: np.array(v) for k, v in results.items()}


  

  
