import numpy as np

class TensorBasis(object):
  """
  Combines single-dimensional basis functions \phi_{d}(z) to form
  orthogonal tensor basis $\phi(z_{1}, \dots, z_{D}) = \prod_{d}
  \phi_{d}(z_{d})$.
  """
  def __init__(self, base, n_basis):
    self.base = base
    self.n_basis = n_basis

  def evaluate(self, z_test):
    """Evaluates tensor basis."""

    n_obs, n_dim = z_test.shape
    assert len(self.n_basis) == n_dim

    basis = np.ones((n_obs, np.prod(self.n_basis)))
    period = 1
    for dim in range(n_dim):

      sub_basis = self.base(self.n_basis[dim]).evaluate(z_test[:, dim])
      col = 0
      for _ in range(np.prod(self.n_basis) // (self.n_basis[dim] * period)):
        for sub_col in range(self.n_basis[dim]):
          for _ in range(period):
            basis[:, col] *= sub_basis[:, sub_col]
            col += 1
      period *= self.n_basis[dim]
    return basis
