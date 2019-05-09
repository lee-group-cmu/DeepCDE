import tensorflow as tf
import numpy as np

def normalize(cde_estimates, tol=1e-6, max_iter=200):
    """Normalizes conditional density estimates to be non-negative and
    integrate to one.

    Assumes densities are evaluated on the unit grid.

    :param cde_estimates: a numpy array or matrix of conditional density estimates.
    :param tol: float, the tolerance to accept for abs(area - 1).
    :param max_iter: int, the maximal number of search iterations.
    :returns: the normalized conditional density estimates.
    :rtype: numpy array or matrix.

    """
    if cde_estimates.ndim == 1:
        _normalize(cde_estimates, tol, max_iter)
    else:
        np.apply_along_axis(_normalize, 1, cde_estimates, tol=tol,
                            max_iter=max_iter)

def _normalize(density, tol=1e-6, max_iter=500):
    """Normalizes a density estimate to be non-negative and integrate to
    one.

    Assumes density is evaluated on the unit grid.

    :param density: a numpy array of density estimates.
    :param z_grid: an array, the grid points at the density is estimated.
    :param tol: float, the tolerance to accept for abs(area - 1).
    :param max_iter: int, the maximal number of search iterations.
    :returns: the normalized density estimate.
    :rtype: numpy array.

    """
    hi = np.max(density)
    lo = 0.0

    area = np.mean(np.maximum(density, 0.0))
    if area == 0.0:
        # replace with uniform if all negative density
        density[:] = 1.0
    elif area < 1:
        density /= area
        density[density < 0.0] = 0.0
        return

    for _ in range(max_iter):
        mid = (hi + lo) / 2
        area = np.mean(np.maximum(density - mid, 0.0))
        if abs(1.0 - area) <= tol:
            break
        if area < 1.0:
            hi = mid
        else:
            lo = mid

    # update in place
    density -= mid
    density[density < 0.0] = 0.0


def cde_loss(beta, z_basis, shrink_factor, name="cde_loss"):
    with tf.name_scope(name):
        complexity_terms = tf.reduce_sum(tf.square(beta), axis=1) + 1.0
        fit_terms = tf.reduce_sum(tf.multiply(beta, z_basis), axis=1) + 1.0
        loss = tf.reduce_mean(complexity_terms - 2 * fit_terms) / shrink_factor
        return loss

def nll_loss(beta, z_basis, shrink_factor, name="nll_loss"):
    with tf.name_scope(name):
        fit_terms = tf.reduce_sum(tf.multiply(beta, z_basis), axis=1) + 1.0
        loss = tf.reduce_mean(-fit_terms) / shrink_factor
        return loss

def cde_layer(inputs, weight_sd, marginal_beta, name="cde_layer"):
    with tf.name_scope(name):
        W_shape = [int(inputs.shape[1]), len(marginal_beta) - 1]
        W = tf.Variable(tf.random_normal(W_shape, stddev=weight_sd), name="W")
        b = tf.get_variable("b", initializer=marginal_beta[1:])
        beta = tf.matmul(inputs, W) + b
        # tf.summary.histogram("weights", W)
        # tf.summary.histogram("biases", b)
        # tf.summary.histogram("betas", beta)
        return beta

def cde_predict(sess, beta, z_min, z_max, z_grid, z_grid_basis, input_dict):
    beta = sess.run(beta, feed_dict=input_dict)
    n_obs = beta.shape[0]
    beta = np.hstack((np.ones((n_obs, 1)), beta))
    cdes = np.matmul(beta, z_grid_basis.T)
    normalize(cdes)
    cdes /= np.prod(z_max - z_min)
    return cdes

def box_transform(z, z_min, z_max):
    return (z - z_min) / (z_max - z_min)
