import numpy as np


def normalize(cde_estimates, tol=1e-6, max_iter=10000):
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

    if max_iter <= 0:
        raise ValueError("max_iter needs to be positive. Currently %s" % (max_iter))

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


def box_transform(z, z_min, z_max):
    """
    Standardazing between 0 and 1 by subtracting the minimum and dividing by the range.
    :param z: a numpy array
    :param z_min: minimum of the numpy array/minimum to be used for standardization
        (has to be smaller or equal to the minimum of the numpy array)
    :param z_max: maximum of the numpy array/minimum to be used for standardization
        (has to be larger or equal to the minimum of the numpy array)
    :return: a normalized numpy array with same shape as input z
    """
    if np.min(z) < z_min or np.max(z) > z_max:
        raise ValueError('Passed minimum and maximum need to be outside the array range.'
                         'Current min: %s, max: %s, while array range: (%s,%s)' % (
            z_min, z_max, np.min(z), np.max(z)
        ))

    return (z - z_min) / (z_max - z_min)


def remove_bumps(cde_estimates, delta, bin_size=0.01):
    """Removes bumps in conditional density estimates

    :param cde_estimates: a numpy array or matrix of conditional density estimates.
    :param delta: float, the threshold for bump removal
    :param bin_size: float, size of the bin for density evaluation
    :returns: the conditional density estimates with bumps removed
    :rtype: numpy array or matrix

    """
    if cde_estimates.ndim == 1:
        _remove_bumps(cde_estimates, delta=delta, bin_size=bin_size)
    else:
        np.apply_along_axis(_remove_bumps, 1, cde_estimates,
                            delta=delta, bin_size=bin_size)

def _remove_bumps(density, delta, bin_size=0.01):
    """Removes bumps in conditional density estimates.

    :param density: a numpy array of conditional density estimate.
    :param delta: float, the threshold for bump removal.
    :param bin_size: float, size of the bin for density evaluation.
    :returns: the conditional density estimate with bumps removed.
    :rtype: numpy array.

    """
    area = 0.0
    left_idx = 0
    for right_idx, val in enumerate(density):
        if val <= 0.0:
            if area < delta:
                density[left_idx:(right_idx + 1)] = 0.0
            left_idx = right_idx + 1
            area = 0.0
        else:
            area += val * bin_size
    if area < delta: # final check at end
        density[left_idx:] = 0.0