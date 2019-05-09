import numpy as np

class CosineBasis(object):
    def __init__(self, n_basis):
        self.n_basis = n_basis

    def evaluate(self, z_test):
        n_obs = z_test.shape[0]
        basis = np.empty((n_obs, self.n_basis))

        z_test = z_test.flatten()

        basis[:, 0] = 1.0
        for col in range(1, self.n_basis):
            basis[:, col] = np.sqrt(2) * np.cos(np.pi * col * z_test)
        return basis
