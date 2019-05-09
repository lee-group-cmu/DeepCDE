import numpy as np
import pywt

class WaveletBasis(object):
    def __init__(self, n_basis, family, n_aux=15):
        self.n_basis = n_basis

        ## Generate wavelet on fine grid
        wav = pywt.DiscreteContinuousWavelet(family)
        _, self.wavelet, self.x_grid = wav.wavefun(n_aux)
        self.N = max(self.x_grid)

    def _wave_fun(self, val):
        if val < 0:
            return 0.0
        if val > self.N:
            return 0.0
        return self.wavelet[np.argmin(abs(val - self.x_grid))]

    def evaluate(self, z_test):
        n_obs = z_test.shape[0]
        basis = np.empty((n_obs, self.n_basis))
        basis[:,0] = 1.0

        z_test = z_test.flatten()

        ## Update wavelet for closest point
        j, k = 0, 0
        for col in range(1, self.n_basis):
            for row, t in enumerate(z_test):
                basis[row, col] = 2 ** (j / 2) * self._wave_fun(2 ** j * t - k)
            k += 1
            if k == 2 ** j:
                j += 1
                k = 0
        return basis

class HaarBasis(WaveletBasis):
    def __init__(self, n_basis):
        super(HaarBasis, self).__init__(n_basis, "db1")
