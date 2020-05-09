import torch.nn as nn
import numpy as np
from .utils import normalize, remove_bumps


### DEFINE CUSTOM LAYER

class cde_layer(nn.Module):

    def __init__(self, D_in, n_basis):
        super(cde_layer, self).__init__()
        self.linear = nn.Linear(D_in, n_basis-1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


### DEFINE CUSTOM LOSS

class cde_loss(nn.Module):

    def __init__(self):
        super(cde_loss, self).__init__()

    def forward(self, beta, z_basis, shrink_factor=1.0):
        complexity_terms = (beta**2).sum(dim=1) + 1.0
        fit_terms = (beta * z_basis.expand_as(beta)).sum(dim=1) + 1.0
        loss = (complexity_terms - 2 * fit_terms).float().mean() / shrink_factor
        return loss

class cde_nll_loss(nn.Module):

    def __init__(self):
        super(cde_nll_loss, self).__init__()

    def forward(self, beta, z_basis, shrink_factor=1.0):
        fit_terms = (beta * z_basis.view(-1, 1).expand_as(beta)).sum(dim=1) + 1.0
        loss = ((- 1 * fit_terms) / shrink_factor).float().mean()
        return loss


class approx_cde_loss(nn.Module):

    def __init__(self):
        super(approx_cde_loss, self).__init__()

    def forward(self, beta, shrink_factor=1.0):
        complexity_terms = (beta**2).sum(dim=1) + 1.0
        loss = (-1*complexity_terms).mean() / shrink_factor
        return loss


#### DEFINE PREDICTION FUNCTION

def cde_predict(model_output, z_min, z_max, z_grid,
                basis, delta=None, bin_size=0.01):

    n_obs = model_output.shape[0]
    beta = np.hstack((np.ones((n_obs, 1)), model_output))
    z_grid_basis = basis.evaluate(z_grid)[:, :basis.n_basis]
    cdes = np.matmul(beta, z_grid_basis.T)
    if delta is not None:
        remove_bumps(cdes, delta=delta, bin_size=bin_size)
    normalize(cdes)
    cdes /= np.prod(z_max - z_min)
    return cdes

