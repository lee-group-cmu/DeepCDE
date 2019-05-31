import torch.nn as nn
from utils import normalize


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

    def forward(self, beta, z_basis, shrink_factor):
        complexity_terms = (beta**2).sum(dim=1) + 1.0
        fit_terms = (beta * z_basis.view(-1, 1).expand_as(beta)).sum(dim=1) + 1.0
        loss = (complexity_terms - 2 * fit_terms).float().mean() / shrink_factor
        return loss

class cde_nll_loss(nn.Module):

    def __init__(self):
        super(cde_nll_loss, self).__init__()

    def forward(self, beta, z_basis, shrink_factor):
        fit_terms = (beta * z_basis.view(-1, 1).expand_as(beta)).sum(dim=1) + 1.0
        loss = ((- 1 * fit_terms) / shrink_factor).float().mean()
        return loss


#### DEFINE PREDICTION FUNCTION

def cde_predict(model_output, z_min, z_max, z_grid_basis):

    n_obs = model_output.shape[0]
    beta = np.hstack((np.ones((n_obs, 1)), model_output))
    cdes = np.matmul(beta, z_grid_basis.T)
    normalize(cdes)
    cdes /= np.prod(z_max - z_min)
    return cdes

