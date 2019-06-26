
### Alexnet - Pytorch
Pytorch implementation of AlexNet for the task of estimating the probability distribution of correct orientations of an image. 
The input to the model consists of (277, 277) colored images with 3 channels (i.e. color bands).
The target is a continuous variable $y \in (0,2\pi)$ for image orientation. 

Please note that the attached implementations do not include code for generating training and testing sets.
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from deepcde.deepcde_pytorch import cde_layer, cde_loss, cde_predict
from deepcde.utils import box_transform
from deepcde.bases.cosine import CosineBasis

# PRE-PROCESSING #############################################################################
# Create basis (in this case 31 cosine basis)
n_basis = 31
basis = CosineBasis(n_basis)

# ... Creation of training and testing set ...

# Evaluate the y_train over the basis
y_train = box_transform(y_train, 0, 2*math.pi)  # transform to a variable between 0 and 1
y_basis = basis.evaluate(y_train)               # evaluate basis
y_basis = y_basis.astype(np.float32)

# ALEXNET DEFINITION #########################################################################
# `basis_size` is the number of basis (in this case 31).
# `marginal_beta` is the initial value for the bias of the cde layer, if available
class AlexNetCDE(nn.Module):
    def __init__(self, basis_size, marginal_beta=None):
        super(AlexNetCDE, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.cde = cde_layer(4096, basis_size - 1)
        if marginal_beta:
            self.cde.bias.data = torch.from_numpy(marginal_beta[1:]).type(torch.FloatTensor)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=3, stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=3, stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(F.relu(self.conv5(x)), kernel_size=3, stride=2)
        x = F.dropout(x.view(x.size(0), 256 * 6 * 6), training=self.training)
        x = F.dropout(F.relu(self.fc1(x)), training=self.training)
        x = F.dropout(F.relu(self.fc2(x)), training=self.training)
        beta = self.cde(x)
        return beta

# Definition of model and loss function (examples)
model = AlexNetCDE(basis_size=n_basis)
loss = cde_loss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.5)

# TRAINING #################################################################################
# ... Creation of `training_set_loader` and `testing_set_loader` object ...
for epoch in range(n_epochs):
    model.train()
    for batch_idx, (x_batch, y_basis_batch) in enumerate(training_set_loader):
        x_batch, y_basis_batch = x_batch.to(device), y_basis_batch.to(device)
        optimizer.zero_grad()
        beta_batch = model(x_batch)
        loss = loss(beta_batch, y_basis_batch)
        loss.backward()
        optimizer.step()

    # ... Evaluation of testing set ...

# PREDICTION ##############################################################################
# ... Selection of `x_test` to get conditional density estimate of ...
y_grid = np.linspace(0, 1, 1000)  # Creating a grid over the density range
beta_prediction = model(x_test)
cdes = cde_predict(beta_prediction, 0, 1, y_grid, basis, n_basis)
predicted_cdes = cdes * 2 * math.pi  # Re-normalize
```