import torch
import torch.nn as nn
import numpy as np
import math

class DataTermLoss(nn.Module):
    def __init__(self):
        super(DataTermLoss, self).__init__()

    def forward(self, predictions, target):
        mse = torch.mean(torch.abs(predictions - target) ** 2)
        return mse

class HelmholtzLoss(nn.Module):
    def __init__(self, c, frequency, num_points=1000):
        super(HelmholtzLoss, self).__init__()
        self.c = c
        self.omega = 2 * np.pi * frequency
        self.num_points = num_points

    def forward(self, predictions, inputs):
        # Calculate the Helmholtz equation residual for each point
        u = predictions.clone().requires_grad_(True)  # Set requires_grad=True
        inputs.requires_grad_(True)  # Set requires_grad=True for inputs

        # Compute laplace_u with create_graph=True and allow_unused=True
        laplace_u = torch.autograd.grad(outputs=u.sum(), inputs=inputs, create_graph=True, allow_unused=True)[0]

        if laplace_u is not None:
            laplace_u = laplace_u.sum(dim=1)
        else:
            laplace_u = torch.zeros_like(u)  # Handle the case where laplace_u is not used in the graph

        pde_residual = laplace_u + (self.c / self.omega) ** 2 * u

        # Calculate the sum of squared residuals for num_points
        mse_pde_loss = torch.mean(torch.abs(pde_residual)** 2)

        return mse_pde_loss
    

class CombinedLoss(nn.Module):
    def __init__(self, data_weight=1.0, pde_weight=1.0,frequency=1000):
        super(CombinedLoss, self).__init__()
        self.data_weight = data_weight
        self.pde_weight = pde_weight
        self.data_loss_fn = DataTermLoss()
        self.pde_loss_fn = HelmholtzLoss(c=340,frequency=frequency)


    def forward(self, predictions, target,input):
        # Calculate individual loss terms
        data_loss = self.data_loss_fn(predictions, target)
        pde_loss = self.pde_loss_fn(predictions,input)

        loss = self.data_weight*data_loss+self.pde_weight*pde_loss
        return loss
    

    
def NMSE(normalized_output,previsions):
    normalized_output = normalized_output.flatten()
    mse = np.sum(np.abs(normalized_output-previsions)**2)
    normalization= np.sum(np.abs(normalized_output)**2)

    nmse = mse/normalization
    nmse_db = 10 * math.log10(nmse)
    return nmse_db


