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
    def __init__(self, c, frequency):
        super(HelmholtzLoss, self).__init__()
        self.c = c
        self.omega = 2 * np.pi * frequency

    def forward(self,inputs,model_estimation):
        inputs = inputs.requires_grad_(True)
        x= inputs[:,2].requires_grad_(True)
        y = inputs[:,1].requires_grad_(True)
        z = inputs[:,0].requires_grad_(True)

        u = model_estimation(x,y,z)
        d_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        d_y= torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        d_z = torch.autograd.grad(u.sum(),z, create_graph=True)[0]
        
        if(d_x.grad_fn is not None):
            d_d_x =  torch.autograd.grad(d_x.sum(), x, create_graph=True)[0] 
        else:
            d_d_x = 0
        if(d_y.grad_fn is not None):
            d_d_y = torch.autograd.grad(d_y.sum(), y, create_graph=True)[0]
        else:
            d_d_y = 0
        if(d_z.grad_fn is not None):
            d_d_z = torch.autograd.grad(d_z.sum(),z,create_graph=True) [0]
        else:
            d_d_z= 0

        laplace_u = d_d_x+d_d_y+d_d_z

        # Calculate the Helmholtz equation residual
        pde_residual = laplace_u + (self.c / self.omega) ** 2 * u

        # Calculate the mean squared error of the PDE residual
        mse_pde_loss = torch.mean(torch.abs(pde_residual)**2)

        return mse_pde_loss
    
    def laplace(fx: torch.Tensor,x:torch.Tensor):
        dfx = torch.autograd(fx,x,create_graph=True)[0]
        ddfx = []
        
    

class CombinedLoss(nn.Module):
    def __init__(self, data_weight=1.0, pde_weight=1.0,frequency=1000):
        super(CombinedLoss, self).__init__()
        self.data_weight = data_weight
        self.pde_weight = pde_weight
        self.data_loss_fn = DataTermLoss()
        self.pde_loss_fn = HelmholtzLoss(c=340,frequency=frequency)


    def forward(self, predictions, target,inputs,model_estimation):
        # Calculate individual loss terms
        data_loss = self.data_loss_fn(predictions, target)
        pde_loss = self.pde_loss_fn(inputs,model_estimation)

        loss = self.data_weight*data_loss + self.pde_weight*pde_loss
        return loss
    

    
def NMSE(normalized_output,previsions):
    normalized_output = normalized_output.flatten()
    mse = np.sum(np.abs(normalized_output-previsions)**2)
    normalization= np.sum(np.abs(normalized_output)**2)

    nmse = mse/normalization
    nmse_db = 10 * math.log10(nmse)
    return nmse_db


