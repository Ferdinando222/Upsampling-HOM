import torch
import torch.nn as nn
import numpy as np


class DataTermLoss(nn.Module):
    def __init__(self):
        super(DataTermLoss, self).__init__()

    def forward(self, predictions, target):
            
            # Combina le due loss in una loss complessa
            loss = torch.mean(torch.abs(target-predictions)**2)
            return loss

class HelmholtzLoss(nn.Module):
    def __init__(self, c, omega):
        super(HelmholtzLoss, self).__init__()
        self.c = c
        self.omega = omega

    def forward(self,inputs,model_estimation):
        inputs = inputs.requires_grad_(True)
        x= inputs[:,0]
        y = inputs[:,1]
        z = inputs[:,2]

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
    
        
    

class CombinedLoss(nn.Module):
    def __init__(self, data_weight=1.0,frequency=1000,pinn=False):
        super(CombinedLoss, self).__init__()
        self.pinn = pinn
        self.data_weight = data_weight
        self.omega = 2 * np.pi * frequency
        if pinn:
            self.pde_weight = 1
            self.pde_loss_fn = HelmholtzLoss(c=340,omega=self.omega)

        self.data_loss_fn = DataTermLoss()


    def forward(self, predictions, target,inputs,model_estimation):
        # Calculate individual loss terms
        data_loss = self.data_loss_fn(predictions, target)

        if self.pinn:
            pde_loss = self.pde_loss_fn(inputs,model_estimation)
            loss = self.data_weight*data_loss + self.pde_weight*pde_loss
        else:
            loss = self.data_weight*data_loss
        return loss
    

    
def NMSE(normalized_output,previsions):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    normalized_output = normalized_output.to(device)
    previsions = previsions.to(device)
    print(normalized_output)
    print(previsions)

    mse = torch.sum(torch.abs(normalized_output - previsions) ** 2)
    print(mse)
    nmse = mse / torch.sum(torch.abs(normalized_output) ** 2)
    nmse_db = 10 * torch.log10(nmse)
    print(nmse_db)
    return nmse_db.item()


