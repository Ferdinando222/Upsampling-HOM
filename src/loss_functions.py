import torch
import torch.nn as nn
import numpy as np
import wandb


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

    def forward(self, inputs, model_estimation):
        x = inputs[:, 0]
        y = inputs[:, 1]
        z = inputs[:, 2]
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        z = z.requires_grad_(True)

        u = model_estimation(x, y, z)
        
        # Calcola le derivate parziali di u rispetto a x, y e z
        d_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        d_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        d_z = torch.autograd.grad(u.sum(), z, create_graph=True)[0]

        # Calcola il laplaciano di u
        laplace_u = torch.autograd.grad(d_x.sum(), x, create_graph=True)[0] + \
                    torch.autograd.grad(d_y.sum(), y, create_graph=True)[0] + \
                    torch.autograd.grad(d_z.sum(), z, create_graph=True)[0]

        # Calcola il residuo dell'equazione di Helmholtz
        pde_residual = laplace_u + (self.c / self.omega) ** 2 * u

        # Calcola la perdita MSE del residuo dell'equazione di Helmholtz
        mse_pde_loss = torch.mean(torch.abs(pde_residual)**2)

        return mse_pde_loss
        
    

class CombinedLoss(nn.Module):
    def __init__(self, data_weight=1.0,pde_weight=1.0,frequency=1000,pinn=False):
        super(CombinedLoss, self).__init__()
        self.pinn = pinn
        self.data_weight = data_weight
        self.pde_weight = pde_weight
        self.omega = 2 * np.pi * frequency
        if pinn:
            self.pde_loss_fn = HelmholtzLoss(c=340,omega=self.omega)

        self.data_loss_fn = DataTermLoss()


    def forward(self, predictions, target,inputs,model_estimation):
        # Calculate individual loss terms
        data_loss = self.data_loss_fn(predictions, target)
        loss_pde = 0
        if self.pinn:
            pde_loss = self.pde_loss_fn(inputs,model_estimation)
            loss = self.data_weight*data_loss
            loss_pde = self.pde_weight*pde_loss
        
        loss_data = self.data_weight*data_loss
        
        loss = loss_pde+loss_data

        return loss,loss_data,loss_pde
    

    
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


