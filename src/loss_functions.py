import torch
import torch.nn as nn
import numpy as np
import global_variables as gb
from torch.utils.checkpoint import checkpoint

# Import necessary libraries and modules

class DataTermLoss(nn.Module):
    """
    Custom loss module for data term loss calculation. Used for compute MSE for complex value.

    This module calculates the mean squared error (MSE) loss between predictions and target values.
    """

    def __init__(self):
        super(DataTermLoss, self).__init__()

    def forward(self, predictions, target):
        """
        Forward pass of the DataTermLoss module.

        Args:
            predictions (torch.Tensor): Predicted values.
            target (torch.Tensor): Target values.

        Returns:
            loss (torch.Tensor): The calculated loss.
        """

        # Calculate the MSE loss: added a frequency-dependent weight.
        loss = torch.mean(torch.abs(target - predictions) ** 2 )

        return loss

class HelmholtzLoss(nn.Module):
    """
    Custom loss module for Helmholtz equation loss calculation.

    This module calculates the mean squared error (MSE) loss for solving the Helmholtz equation.
    """

    def __init__(self, c, omega):
        super(HelmholtzLoss, self).__init__()
        self.c = c
        self.omega = omega

    def forward(self, inputs, model_estimation):
        """
        Forward pass of the HelmholtzLoss module.

        Args:
            inputs (torch.Tensor): Input coordinates.
            model_estimation (callable): A callable model function for estimating pressure.

        Returns:
            mse_pde_loss (torch.Tensor): The calculated loss for the Helmholtz equation.
        """

        x = inputs[:, 0].to(gb.device)
        y = inputs[:, 1].to(gb.device)
        z = inputs[:, 2].to(gb.device)

        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        z = z.requires_grad_(True)

        u = model_estimation(x, y, z).to(gb.device)

        self.omega = torch.tensor(self.omega,dtype=torch.float32)
        self.omega = self.omega.to(gb.device)

        laplace = torch.zeros(u.shape[0],u.shape[1]-1, dtype=torch.cfloat, device=gb.device)

        # Calculate partial derivatives of u with respect to x, y, and z
        for i in range(u.shape[1]-1):
            u_i_real = u[:,i+1].real
            u_i_imag = u[:,i+1].imag
            d_x_real = torch.autograd.grad(u_i_real.sum(),x,create_graph=True)[0]
            d_y_real = torch.autograd.grad(u_i_real.sum(),y,create_graph=True)[0]
            d_z_real = torch.autograd.grad(u_i_real.sum(),z,create_graph=True)[0]

            d_x_imag = torch.autograd.grad(u_i_imag.sum(),x,create_graph=True)[0]
            d_y_imag = torch.autograd.grad(u_i_imag.sum(),y,create_graph=True)[0]
            d_z_imag = torch.autograd.grad(u_i_imag.sum(),z,create_graph=True)[0]


            # Calculate the Laplacian of u

            laplace_u = torch.autograd.grad(d_x_real.sum(), x,retain_graph=True)[0]+ \
                        torch.autograd.grad(d_y_real.sum(), y,retain_graph=True)[0] + \
                        torch.autograd.grad(d_z_real.sum(), z,retain_graph=True)[0]+ \
                        1j*torch.autograd.grad(d_x_imag.sum(), x, retain_graph=True)[0] + \
                        1j*torch.autograd.grad(d_y_imag.sum(), y, retain_graph=True)[0] + \
                        1j*torch.autograd.grad(d_z_imag.sum(), z, retain_graph=True)[0]

            laplace[:,i] = laplace_u

        laplace = laplace.to(gb.device)
        pde_residual = laplace + (self.omega[1:]/self.c) ** 2 * u[:,1:]
        loss = torch.mean(torch.abs(pde_residual)**2,dim=1)
        loss = torch.mean(loss)

        return loss

class BCLoss(nn.Module):

    def __init__(self):
        super(BCLoss, self).__init__()

    def forward(self,inputs,model_estimation):

        x = inputs[:, 0].to(gb.device)
        y = inputs[:, 1].to(gb.device)
        z = inputs[:, 2].to(gb.device)

        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        z = z.requires_grad_(True)

        u = model_estimation(x, y, z)

        d_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        d_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        d_z = torch.autograd.grad(u.sum(), z, create_graph=True)[0]

        coordinates = torch.stack((x, y, z), dim=1)
        derivatives = torch.transpose(torch.stack((d_x, d_y, d_z), dim=1),0,1)

        bc_residual = torch.matmul(coordinates, derivatives)
        mse_bc = torch.mean(torch.abs(bc_residual) ** 2)

        return mse_bc


class CombinedLoss(nn.Module):
    
    """
    Combined loss module for both data term and Helmholtz equation term losses.

    This module calculates a combined loss that consists of a data term loss and a Helmholtz equation term loss.
    """

    def __init__(self, data_weight=1.0, pde_weight=1.0,bc_weight=1.0, pinn=False):
        super(CombinedLoss, self).__init__()
        self.pinn = pinn
        self.bc_weight = bc_weight
        self.data_weight = data_weight
        self.pde_weight = pde_weight
        self.omega = 2 * np.pi * gb.frequency
        if pinn:
            self.pde_loss_fn = HelmholtzLoss(c=340, omega=self.omega)
            #self.bc_loss_fn = BCLoss()

        self.data_loss_fn = DataTermLoss()

    def forward(self, predictions, target, inputs, model_estimation):
        """
        Forward pass of the CombinedLoss module.

        Args:
            predictions (torch.Tensor): Predicted values.
            target (torch.Tensor): Target values.
            inputs (torch.Tensor): Input coordinates.
            model_estimation (callable): A callable model function for estimating u.

        Returns:
            loss (torch.Tensor): The combined loss.
            loss_data (torch.Tensor): The data term loss.
            loss_pde (torch.Tensor): The Helmholtz equation term loss (if applicable).
        """
        # Calculate individual loss terms
        data_loss = self.data_loss_fn(predictions, target)
        loss_pde = 0
        loss_bc = 0
        if self.pinn:
            pde_loss = self.pde_loss_fn(inputs, model_estimation)
            #bc_loss = self.bc_loss_fn(inputs,model_estimation)
            loss_pde = self.pde_weight*pde_loss
            #loss_bc = self.bc_weight *bc_loss

        loss_data = self.data_weight * data_loss 

        # Calculate the combined loss
        loss = loss_pde + loss_data + loss_bc

        return loss, loss_data, loss_pde,loss_bc


def NMSE(normalized_output, predictions):
    """
    Calculate the Normalized Mean Squared Error (NMSE) in decibels (dB) between two tensors.

    Args:
        normalized_output (torch.Tensor): The normalized output.
        predictions (torch.Tensor): The predicted values.

    Returns:
        nmse_db (float): The NMSE in decibels (dB).
    """
    normalized_output = normalized_output.to(gb.device)
    predictions = predictions.to(gb.device)

    # Calculate Mean Squared Error (MSE)
    mse = torch.abs(normalized_output - predictions) ** 2

    mse = torch.sum(mse,dim=1)

    norm = torch.sum(torch.abs(normalized_output) ** 2,dim=1)

    # Calculate NMSE in dB
    nmse = mse / norm

    nmse_db = 10 * torch.log10(torch.mean(nmse))


    return nmse_db,nmse

