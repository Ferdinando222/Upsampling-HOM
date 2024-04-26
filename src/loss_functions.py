import torch
import torch.nn as nn
import numpy as np
import global_variables as gb
import matplotlib.pyplot as plt
import torch.nn.functional as F


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



        mse =torch.mean(torch.abs(target-predictions)**2,dim=1)
        loss = torch.mean(mse)



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
        u_i_real = u.real
        u_i_imag = u.imag
        pde_res = torch.empty(u.shape[0],u.shape[1],dtype=torch.cfloat).to(gb.device)

        du_real = torch.autograd.grad(u_i_real,(x,y,z),create_graph=True,grad_outputs=(torch.ones_like(u_i_real)))[0]
        du_imag = torch.autograd.grad(u_i_imag,(x,y,z),create_graph=True,grad_outputs=(torch.ones_like(u_i_imag)))[0]

        ddu_real = torch.autograd.grad(du_real, (x,y,z),retain_graph=True,grad_outputs=(torch.ones_like(du_real)))[0]
        ddu_imag = torch.autograd.grad(du_imag, (x,y,z),retain_graph=True,grad_outputs=(torch.ones_like(du_imag)))[0]

        ddu_x_real = ddu_real[:][0]
        ddu_y_real = ddu_real[:][1]
        ddu_z_real = ddu_real[:][2]

        ddu_x_imag = ddu_imag[:][0]
        ddu_y_imag = ddu_imag[:][1]
        ddu_z_imag = ddu_imag[:][2]


        laplacian = ddu_x_real+ddu_y_real+ddu_z_real+1j*(ddu_y_imag+ddu_x_imag+ddu_z_imag)

        k = torch.tensor((self.omega/self.c),dtype=torch.cfloat).to(gb.device)

        pde_res = laplacian.view(-1,1)+k**2*u

        loss = torch.mean(torch.abs(pde_res)**2)

        # x = inputs[:, 0].to(gb.device)
        # y = inputs[:, 1].to(gb.device)
        # z = inputs[:, 2].to(gb.device)


        # u = model_estimation(x, y, z).to(gb.device)#Shape u -> [1202,512]
        # u_i_real = u.real
        # u_i_imag = u.imag
        # pde_res = torch.empty(u.shape[0],u.shape[1],dtype=torch.cfloat).to(gb.device)

        # for i in range(u.shape[1]):
        #     # Calculate partial derivatives of u with respect to x, y, and z
        #     d_x_real,d_y_real,d_z_real = torch.autograd.grad(u_i_real[:,i],(x,y,z),create_graph=True,grad_outputs=(torch.ones_like(u_i_real[:,i])))
        #     d_x_imag,d_y_imag,d_z_imag = torch.autograd.grad(u_i_imag[:,i],(x,y,z),create_graph=True,grad_outputs=(torch.ones_like(u_i_imag[:,i])))


        #     # Calculate the Laplacian of u
        #     laplace = torch.autograd.grad(d_x_real, x,retain_graph=True,grad_outputs=(torch.ones_like(d_x_real)))[0]+ \
        #                    torch.autograd.grad(d_y_real, y,retain_graph=True,grad_outputs=(torch.ones_like(d_y_real)))[0] + \
        #                    torch.autograd.grad(d_z_real, z,retain_graph=True,grad_outputs=(torch.ones_like(d_z_real)))[0]+\
        #                 1j*torch.autograd.grad(d_x_imag, x, retain_graph=True,grad_outputs=(torch.ones_like(d_x_imag)))[0] + \
        #                 1j*torch.autograd.grad(d_y_imag, y, retain_graph=True,grad_outputs=(torch.ones_like(d_y_imag)))[0] + \
        #                 1j*torch.autograd.grad(d_z_imag, z, retain_graph=True,grad_outputs=(torch.ones_like(d_z_imag)))[0]

        #     laplace = laplace.to(gb.device)
        #     k = torch.tensor((self.omega[i]/self.c),dtype=torch.cfloat).to(gb.device)

        #     pde_res[:,i] = laplace + k**2 *u[:,i]

        # norm_pde = 1/(u.shape[1])*torch.norm(pde_res,p=2,dim=1)
        # loss = torch.mean(norm_pde**2)
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

        # Calculate partial derivatives of u with respect to x, y, and z
        u_i_real = u.real
        u_i_imag = u.imag

        bc_res = torch.empty(u.shape[0],u.shape[1],dtype=torch.cfloat).to(gb.device)

        d_x_real = torch.autograd.grad(u_i_real,x,retain_graph=True,grad_outputs=(torch.ones_like(u_i_real)))[0]
        d_y_real = torch.autograd.grad(u_i_real,y,retain_graph=True,grad_outputs=(torch.ones_like(u_i_real)))[0]
        d_z_real = torch.autograd.grad(u_i_real,z,retain_graph=True,grad_outputs=(torch.ones_like(u_i_real)))[0]
        d_x_imag = torch.autograd.grad(u_i_imag,x,retain_graph=True,grad_outputs=(torch.ones_like(u_i_imag )))[0]
        d_y_imag = torch.autograd.grad(u_i_imag,y,retain_graph=True,grad_outputs=(torch.ones_like(u_i_imag )))[0]
        d_z_imag = torch.autograd.grad(u_i_imag,z,retain_graph=True,grad_outputs=(torch.ones_like(u_i_imag )))[0]
        coordinates = torch.stack((x, y, z), dim=1)
        derivatives_real = torch.transpose(torch.stack((d_x_real, d_y_real, d_z_real), dim=1),0,1)
        derivatives_imag = torch.transpose(torch.stack((d_x_imag, d_y_imag, d_z_imag), dim=1),0,1)
        bc_residual_real= torch.diag(coordinates@derivatives_real)
        bc_residual_imag= torch.diag(coordinates@derivatives_imag)
        bc_res = bc_residual_real+bc_residual_imag
        loss = torch.mean(torch.abs(bc_res)**2)
        return loss


class CombinedLoss(nn.Module):

    """
    Combined loss module for both data term and Helmholtz equation term losses.

    This module calculates a combined loss that consists of a data term loss and a Helmholtz equation term loss.
    """

    def __init__(self, pinn=False,alpha=0.999,temperature = 0.01,rho=0.99):
        super(CombinedLoss, self).__init__()
        self.pinn = pinn
        self.omega = 2 * np.pi * gb.frequency
        self.call_count = torch.tensor(0).to(gb.device)
        self.alpha = alpha
        self.rho = rho
        self.temperature = temperature
        self.lambdas = [torch.tensor(1.0).to(gb.device) for _ in range(2)]
        self.last_losses = [torch.tensor(1.0).to(gb.device) for _ in range(2)]
        self.init_losses = [torch.tensor(1.0).to(gb.device) for _ in range(2)]
        if pinn:
            self.pde_loss_fn = HelmholtzLoss(c=343, omega=self.omega)
            #self.bc_loss_fn = BCLoss()

        self.data_loss_fn = DataTermLoss()

    def forward(self, predictions, target, inputs, model_estimation):
        # Calcola le perdite individuali
        data_loss = self.data_loss_fn(predictions, target)
        losses = [data_loss]
        loss_pde = 0
        loss_bc = 0

        if self.pinn:
            pde_loss = self.pde_loss_fn(inputs, model_estimation)
            loss_pde = pde_loss
            #loss_bc = self.bc_loss_fn(inputs,model_estimation)
            losses = [data_loss, loss_pde]
            loss = gb.e_d*losses[0]+gb.e_f*losses[1]
        else:
            loss = gb.e_d*losses[0]


        # alpha = torch.tensor(1.0) if self.call_count == 0 else \
        #     (torch.tensor(0.0) if self.call_count == 1 else self.alpha)
        # rho = torch.tensor(1.0) if self.call_count == 0 else \
        #     (torch.tensor(1.0) if self.call_count == 1 else torch.bernoulli(
        #         torch.tensor(self.rho, dtype=torch.float32)))
        # lambdas_hat = [(losses[i] / (self.last_losses[i] * self.temperature + EPS)) for i in range(len(losses))]
        # max_value, _ = torch.max(torch.stack(lambdas_hat), dim=0)
        # lambdas_hat = F.softmax(torch.stack(lambdas_hat) - max_value, dim=0) * len(losses)
        # init_lambdas_hat = [(losses[i] / (self.init_losses[i] * self.temperature + EPS)) for i in range(len(losses))]
        # max_value_init, _ = torch.max(torch.stack(init_lambdas_hat), dim=0)
        # init_lambdas_hat = F.softmax(torch.stack(init_lambdas_hat) - max_value_init, dim=0) * len(losses)
        # new_lambdas = [
        #     (rho * alpha * self.lambdas[i] + (1 - rho) * alpha * init_lambdas_hat[i] + (1 - alpha) * lambdas_hat[i]) for
        #     i in range(len(losses))]
        # self.lambdas = [lam.detach().clone() for lam in new_lambdas]
        # for i in range(len(losses)):
        #     if self.call_count == 0:
        #         self.init_losses[i] = losses[i].clone().detach()
        #     else:
        #         self.last_losses[i] = losses[i].clone().detach()
        # gb.e_f = self.lambdas[1]
        # gb.e_d = self.lambdas[0]
        # self.call_count = self.call_count+1
        # loss = torch.sum(torch.stack([new_lambdas[i] * losses[i] for i in range(len(losses))]))

        return loss, data_loss, loss_pde, loss_bc

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
    mse = (normalized_output - predictions) ** 2
    mse = torch.sum(mse, dim=1)
    norm = torch.sum(normalized_output ** 2, dim=1)

    # Calculate NMSE in dB
    nmse = mse / norm
    std_dev = torch.std(nmse)
    nmse_db = 10 * torch.log10(torch.mean(nmse))
    nmse_db_p = 10*torch.log10(nmse)
    nmse_db_p = torch.mean(nmse_db_p)
    return nmse_db, nmse, std_dev

def NMSE_freq(normalized_output, predictions):
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

    mse = torch.sum(mse,dim=0)

    norm = torch.sum(torch.abs(normalized_output) ** 2,dim=0)

    # Calculate NMSE in dB
    nmse = mse / norm
    nmse_db = 10 * torch.log10(torch.mean(nmse))


    return nmse_db,nmse


