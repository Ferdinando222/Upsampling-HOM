import torch
import numpy as np
import torch.nn as nn
import global_variables as gb
import loss_functions

class PINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) class.

    This class defines a PINN model that can be used for solving partial differential equations (PDEs)
    with neural networks.

    Args:
        input_size (int): The input size of the PINN model.
        output_size (int): The output size of the PINN model.
        hidden_size (int): The size of the hidden layers in the neural network.

    Attributes:
        fc1 (nn.Linear): First fully connected layer for the real part of the PINN.
        fc2 (nn.Linear): Second fully connected layer for the real part of the PINN.
        fc3 (nn.Linear): Third fully connected layer for the real part of the PINN.
        fc4 (nn.Linear): Fourth fully connected layer for the real part of the PINN.
        activation (nn.Tanh): Activation function (tanh).
        hidden_size (int): Size of the hidden layers.
    """

    def __init__(self, input_size, output_size, hidden_size):
        super(PINN, self).__init__()

        # Define the layers for the real part
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc3 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc4 = nn.Linear(hidden_size, 2 * output_size)

        # Define the activation function (tanh)
        self.activation = nn.Tanh()
        self.hidden_size = hidden_size

    def forward(self, x, y, z,f):
        """
        Forward pass of the PINN model.

        Args:
            x (torch.Tensor): Input for the x-coordinate.
            y (torch.Tensor): Input for the y-coordinate.
            z (torch.Tensor): Input for the z-coordinate.

        Returns:
            out (torch.Tensor): Complex-valued output.
        """
        x = torch.stack([x, y, z,f], dim=1).to(gb.device)  # Concatenate the inputs along dimension 1

        # Forward pass for the real part
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)

        real_output, imag_output = x.chunk(2, dim=1)
        out = torch.complex(real_output, imag_output)

        return out

    def train_epoch(self, loader, inputs_not_sampled, optimizer, data_weights, pde_weights,bc_weights, points_sampled, pinn=False):
        """
        Train the PINN model for one epoch.

        Args:
            loader (torch.utils.data.DataLoader): Data loader for training data.
            inputs_not_sampled (torch.Tensor): Inputs not sampled by the PINN.
            optimizer (torch.optim.Optimizer): Optimizer for model training.
            data_weights (float): Weight for data term loss.
            pde_weights (float): Weight for PDE term loss.
            points_sampled (int): Number of sampled points.
            pinn (bool): Flag to enable PINN training.

        Returns:
            mean_loss (torch.Tensor): Mean loss of the epoch.
            mean_loss_data (torch.Tensor): Mean data term loss of the epoch.
            mean_loss_pde (torch.Tensor): Mean PDE term loss of the epoch.
        """
        self.pinn = pinn
        self.points_sampled = points_sampled
        inputs_not_sampled = inputs_not_sampled.to(gb.device)

        cum_loss_data = []
        cum_loss_pde = []
        cum_loss_bc = []
        cum_loss = []

        for i, (data, target) in enumerate(loader):
            for j in range(100):
                x = data[:,j,0].to(gb.device)
                y = data[:,j,1].to(gb.device)
                z = data[:,j,2].to(gb.device)
                f = data[:,j,3].to(gb.device)
                targets = target[:,1000+j].to(gb.device)

                optimizer.zero_grad()
                predictions = self(x, y, z,f)
                loss, loss_data, loss_pde,loss_bc= loss_functions.CombinedLoss(data_weights, pde_weights,bc_weights, pinn)(predictions, targets,
                                                                                                       inputs_not_sampled,self)

                cum_loss_data.append(loss_data)
                cum_loss_pde.append(loss_pde)
                cum_loss_bc.append(loss_bc)
                cum_loss.append(loss)

                loss.backward()
                optimizer.step()

        # Convert cum_loss to a tensor
        cum_loss_tensor = torch.tensor(cum_loss, dtype=torch.float32)

        # Calculate the mean of cum_loss_tensor
        mean_loss = torch.mean(cum_loss_tensor)

        # Calculate the mean of other losses
        mean_loss_data = torch.mean(torch.tensor(cum_loss_data, dtype=torch.float32))
        mean_loss_pde = torch.mean(torch.tensor(cum_loss_pde, dtype=torch.float32))
        mean_loss_bc = torch.mean(torch.tensor(cum_loss_bc, dtype=torch.float32))

        return mean_loss, mean_loss_data, mean_loss_pde,mean_loss_bc
    
    def test_epoch(self, val_loader):
        batch_losses = []
        self.eval()
        with torch.no_grad():
            for i, (data, target) in enumerate(val_loader):
                for j in range(100):
                    x = data[:,j,0].to(gb.device)
                    y = data[:,j,1].to(gb.device)
                    z = data[:,j,2].to(gb.device)
                    f = data[:,j,3].to(gb.device)
                    targets = target[:,1000+j].to(gb.device)
                    predictions = self(x,y,z,f)

                    loss= loss_functions.DataTermLoss()(predictions,targets)
                    batch_losses.append(loss)
        

        val_loss_tensor = torch.tensor(batch_losses, dtype=torch.float32)

        # Calculate the mean of cum_loss_tensor
        val_mean_loss = torch.mean(val_loss_tensor)

        return val_mean_loss



    def test(self, data_handler):
        """
        Test the PINN model.

        Args:
            data_handler: An instance of the data handler.

        Returns:
            nmse_db (float): Normalized Mean Squared Error (NMSE) in decibels (dB).
        """
        normalized_output_data = torch.tensor(data_handler.NORMALIZED_OUTPUT, dtype=torch.cfloat)
        X_data_not_sampled= data_handler.X_data
        X_data_not_sampled = X_data_not_sampled.to(gb.device)
        previsions = self.make_previsions(X_data_not_sampled)
        nmse = loss_functions.NMSE(normalized_output_data, previsions)
        return nmse

    def make_previsions(self, input_data):
        """
        Make predictions using the PINN model.

        Args:
            input_data (torch.Tensor): Input data for predictions.

        Returns:
            previsions (torch.Tensor): Model predictions.
        """
        previsions = []
        for i in range(100):
            x = input_data[:,i,0].to(gb.device)
            y = input_data[:,i,1].to(gb.device)
            z = input_data[:,i,2].to(gb.device)
            f = input_data[:,i,3].to(gb.device)
            prev = self(x,y,z,f).detach().numpy()
            previsions.append(prev)

        previsions = np.array(previsions)
        previsions = previsions.transpose(1,0,2)
        previsions = np.squeeze(previsions, axis=2)
        previsions = torch.tensor(previsions,dtype=torch.cfloat)
        return previsions
