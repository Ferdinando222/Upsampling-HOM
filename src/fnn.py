import torch
import torch.nn as nn
import global_variables as gb
import loss_functions
from siren import SIREN
import spherical_harmonics as sh


class PINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) class.

    This class defines a PINN model that can be used for solving partial differential equations (PDEs)
    with neural networks.

    Args:
        input_size (int): The input size of the PINN model.
        output_size (int): The output size of the PINN model.
        hidden_size (int): The size of the hidden layers in the neural network.
        layers (int): The number of hidden layers
        w0 (float): A Siren parameter
        w0_initial (float): A Siren parameter
        c (flaot): A Siren parameter

    Attributes:
        network: network Siren
    """

    def __init__(self, input_size, output_size, hidden_size, layers=5,w0=1,w0_init=30,c=6):
        super(PINN, self).__init__()

        self.network_real= SIREN(
            layers=[hidden_size] * layers,
            in_features=input_size,
            out_features=output_size,
            w0=w0,
            w0_initial=w0_init,
            c=c,
            initializer='siren'
        )

        self.network_imag = SIREN(
            layers=[hidden_size] * layers,
            in_features=input_size,
            out_features=output_size,
            w0=w0,
            w0_initial=w0_init,
            c=c,
            initializer='siren'
        )

        self.sampling = 0

    def forward(self, x, y, z):
        """
        Forward pass of the PINN model.

        Args:
            x (torch.Tensor): Input for the x-coordinate.
            y (torch.Tensor): Input for the y-coordinate.
            z (torch.Tensor): Input for the z-coordinate.

        Returns:
            out (torch.Tensor): Complex-valued output.
        """
        r = torch.sqrt(x**2 + y**2 + z**2)
        theta = torch.atan2(y,x)
        theta = (theta + 2 * torch.pi) % (2 * torch.pi)  # Trasformazione per ottenere un angolo tra 0 e pi
        phi = torch.arccos(z/r)
        x = torch.stack([x, y, z], dim=1).to(gb.device)

        # Forward pass through the deep network
        x_real = self.network_real(x)
        x_imag = self.network_imag(x)

        out = torch.complex(x_real, x_imag)
        N = 6
        out = torch.reshape(out,(x.size(0),(N+1)**2,-1))


        P_values = torch.zeros(x.size(0),len(gb.frequency), dtype=torch.cfloat).to(gb.device)
        
        ##Computing the pressure field for position x,y,z
        for idx, k in enumerate(gb.frequency):
            P = 0.0
            i  = 0
            for n in range(N + 1):
                for m in range(-n, n + 1):
                    Y_nm = sh.spherical_harmonics(theta,phi,n,m)  # Armonici sferici
                    j_nkr = sh.spherical_bessel(n, torch.tensor(k,dtype=torch.float32),r)  # Funzione di Bessel sferica
                    A_nm = out[:,i,idx] # Coefficienti sferici casuali (puoi sostituirli con i tuoi valori)

                    # Somma il contributo di questa armonica sferica al campo acustico totale
                    P += Y_nm * A_nm * j_nkr
                    i = i+1

            # Assegna il valore del campo acustico per questa frequenza
            
            P_values[:,idx] = P

        P_values = P_values.requires_grad_()
        return P_values

    def train_epoch(self, loader, inputs_not_sampled, optimizer, loss_comb,points_sampled, pinn=False):
        """
        Train the PINN model for one epoch.

        Args:
            loader (torch.utils.data.DataLoader): Data loader for training data.
            inputs_not_sampled (torch.Tensor): Inputs not sampled by the PINN.
            optimizer (torch.optim.Optimizer): Optimizer for model training.
            loss_comb : Loss for training
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

        for _, (data, target) in enumerate(loader):
            x = data[:, 0].to(gb.device)
            y = data[:, 1].to(gb.device)
            z = data[:, 2].to(gb.device)
            target = target.to(gb.device)
            predictions = self(x, y, z)
            loss, loss_data, loss_pde,loss_bc = loss_comb(predictions, target,inputs_not_sampled,self)
            cum_loss_data.append(loss_data)
            cum_loss_pde.append(loss_pde)
            cum_loss_bc.append(loss_bc)
            cum_loss.append(loss)
            optimizer.zero_grad()
            loss.backward()

            # For check if the gradients are not None

            # for name, param in self.named_parameters():
            #     print(f'Gradient {name}: {param.grad}')
            optimizer.step()
            self.sampling = self.sampling+1

            if(self.sampling>13):
                self.sampling = 0

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
            for _, (data, targets) in enumerate(val_loader):

                x = data[:, 0].to(gb.device)
                y = data[:, 1].to(gb.device)
                z = data[:, 2].to(gb.device)
                targets = targets.to(gb.device)
                predictions = self(x,y,z)

                loss = loss_functions.DataTermLoss()(predictions,targets)
                batch_losses.append(loss)
        

        val_loss_tensor = torch.tensor(batch_losses, dtype=torch.float32)

        # Calculate the mean of cum_loss_tensor
        val_mean_loss = torch.mean(val_loss_tensor)
        self.train()
        return val_mean_loss



    def test(self, data_handler):
        """
        Test the PINN model.

        Args:
            data_handler: An instance of the data handler.

        Returns:
            nmse_db (float): Normalized Mean Squared Error (NMSE) in decibels (dB).
        """
        self.eval()
        X_data_not_sampled= data_handler.X_data
        X_data_not_sampled = X_data_not_sampled.to(gb.device)
        previsions = self.make_previsions(X_data_not_sampled)
        nmse = []
        normalized_output_data = torch.tensor(data_handler.NORMALIZED_OUTPUT, dtype=torch.cfloat)
        nmse.append(loss_functions.NMSE(normalized_output_data, previsions))
        self.train()

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
        x = input_data[:, 0].to(gb.device)
        y = input_data[:, 1].to(gb.device)
        z = input_data[:, 2].to(gb.device)
        previsions = self(x, y, z)
        previsions = previsions.clone().detach()
        return previsions

