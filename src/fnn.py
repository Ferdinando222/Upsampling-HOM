import torch
import torch.nn as nn
import global_variables as gb
import loss_functions
from siren import SIREN
from siren.init import siren_uniform_
import torch.nn.init as init

class Sine(nn.Module):
    def __init__(self, w0: float = 1.0):
        """Sine activation function with w0 scaling support.

        Example:
            >>> w = torch.tensor([3.14, 1.57])
            >>> Sine(w0=1)(w)
            torch.Tensor([0, 1])

        :param w0: w0 in the activation step `act(x; w0) = sin(w0 * x)`.
            defaults to 1.0
        :type w0: float, optional
        """

        super(Sine, self).__init__()
        self.w0  = w0

        # self.w4 = nn.Parameter(torch.tensor(10.0))
        # self.w5 = nn.Parameter(torch.tensor(20.0))
        # self.w6 = nn.Parameter(torch.tensor(30.0))
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        return torch.sin(self.w0 * x)
    #            torch.sin(self.w1*x)+torch.sin(self.w2*x)+torch.sin(self.w3*x)
    #             +0.01*torch.sin(self.w4*x)+0.01*torch.sin(self.w5*x)

    @staticmethod
    def _check_input(x):
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                'input to forward() must be torch.xTensor')
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

    def __init__(self, input_size, output_size, hidden_size, num_layers=5,w0=1,w0_init=30,c=6):
        super(PINN, self).__init__()

        layers = [hidden_size]*num_layers
        self.w0 = w0
        self.w0_init= w0_init

        self.layers_real = [nn.Linear(input_size, layers[0], bias=True), Sine(w0=self.w0_init)]

        for index in range(len(layers) - 1):
            self.layers_real.extend([
                nn.Linear(layers[index], layers[index + 1], bias=True), Sine(w0=self.w0)])

        self.layers_real.append(nn.Linear(layers[-1], output_size, bias=True))
        self.network_real = nn.Sequential(*self.layers_real)

        self.layers_imag = [nn.Linear(input_size, layers[0], bias=True), Sine(w0=self.w0_init)]

        for index in range(len(layers) - 1):
            self.layers_imag.extend([
                nn.Linear(layers[index], layers[index + 1], bias=True), Sine(w0=self.w0)])

        self.layers_imag.append(nn.Linear(layers[-1], output_size, bias=True))
        self.network_imag = nn.Sequential(*self.layers_imag)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                siren_uniform_(m.weight, mode='fan_in', c=c)

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

        x = torch.stack([x, y, z], dim=1).to(gb.device)

        # x_real = self.fc_out_real(x_real)
        # x_img = self.fc_out_img(x_img)

        # SIREN
        x_real = self.network_real(x)
        x_imag = self.network_imag(x)

        out = torch.complex(x_real, x_imag)
        return out

    def load_state_cust_dict(self, state_dict):
        # Carica il valore di w0
        if 'w0' in state_dict:
            self.w0.data.copy_(state_dict['w0'])

        # Carica i pesi e i bias per la rete network_real e network_imag
        for network_name in ['network_real', 'network_imag']:
            for i, layer in enumerate(f'{network_name}.network'):
                layer_name = f'{network_name}.network.{i}'

                # Carica i pesi
                weight_key = f'{layer_name}.weight'
                if weight_key in state_dict:
                    getattr(getattr(self, network_name).network[i], 'weight').data.copy_(state_dict[weight_key])

                # Carica i bias
                bias_key = f'{layer_name}.bias'
                if bias_key in state_dict:
                    getattr(getattr(self, network_name).network[i], 'bias').data.copy_(state_dict[bias_key])

                # Se presenti, aggiorna anche w0 nei moduli Sine della rete
                for module in getattr(self, network_name).network:
                    if isinstance(module, Sine):
                        module.w0 = self.w0.data

        # Rimuovi i valori di w0, pesi e bias dallo stato del dizionario
        state_dict.pop('w0', None)
        for network_name in ['network_real', 'network_imag']:
            for i, _ in enumerate(f'{network_name}.network'):
                layer_name = f'{network_name}.network.{i}'
                state_dict.pop(f'{layer_name}.weight', None)
                state_dict.pop(f'{layer_name}.bias', None)

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
        nmse.append(loss_functions.NMSE_freq(normalized_output_data, previsions))
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
        previsions = torch.tensor(previsions,dtype=torch.cfloat)
        return previsions

