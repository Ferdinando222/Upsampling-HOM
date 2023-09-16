import torch
import torch.nn as nn

class FNN(nn.Module):
    def __init__(self, input_size, output_size,hidden_size):
        super(FNN, self).__init__()
        
        # Define the layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        
        # Define the activation function (tanh)
        self.activation = nn.Tanh()
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        real_out = self.fc4(x)
        imag_out = self.fc4(x)
        out = torch.complex(real_out, imag_out)
        return out