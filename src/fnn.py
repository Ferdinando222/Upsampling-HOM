import numpy as np
import fnn
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import loss_functions

class PINN(nn.Module):
    def __init__(self, input_size, output_size,hidden_size):
        super(PINN, self).__init__()
        
        # Define the layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        
        # Define the activation function (tanh)
        self.activation = nn.Tanh()
    
    def forward(self, x,y,z):
        x = torch.stack([x,y,z], dim=1)  # Concatenate the inputs along dimension 1
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        real_out = self.fc4(x)
        imag_out = self.fc4(x)
        out = torch.complex(real_out, imag_out)
        return out
    
    def train_PINN(self,data_handler,hidden_size,frequency,epochs,points_sampled):
        X_data,X_data_not_sampled,Y_data = data_handler.create_tensors()
        x=X_data[:,2]
        y=X_data[:,1]
        z=X_data[:,0]

        print("Hidden_size:",hidden_size)
        writer = SummaryWriter(f"runs_pinn/hidden_size_{hidden_size}_{points_sampled}")
        writer_nmse = SummaryWriter(f"result_pinn/nmse-size_{points_sampled}")
            
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        for epoch in range(epochs):
            predictions = self(x,y,z)

            loss = loss_functions.CombinedLoss(1,1,frequency=frequency)(
            predictions,Y_data,X_data_not_sampled,self)
                
            writer.add_scalar("Loss/train", loss, epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            if epoch % 50 == 0:
                 print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item()}')
        writer.close()

        normalized_output_data = data_handler.NORMALIZED_OUTPUT_DATA
        previsions = self.make_previsions(X_data_not_sampled)
        nmse = loss_functions.NMSE(normalized_output_data,previsions)
        writer_nmse.add_scalar("NMSE",nmse,hidden_size)

    def make_previsions(self,input_data):
        previsions =[]
        r = input_data[:,2]
        colatitude = input_data[:,1]
        azimuth = input_data[:,0]
        previsions = np.array(self(r,colatitude,azimuth).detach().numpy())
        previsions = previsions.flatten()
    
        return previsions


        