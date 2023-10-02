import wandb
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import loss_functions

class PINN(nn.Module):
    def __init__(self, input_size, output_size,hidden_size):
        super(PINN, self).__init__()
        
        #Define the layers for the real part
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc3 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc4 = nn.Linear(hidden_size, 2 * output_size)
        

        # Define the activation function (tanh)
        self.activation = nn.Tanh()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size

    
    def forward(self, x, y, z):
            x = torch.stack([x, y, z], dim=1).to(self.device)  # Concatenate the inputs along dimension 1
            # Forward pass for the real part
            x = self.activation(self.fc1(x))
            x = self.activation(self.fc2(x))
            x = self.activation(self.fc3(x))

            x = self.fc4(x)
        
            real_output, imag_output = x.chunk(2, dim=1)
            out = torch.complex(real_output, imag_output)
            
            return out
    


    
    def train_epoch(self,loader,inputs_not_sampled,optimizer,data_weights,pde_weights,frequency,points_sampled,pinn=False):
        self.pinn = pinn
        self.points_sampled = points_sampled
        inputs_not_sampled = inputs_not_sampled.to(self.device)

        cum_loss_data = []
        cum_loss_pde = []
        cum_loss = []

        for _,(data,target) in enumerate(loader):
            x=data[:,0].to(self.device)
            y=data[:,1].to(self.device)
            z=data[:,2].to(self.device)
            target = target.to(self.device)
        
            optimizer.zero_grad()
            predictions = self(x,y,z)
            loss,loss_data,loss_pde = loss_functions.CombinedLoss(data_weights,pde_weights,frequency,pinn)(predictions,target,inputs_not_sampled,self)
                
            cum_loss_data.append(loss_data)
            cum_loss_pde.append(loss_pde)
            cum_loss.append(loss)

            loss.backward()
            optimizer.step()
                
        # Convert cum_loss to a tensor
        cum_loss_tensor = torch.tensor(cum_loss,dtype=torch.float32)

        # Calculate the mean of cum_loss_tensor
        mean_loss = torch.mean(cum_loss_tensor)

        # Calculate the mean of other losses
        mean_loss_data = torch.mean(torch.tensor(cum_loss_data,dtype=torch.float32))
        mean_loss_pde = torch.mean(torch.tensor(cum_loss_pde,dtype=torch.float32))

        return mean_loss, mean_loss_data, mean_loss_pde

    

    def test(self,data_handler):        
        normalized_output_data = torch.tensor(data_handler.NORMALIZED_OUTPUT_DATA,dtype=torch.complex32)
        _,X_data_not_sampled,_ = data_handler.create_tensors()
        X_data_not_sampled = X_data_not_sampled.to(self.device)
        previsions = self.make_previsions(X_data_not_sampled)
        nmse = loss_functions.NMSE(normalized_output_data,previsions)
        return nmse

              

    def make_previsions(self,input_data):
        previsions =[]
        x = input_data[:,0].to(self.device)
        y= input_data[:,1].to(self.device)
        z = input_data[:,2].to(self.device)
        previsions = self(x,y,z)
        return previsions


        