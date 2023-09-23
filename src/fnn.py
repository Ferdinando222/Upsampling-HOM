import wandb
import torch.optim as optim
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

    
    def forward(self, x, y, z):
            x = torch.stack([x, y, z], dim=1)  # Concatenate the inputs along dimension 1
            # Forward pass for the real part
            x = self.activation(self.fc1(x))
            x = self.activation(self.fc2(x))
            x = self.activation(self.fc3(x))

            x = self.fc4(x)
        
            # Dividi l'output in parte reale e immaginaria
            real_output, imag_output = x.chunk(2, dim=1)
            out = torch.complex(real_output, imag_output)
            
            return out


    
    def train_PINN(self,data_handler,hidden_size,frequency,epochs,points_sampled,pinn=False):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        X_data,X_data_not_sampled,Y_data = data_handler.create_tensors()
        Y_data = Y_data.to(device)
        X_data = X_data.to(device)
        X_data_not_sampled = X_data_not_sampled.to(device)
        x=X_data[:,0].to(device)
        y=X_data[:,1].to(device)
        z=X_data[:,2].to(device)


        print("Hidden_size:",hidden_size)

        wandb.init(
                # set the wandb project where this run will be logged
                project="UPSAMPLING-HOM",
                
                # track hyperparameters and run metadata
                config={
                "learning_rate": 0.01,
                "architecture": "FNN",
                "dataset": "DRIR_CR1",
                "epochs": 10000,
                }
            )
        #if pinn:
        #     writer = SummaryWriter(f"runs_pinn/hidden_size_{hidden_size}_{points_sampled}")
        #     writer_nmse = SummaryWriter(f"result_pinn/nmse-size_{points_sampled}")
        #else:
        #     writer = SummaryWriter(f"runs_no_pinn/hidden_size_{hidden_size}_{points_sampled}")
        #     writer_nmse = SummaryWriter(f"result_no_pinn/nmse-size_{points_sampled}")
            
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = self(x,y,z)
            loss = loss_functions.CombinedLoss(1,frequency,pinn)(predictions,Y_data,X_data_not_sampled,self)
                
            #writer.add_scalar("Loss", loss, epoch)
            wandb.log({ f"{hidden_size}_loss": loss})
            loss.backward()
            optimizer.step()
                
            if epoch % 50 == 0:
                 print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item()}')
        #writer.close()

        normalized_output_data = torch.tensor(data_handler.NORMALIZED_OUTPUT_DATA,dtype=torch.complex32)
        previsions = self.make_previsions(X_data_not_sampled)
        nmse = loss_functions.NMSE(normalized_output_data,previsions)
        #writer_nmse.add_scalar("NMSE",nmse,hidden_size)
        wandb.log({f"{points_sampled}_nmse":nmse},step=hidden_size)
        wandb.finish()

    def make_previsions(self,input_data):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        previsions =[]
        x = input_data[:,0].to(device)
        y= input_data[:,1].to(device)
        z = input_data[:,2].to(device)
        previsions = self(x,y,z)
        return previsions


        