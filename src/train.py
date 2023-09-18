import numpy as np
import fnn
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
import loss_functions

def train_PINN(points_sampled,input_dim,output_dim,epochs,X_data,Y_data,X_data_not_sampled,frequency,input_data,normalized_output_data,hidden_size):
    r = X_data[:,2]
    colatitude = X_data[:,1]
    azimuth = X_data[:,1]
    
    for size in range(1):
        print("Hidden_size:",hidden_size)
        writer = SummaryWriter(f"runs_pinn/hidden_size_{hidden_size}_{points_sampled}")
        writer_nmse = SummaryWriter(f"result_pinn/nmse-size_{points_sampled}")
        model = fnn.PINN(input_dim, output_dim,hidden_size)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(epochs):
            predictions = model(r,colatitude,azimuth)

            loss = loss_functions.CombinedLoss(1,1,frequency=frequency)(
                predictions,Y_data,X_data_not_sampled,model)
            
            writer.add_scalar("Loss/train", loss, epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 50 == 0:
                print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item()}')
        writer.close()

        previsions =[]

        for data in input_data:
            with torch.no_grad():
                temp = np.array(model(torch.tensor(data, dtype=torch.float32)))
            previsions.append(temp)
        previsions = np.array(previsions).flatten()

        nmse = loss_functions.NMSE(normalized_output_data,previsions)
        writer_nmse.add_scalar("NMSE",nmse,hidden_size)
        hidden_size = hidden_size+1

    return model

def make_previsions(input_data,model):
    previsions =[]
    r = input[:,2]
    colatitude = input[:,1]
    azimuth = input[:,0]
    for data in input_data:
            with torch.no_grad():
                temp = np.array(model(torch.tensor(r,colatitude,azimuth, dtype=torch.float32)))
            previsions.append(temp)
    previsions = np.array(previsions).flatten()
    return previsions
