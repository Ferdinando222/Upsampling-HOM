#%%
import data as dt
import utils
import fnn
import torch
import numpy as np

# Hyperparameters
frequency = 1000
input_dim = 3 
output_dim = 1 
hidden_size = 2
epochs = 10000
points_sampled =1202
path_data = "../dataset/DRIR_CR1_VSA_1202RS_R.sofa"
data_handler = dt.DataHandler(path_data,frequency)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Start Training
#%%
for j in range(3):
    hidden_size = 2
    for i in range(1):
        #model = fnn.PINN(input_dim,output_dim,hidden_size)
        model_no_pinn = fnn.PINN(input_dim,output_dim,hidden_size)

        # print("Start Training PINN")
        # model = model.to(device)
        # model.train_PINN(data_handler,hidden_size,frequency,epochs,points_sampled,pinn=True)

        print("Start Training no PINN")
        model_no_pinn = model_no_pinn.to(device)
        model_no_pinn.train_PINN(data_handler,hidden_size,frequency,epochs,points_sampled)
        hidden_size = hidden_size+5

    # plot and NMSE of the model
    points_sampled = int(np.ceil(points_sampled/2))
    data_handler.remove_points(points_sampled)
    _,input_data,_ = data_handler.create_tensors()
    input_data = input_data.to(device)
    azimuth = data_handler.azimuth
    colatitude = data_handler.colatitude
    normalized_output_data = data_handler.NORMALIZED_OUTPUT_DATA
    #previsions_pinn = model.make_previsions(input_data)
    #previsions_pinn = previsions_pinn.cpu().detach().numpy()

    previsions_no_pinn = model_no_pinn.make_previsions(input_data)
    previsions_no_pinn = previsions_no_pinn.cpu().detach().numpy()


    #utils.plot_model(azimuth,colatitude,normalized_output_data,previsions_pinn,points_sampled,True)
    utils.plot_model(azimuth,colatitude,normalized_output_data,previsions_no_pinn,points_sampled)

# %%
