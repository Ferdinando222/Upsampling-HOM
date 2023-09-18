#%%
import data as dt
import utils
import train
import fnn
import torch
# %%
# Hyperparameters
frequency = 1000
input_dim = 3 
output_dim = 1 
hidden_size = 10
epochs = 20000
batch_size = 64
points_sampled = 10

# Extract Data

input_data,output_data,azimuth,colatitude,radius = dt.extract_data(frequency)

x,y,z = dt.convert_cartesian(azimuth,colatitude,radius)

# Sampling points
input_sampled,output_sampled = dt.sampling_points(points_sampled)

# Normalization
normalized_output_data,normalized_output_sampled = dt.normalize_data(output_sampled)

# Create Tensors
X_data,X_data_not_sampled,Y_data = dt.create_tensors(input_sampled,normalized_output_sampled)




#%%
# Start Training
model_PINN = train.train_PINN(points_sampled,input_dim,output_dim,epochs,X_data,Y_data,X_data_not_sampled,frequency,input_data,normalized_output_data,hidden_size)
previsions_PINN = train.make_previsions(input_data,model_PINN)

#%%

utils.plot_model(azimuth,colatitude,normalized_output_data,previsions_PINN,points_sampled)
# plot and NMSE of the model
# %%
