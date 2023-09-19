#%%
import data as dt
import utils
import fnn

# Hyperparameters
frequency = 1000
input_dim = 3 
output_dim = 1 
hidden_size = 10
epochs = 10
batch_size = 64
points_sampled = 10
path_data = "../dataset/DRIR_CR1_VSA_1202RS_R.sofa"

#
print("Start data handling")
data_handler = dt.DataHandler(path_data,frequency,points_sampled)


# Start Training
print("Start Training")
model = fnn.PINN(input_dim,output_dim,hidden_size)
model.train_PINN(data_handler,hidden_size,frequency,epochs,points_sampled)

# plot and NMSE of the model
#%%
_,input_data,_ = data_handler.create_tensors()
azimuth = data_handler.azimuth
colatitude = data_handler.colatitude
normalized_output_data = data_handler.NORMALIZED_OUTPUT_DATA
previsions = model.make_previsions(input_data)
utils.plot_model(azimuth,colatitude,normalized_output_data,previsions,points_sampled)


# %%
