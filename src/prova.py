#%%
import data as dt
import global_variables as gb
import fnn
import torch.optim as optim

path_data = "../dataset/DRIR_CR1_VSA_1202RS_R.sofa"
data_handler = dt.DataHandler(path_data)
points_sampled =38
data_handler.remove_points(11)
train_dataset,val_dataset = data_handler.data_loader(128)
inputs_not_sampled= data_handler.X_data
time = data_handler.time_values.repeat(len(inputs_not_sampled),1)

# %%
 #CREATE_NETWORK
learning_rate = 0.1
model = fnn.PINN(gb.input_dim,gb.output_dim,22,1)
model = model.to(gb.device)
#CREATE OPTIMIZER
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pinn=False

best_val_loss = float('inf')
patience = 200
counter = 0

data_weights = 1
pde_weights = 1
bc_weights = 0

print(gb.device)
for epoch in range(100000000):
    loss,loss_data,loss_pde,loss_bc = model.train_epoch(train_dataset,inputs_not_sampled,time,optimizer,data_weights,pde_weights,bc_weights,points_sampled,pinn=pinn)
    val_loss = model.test_epoch(val_dataset)

    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/{10000}], Loss: {loss.item()},Val_Loss:{val_loss.item()}')

    # Check for early stopping criterion

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
        counter = 0
    else:
        counter += 1
        if(counter %50 == 0):
            learning_rate = learning_rate/5
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            print(f"Decrease learning rate {learning_rate} epochs.")
                  
    if counter >= patience:
        print(f"Early stopping after {epoch + 1} epochs.")
        break

        
nmse = best_model.test(data_handler)
        
# %%
import utils
import numpy as np

# plot and NMSE of the model;
nmse = best_model.test(data_handler)
input_data = data_handler.X_data
time = data_handler.time_values.repeat(len(input_data),1)
input_data = input_data.to(gb.device)
previsions_pinn = best_model.make_previsions(input_data,time)
previsions_pinn = previsions_pinn.cpu().detach().numpy()


#%%
norm = data_handler.NORMALIZED_OUTPUT
utils.plot_model(data_handler,previsions_pinn,points_sampled,time[0],pinn)


# %%
