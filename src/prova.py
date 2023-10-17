#%%
import data as dt
import global_variables as gb
import fnn
import torch.optim as optim

path_data = "../dataset/DRIR_CR1_VSA_1202RS_R.sofa"
data_handler = dt.DataHandler(path_data,gb.frequency)
points_sampled =14
data_handler.remove_points(2)
train_dataset,val_dataset,inputs_not_sampled = data_handler.data_loader(128)

# %%
 #CREATE_NETWORK
model = fnn.PINN(gb.input_dim,gb.output_dim,22)
model = model.to(gb.device)
#CREATE OPTIMIZER
optimizer = optim.Adam(model.parameters(), lr=0.001)

pinn=False

best_val_loss = float('inf')
patience = 15
counter = 0

print(gb.device)
for epoch in range(10000):
    loss,loss_data,loss_pde,loss_bc = model.train_epoch(train_dataset,inputs_not_sampled,optimizer,1,10,0,points_sampled,pinn=pinn)
    val_loss = model.test_epoch(val_dataset)

    if epoch % 1 == 0:
        print(f'Epoch [{epoch}/{10000}], Loss: {loss.item()},Val_Loss:{val_loss.item()}')

        # Check for early stopping criterion



    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if(counter== 5):
            learning_rate = 0.001/10
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            print(f"Decrease learning rate {learning_rate} epochs.")
                  
    if counter >= patience:
        print(f"Early stopping after {epoch + 1} epochs.")
        break

        
nmse = model.test(data_handler)
        
# %%
import utils
import numpy as np

 # plot and NMSE of the model;
input_data = data_handler.X_data
input_data = input_data.to(gb.device)
previsions_pinn = model.make_previsions(input_data)
previsions_pinn = previsions_pinn.cpu().detach().numpy()
index = 80

#%%
previsions_pinn = np.squeeze(previsions_pinn[:,index])
utils.plot_model(data_handler,previsions_pinn,points_sampled,index,pinn)

# %%
