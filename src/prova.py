#%%
import data as dt
import global_variables as gb
import fnn
import torch.optim as optim

path_data = "../dataset/DRIR_CR1_VSA_1202RS_R.sofa"
data_handler = dt.DataHandler(path_data)
data_handler.remove_points(4)
points_sampled =len(data_handler.INPUT_SAMPLED)
gb.points_sampled = points_sampled
print(points_sampled)
train_dataset,val_dataset = data_handler.data_loader(128)
inputs_not_sampled= data_handler.X_data

# %%
 #CREATE_NETWORK
learning_rate = 0.001
model = fnn.PINN(gb.input_dim,gb.output_dim,512,15)
model = model.to(gb.device)
#CREATE OPTIMIZER
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pinn=False

best_val_loss = float('inf')
patience = 500
counter = 0

data_weights = 1
pde_weights = 0.0001
bc_weights = 0

print(gb.device)
for epoch in range(100000000):
    loss,loss_data,loss_pde,loss_bc = model.train_epoch(train_dataset,inputs_not_sampled,optimizer,data_weights,pde_weights,bc_weights,points_sampled,pinn=pinn)
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
        if(counter %25 == 0 and learning_rate> 0.0001):
            learning_rate = learning_rate/10
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            print(f"Decrease learning rate {learning_rate} epochs.")
                  
    if counter >= patience:
        print(f"Early stopping after {epoch + 1} epochs.")
        counter = 0
        break

        
nmse = best_model.test(data_handler)
        
# %%
import utils
import numpy as np


# plot and NMSE of the model;
input_data = data_handler.X_data
input_data = input_data.to(gb.device)
previsions_pinn = best_model.make_previsions(input_data)

#%%
index = 10
previsions_pinn = previsions_pinn.cpu().detach().numpy()[:,index]

gb.frequency = (index/17000)*48000

#%%
previsions_pinn = np.squeeze(previsions_pinn)

#%%
utils.plot_model(data_handler,previsions_pinn,points_sampled,pinn,index)

# %%

# plot comparison nmse btw sarita and PINN

import pandas as pd
import matplotlib.pyplot as plt

path_sarita = '../dataset/nmse_db_sarita.csv'
nmse_sarita = pd.read_csv(path_sarita).to_numpy().transpose().flatten()

nmse_mean = np.mean(nmse)
nmse_sar_mean = np.mean(nmse_sarita[1::100])
# Creazione del grafico con legende personalizzate
plt.plot(gb.frequency,nmse, label='NMSE Pinn')
plt.plot(gb.frequency,nmse_sarita[1::100], label='Sarita')

# Aggiungi etichette agli assi
plt.xlabel('FREQUENCY')
plt.ylabel('NMSE (DB)')
plt.grid(True)
# Aggiungi una legenda
plt.legend()

# Mostra il grafico
plt.show()

# %%
