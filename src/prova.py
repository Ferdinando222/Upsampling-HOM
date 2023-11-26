#%%
import data as dt
import global_variables as gb
import fnn
import torch.optim as optim

path_data = "../dataset/DRIR_CR1_VSA_1202RS_R.sofa"
#DOWNSAMPLING FACTOR
M = 1
data_handler = dt.DataHandler(path_data,M)
data_handler.remove_points(4)
points_sampled =len(data_handler.INPUT_SAMPLED)
gb.points_sampled = points_sampled
print(points_sampled)
train_dataset,val_dataset = data_handler.data_loader(32)
inputs_not_sampled= data_handler.X_data

# %%
 #CREATE_NETWORK
learning_rate = 0.001
model = fnn.PINN(gb.input_dim,gb.output_dim,1024,4,2,6,1)
model = model.to(gb.device)
#CREATE OPTIMIZER
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pinn=False

best_val_loss = float('inf')
patience = 2000
counter = 0

data_weights = 1
pde_weights = 1e-11
bc_weights = 0

print(gb.device)
for epoch in range(100000):
    loss,loss_data,loss_pde,loss_bc = model.train_epoch(train_dataset,inputs_not_sampled,optimizer,data_weights,pde_weights,bc_weights,points_sampled,pinn=pinn)
    val_loss = model.test_epoch(val_dataset)

    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/{100000}], Loss: {loss.item()},Val_Loss:{val_loss.item()}')

    # Check for early stopping criterion

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
        counter = 0
    else:
        counter += 1
        if(counter %125 == 0 and learning_rate> 0.0001):
            learning_rate = learning_rate/10
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            print(f"Decrease learning rate {learning_rate} epochs.")
#
    if counter  >= patience:
        print(f"Early stopping after {epoch + 1} epochs.")
        counter = 0
        break

        
nmse = model.test(data_handler)
        
# %%
import utils
import numpy as np


# plot and NMSE of the model;
input_data = data_handler.X_data
input_data = input_data.to(gb.device)
previsions_pinn = model.make_previsions(input_data)

#%%
index = 2
previsions_pinn = previsions_pinn.cpu().detach().numpy()[:,index]

#%%
previsions_pinn = np.squeeze(previsions_pinn)

#%%
utils.plot_model(data_handler,previsions_pinn,points_sampled,pinn,index)

# %%

# plot comparison nmse btw sarita and PINN

import pandas as pd
import matplotlib.pyplot as plt

path_sarita = '../dataset/nmse_db_sarita.csv'
path_no_pinn = '../dataset/nmse_nopinn.csv'
nmse_sarita = pd.read_csv(path_sarita).to_numpy().transpose().flatten()
nmse_no_pinn = pd.read_csv(path_no_pinn).to_numpy().transpose().flatten()

nmse_mean = np.mean(nmse)
nmse_sar_mean = np.mean(nmse_sarita[1::500])
nmse_no_pinn_mean = np.mean(nmse_no_pinn)

# Creazione del grafico con legende personalizzate
plt.plot(gb.frequency,nmse, label='NMSE No Pinn')
plt.plot(gb.frequency,nmse_sarita[1::500], label='Sarita')
#plt.plot(gb.frequency,nmse_no_pinn, label='NMSE No PINN')

# Aggiungi etichette agli assi
plt.xlabel('FREQUENCY')
plt.ylabel('NMSE (DB)')
plt.grid(True)
# Aggiungi una legenda
plt.legend()

# Mostra il grafico
plt.show()


#%%
## SAVE RESULT
##import csv
##
##csv_file_path = '../dataset/nmse_nopinn.csv'
##
##with open(csv_file_path, 'w', newline='') as csv_file:
##    csv_writer = csv.writer(csv_file)
##    csv_writer.writerow(nmse)

