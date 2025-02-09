#%%
import data as dt
import numpy as np
import global_variables as gb
import loss_functions as lf
import fnn
import torch
import torch.optim as optim
import loss_functions
from torch.optim.lr_scheduler import CosineAnnealingLR

path_data = "../dataset/dataset_daga/Pos1_DRIR_LS_0.sofa"
#DOWNSAMPLING FACTOR
M = 3
NFFT = int(np.round(17000/M))
data_handler = dt.DataHandler(path_data,M)
data_handler.remove_points(4,"fliege")
points_sampled =len(data_handler.INPUT_SAMPLED)
gb.points_sampled = points_sampled
print(points_sampled)
train_dataset,val_dataset = data_handler.data_loader(128)
inputs_not_sampled= data_handler.X_data

# %%
 #CREATE_NETWORK
learning_rate = 0.0001
model = fnn.PINN(gb.input_dim,gb.output_dim,512,4,1.0,5.0,5,rowdy=True)
model = model.to(gb.device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

# %%

#CREATE OPTIMIZER
optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay =1e-3)
scheduler = CosineAnnealingLR(optimizer,T_max=10000,eta_min=0)
pinn= False
loss_comb= loss_functions.CombinedLoss(pinn)

best_val_loss = float('inf')
patience =200
counter = 0

print(gb.device)
for epoch in range(10000):
    loss,loss_data,loss_pde,loss_bc = model.train_epoch(train_dataset,inputs_not_sampled,optimizer,loss_comb,points_sampled,pinn=pinn)
    val_loss = model.test_epoch(val_dataset)
    scheduler.step()


    print(f'Epoch [{epoch}/{100000}],Loss_data:{loss_data.item()},Loss_bc:{loss_bc.item()},Loss_pde:{gb.e_f*loss_pde.item()},Loss: {loss.item()},Val_Loss:{val_loss.item()}')

    # Check for early stopping criterion

    if val_loss < best_val_loss:
        print("Reached best val loss")
        best_val_loss = val_loss
        best_model = model
        counter = 0
    else:
        counter += 1
#

    # if counter >= patience:
    #     print(f"Early stopping after {epoch + 1} epochs.")
    #     counter = 0
    #     break

        
nmse = model.test(data_handler)
        
# %%
import utils
import pandas as pd
import numpy as np
percorso_file_csv = "../dataset/results_sarita/nmse_freq_sarita_16.csv"

# Leggi il file CSV
sarita_nmse = pd.read_csv(percorso_file_csv).values


# plot and NMSE of the model;
input_data = data_handler.X_data
input_data = input_data.to(gb.device)
previsions_pinn = best_model.make_previsions(input_data)


mean_nmse_fre,nmse_freq = lf.NMSE_freq(torch.tensor(data_handler.NORMALIZED_OUTPUT,dtype=torch.cfloat),previsions_pinn)

previsions_pinn = previsions_pinn.cpu().detach().numpy()
fft_result_full = np.concatenate((previsions_pinn, np.conj(np.fliplr(previsions_pinn[:, 1:]))), axis=1)
previsions_time = np.real(np.fft.ifft(fft_result_full))

#COMPUTE NMSE
drir_ref = data_handler.drir
drir_ref = torch.tensor(drir_ref,dtype=torch.float32)
drir_prev = torch.tensor(previsions_time,dtype=torch.float32)
mean_nmse_time,nmse_time,_= lf.NMSE(drir_ref,drir_prev)

print(mean_nmse_time)

import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(10*torch.log10(nmse_time.cpu()))
plt.show()
#%%

plt.figure(2)
plt.plot(drir_ref[30,:])
plt.plot(drir_prev[30,:],'--')
plt.show()

plt.figure(3)
sarita_nmse = np.array(sarita_nmse)
plt.plot(data_handler.frequencies,10*torch.log10(nmse_freq.cpu()))
plt.plot(data_handler.frequencies,10*np.log10(sarita_nmse.T),)
plt.show()

# %%
import utils
# plot and NMSE of the model;
input_data = data_handler.X_data
input_data = input_data.to(gb.device)
previsions_pinn = model.make_previsions(input_data)

#%%
index = 50
previsions_pinn = previsions_pinn.cpu().detach().numpy()[:,index]

#%%
previsions_pinn = np.squeeze(previsions_pinn)

#%%
utils.plot_model(data_handler,previsions_pinn,points_sampled,pinn,index)

#%%
from sound_field_analysis import utils as ut
import matplotlib.colors as mcolors
plt.figure(4)

az = data_handler.colatitude
col = data_handler.azimuth


# Creare lo scatter plot

sc = plt.scatter(az, col, c=10*np.log10(nmse_time.cpu()), cmap='coolwarm',vmax=1,vmin=-15)
azimuth_sampled,colatitude_sampled,_ = ut.cart2sph((data_handler.INPUT_SAMPLED[:,0],data_handler.INPUT_SAMPLED[:,1],
                                                       data_handler.INPUT_SAMPLED[:,2]))
microphone_positions = np.column_stack((azimuth_sampled, colatitude_sampled))

plt.scatter(microphone_positions[:,0],microphone_positions[:,1],facecolors='none',
            color='black', marker='o', label='Microphones')
# Impostare manualmente i segnaposti e le etichette sugli assi x e y
azimuth_ticks = [0, np.pi / 2, np.pi, 2 * np.pi]
colatitude_ticks = [0, np.pi / 2, np.pi]
cbar2 = plt.colorbar(sc, label='Pressure Difference')

plt.xticks(azimuth_ticks, ['0', 'π/2', 'π', '2π'])
plt.yticks(colatitude_ticks, ['0', 'π/2', 'π'])

plt.show()

percorso_file_csv = "../dataset/nmse_CHANNEL_86_4-1024.csv"
sarita_nmse = pd.read_csv(percorso_file_csv).values

plt.figure(5)

az = data_handler.azimuth
col = data_handler.colatitude


# Creare lo scatter plot

sc = plt.scatter(az, col, c=10*np.log10(sarita_nmse), cmap='coolwarm',vmax=1,vmin=-15)
azimuth_sampled,colatitude_sampled,_ = ut.cart2sph((data_handler.INPUT_SAMPLED[:,0],data_handler.INPUT_SAMPLED[:,1],
                                                       data_handler.INPUT_SAMPLED[:,2]))
microphone_positions = np.column_stack((azimuth_sampled, colatitude_sampled))

plt.scatter(microphone_positions[:,0],microphone_positions[:,1],facecolors='none',
            color='black', marker='o', label='Microphones')
# Impostare manualmente i segnaposti e le etichette sugli assi x e y
azimuth_ticks = [0, np.pi / 2, np.pi, 2 * np.pi]
colatitude_ticks = [0, np.pi / 2, np.pi]
cbar2 = plt.colorbar(sc, label='Pressure Difference')

plt.xticks(azimuth_ticks, ['0', 'π/2', 'π', '2π'])
plt.yticks(colatitude_ticks, ['0', 'π/2', 'π'])

plt.show()

# %%
