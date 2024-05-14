# %%
import torch
import fnn
import global_variables as gb
import pandas as pd
import numpy as np
import data as dt
import loss_functions as lf
import matplotlib.pyplot as plt
import utils
import binaural
from sound_field_analysis import utils as ut
from sound_field_analysis import process,gen



#------------------------- SIMPLE GUI FOR USER --------------------------------#

options_points = ["4", "9", "16", "25"]

print("For which points do you want to test the model?")

print("Options:")
for i, option in enumerate(options_points, start=1):
    print(f"{i}) {option}")

while True:
    index = input("Please specify the index of the points you want to test: ")

    # Validate the input
    try:
        index_points = int(index)
        if 1 <= index_points <= len(options_points):
            points = options_points[index_points- 1]
            print("Chosen points:", points)
            break  # Exit the loop if the input is valid
        else:
            print("Invalid index. Please enter a number between 1 and", len(options_points))
    except ValueError:
        print("Invalid input. Please enter a number.")


options_model = ["Siren","Siren+Pde","Siren+Pde+Rowdy"]

print("Which model?")

print("Options:")
for i, option in enumerate(options_model, start=1):
    print(f"{i}) {option}")

while True:
    index = input("Please specify the index of the model you want to test: ")

    # Validate the input
    try:
        index_model = int(index)
        if 1 <= index_model <= len(options_model):
            models = options_model[index_model - 1]
            print("Chosen points:", points)
            break  # Exit the loop if the input is valid
        else:
            print("Invalid index. Please enter a number between 1 and", len(options_model))
    except ValueError:
        print("Invalid input. Please enter a number.")

if(models == options_model[0]):
    models = "False"
elif(models == options_model[1]):
    models = "True"
elif(models == options_model[2]):
    models="True_rowdy"

#####################################################################################################

#---------------------------------- IMPORTING PATH AND EXTRACT DATA --------------------------------#

#####################################################################################################

# IMPORT PATH
path_saving = f"../src/models/models_32_{models}_513_{points}.pth"
path_sarita_freq = f"../dataset/results_sarita/nmse_freq_sarita_{points}.csv"
path_data = "../dataset/dataset_daga/Pos1_DRIR_LS_0.sofa"
path_sarita_time = f"../dataset/results_sarita/nmse_time_sarita_{points}.csv"
path_signal_sarita =f"../dataset/results_sarita/signal_time_{points}.csv"

# EXTRACT DATA
sarita_nmse_freq = pd.read_csv(path_sarita_freq ).values
sarita_nmse = pd.read_csv(path_sarita_time).values
signal_sarita = pd.read_csv(path_signal_sarita).values
M = 3
data_handler = dt.DataHandler(path_data,M)
data_handler.remove_points(index_points,"fliege")
data_handler.compute_sh(4)
points_sampled =len(data_handler.INPUT_SAMPLED)
gb.points_sampled = points
print(points_sampled)
train_dataset,val_dataset = data_handler.data_loader(128)
inputs_not_sampled= data_handler.X_data

# CREATE MODEL
if models == "True_rowdy":
    rowdy=True
else:
    rowdy =False
model = fnn.PINN(gb.input_dim,gb.output_dim,512,4,1.0,5.0,5,rowdy=rowdy).to(gb.device)
model.load_state_dict(torch.load(path_saving))
model.eval()

#########################################################################################################################

#-------------------------------------------------- PRINT RESULTS-------------------------------------------------------#

#########################################################################################################################


# Make previsions
input_data = data_handler.X_data.to(gb.device)
previsions_pinn = model.make_previsions(input_data)
mean_nmse_fre,nmse_freq = lf.NMSE_freq(torch.tensor(data_handler.NORMALIZED_OUTPUT, dtype=torch.cfloat), previsions_pinn)
previsions_pinn = previsions_pinn.cpu().detach().numpy()
prevision_freq = previsions_pinn
fft_result_full = np.concatenate((previsions_pinn, np.conj(np.fliplr(previsions_pinn[:, 1:]))), axis=1)
previsions_time = np.real(np.fft.ifft(fft_result_full))

# Compute NMSE in time for each channel and NMSE in frequency
drir_ref = data_handler.drir
drir_ref = torch.tensor(drir_ref,dtype=torch.float32)
drir_prev = torch.tensor(previsions_time,dtype=torch.float32)
mean_nmse_time,nmse_time,std_dev= lf.NMSE(drir_ref,drir_prev)
diff = np.abs(drir_ref.cpu().numpy()-previsions_time)

print("NMSE TIME:",mean_nmse_time,std_dev)
print("NMSE FREQ:",mean_nmse_fre)

# Plot NMSE in time for each channel
plt.figure(1)
plt.plot(10*torch.log10(nmse_time.cpu()))
plt.show()

# Plot signal in time
plt.figure(2)
plt.plot(drir_ref[30,:],label='Groundtruth')
plt.plot(drir_prev[30,:],'--',label='PINN')
plt.legend()
plt.show()
plt.plot(drir_ref[30,:],label='Groundtruth')
plt.plot(signal_sarita.T,'--',label='SARITA')
plt.legend()
plt.show()

# Plot NMSE in frequency
plt.figure(3)
sarita_nmse_freq = np.array(sarita_nmse_freq)
plt.plot(data_handler.frequencies,10*torch.log10(nmse_freq.cpu()),label="PINN")
plt.plot(data_handler.frequencies,10*np.log10(sarita_nmse_freq.T),label="SARITA")
plt.legend()
plt.show()

## Plot Magnitude signal for each channel with fixed frequency
index = 50
pinn = True
previsions_pinn = previsions_pinn[:,index]
previsions_pinn = np.squeeze(previsions_pinn)
utils.plot_model(data_handler,previsions_pinn,points_sampled,pinn,index)

# Plot average NMSE for each channel
plt.figure(4)
az = data_handler.azimuth
col = data_handler.colatitude
fig, (ax,ax1) = plt.subplots(1, 2, figsize=(12, 5))

# Create scatter plot
sc = ax.scatter(az, col, c=10*np.log10(nmse_time.cpu()), cmap='coolwarm',vmax=1,vmin=-15)
sc1 = ax1.scatter(az, col, c=10*np.log10(sarita_nmse), cmap='coolwarm',vmax=1,vmin=-15)
azimuth_sampled,colatitude_sampled,_ = ut.cart2sph((data_handler.INPUT_SAMPLED[:,0],data_handler.INPUT_SAMPLED[:,1],
                                                       data_handler.INPUT_SAMPLED[:,2]))
microphone_positions = np.column_stack((azimuth_sampled, colatitude_sampled))
ax.scatter(microphone_positions[:,0],microphone_positions[:,1],facecolors='none',
            color='black', marker='o', label='Microphones')
ax1.scatter(microphone_positions[:,0],microphone_positions[:,1],facecolors='none',
            color='black', marker='o', label='Microphones')
azimuth_ticks = [0, np.pi / 2, np.pi, 2 * np.pi]
colatitude_ticks = [0, np.pi / 2, np.pi]
cbar = plt.colorbar(sc, label='NMSE (dB)')
cbar1 = plt.colorbar(sc1, label='NMSE (dB)')
ax.set_xticks(azimuth_ticks)
ax.set_yticks(colatitude_ticks)
ax1.set_xticks(azimuth_ticks)
ax1.set_yticks(colatitude_ticks)
ax.set_xticklabels(['0', 'π/2', 'π', '2π'])
ax.set_yticklabels(['0', 'π/2', 'π'])
ax1.set_xticklabels(['0', 'π/2', 'π', '2π'])
ax1.set_yticklabels(['0', 'π/2', 'π'])
plt.show()

# Compute nmse for sh armonics
sh_ground = process.spatFT(prevision_freq,gb.spherical_grid,4)
mse = np.abs(sh_ground - gb.sh_lower) ** 2
mse = np.sum(mse,axis=1)
norm = np.sum(np.abs(gb.sh_lower) ** 2,axis=1)

# Calculate NMSE in dB
nmse_all_sh = mse / norm
nmse_all_sh_db = 10*np.log10(nmse_all_sh)
nmse_sh = 10 * np.log10(np.mean(nmse_all_sh))

plt.plot(nmse_all_sh_db)
plt.xlabel('SH coefficients')
plt.ylabel('NMSE')
plt.title('NMSE')
plt.grid(True)
plt.show()


import csv
#######################################################################################
#---------------------------------------SAVE RESULTS----------------------------------#
#######################################################################################

with open('../results/mean_nmse_time.txt', 'r') as file:
    table = csv.reader(file, delimiter='|')
    results = list(table)


results[index_points][index_model+1] = str(mean_nmse_time.numpy())

with open('../results/mean_nmse_time.txt', 'w', newline='') as file:
    writer = csv.writer(file, delimiter='|')
    writer.writerows(results)

with open('../results/mean_nmse_time.txt', 'r') as file:
    table = csv.reader(file, delimiter='|')
    results = list(table)


results[index_points][index_model+1] = str(mean_nmse_time.numpy())


def create_file(freq, nmse, nome_file):
    with open(nome_file, 'w') as file:
        file.write('Frequency:\n')
        for f in freq:
            file.write(str(f) + ' ')
        
        file.write('\n\n')  

        file.write('Vaue NMSE:\n')
        for n in nmse:
            file.write(str(n) + ' ')



nome_file = f'../results/nmse_freq_{models}_{points}.txt'

create_file(data_handler.frequencies,nmse_freq.cpu().numpy(), nome_file)
print(f"File '{nome_file}' created!")

#%%
## BINAURAL SIGNAL
brir = binaural.compute_binaural_signal(sh_ground,data_handler.NFFT_down,4,data_handler.configuration,True)
# %%
