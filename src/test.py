# %%
import torch
import torch.nn as nn
import fnn
import global_variables as gb
import pandas as pd
import numpy as np
import data as dt
import loss_functions as lf
import matplotlib.pyplot as plt
import utils
from sound_field_analysis import utils as ut
from sound_field_analysis import plot,process,gen
from plotly.offline import plot as plotof
import plotly.graph_objs as go

#IMPORT PATH
path_saving = "../src/models/models_32_513_14.pth"
path_sarita_freq = "../dataset/nmse_db_sarita_down16-1024.csv"
path_data = "../dataset/dataset_daga/Pos1_DRIR_LS_0.sofa"
path_sarita_time = "../dataset/nmse_CHANNEL_32_14-513.csv"

#EXTRACT DATA
sarita_nmse_freq = pd.read_csv(path_sarita_freq ).values
sarita_nmse = pd.read_csv(path_sarita_time).values
M = 3
data_handler = dt.DataHandler(path_data,M)
data_handler.remove_points(2)
points_sampled =len(data_handler.INPUT_SAMPLED)
gb.points_sampled = points_sampled
print(points_sampled)
train_dataset,val_dataset = data_handler.data_loader(128)
inputs_not_sampled= data_handler.X_data

#CREATE MODEL
model = fnn.PINN(gb.input_dim,gb.output_dim,512,4,1,5,5).to(gb.device)
model.load_state_dict(torch.load(path_saving))
model.eval()



##PRINT RESULTS

#Make previsions
input_data = data_handler.X_data
input_data = input_data.to(gb.device)
previsions_pinn = model.make_previsions(input_data)
mean_nmse_fre,nmse_freq = lf.NMSE_freq(torch.tensor(data_handler.NORMALIZED_OUTPUT,dtype=torch.cfloat),previsions_pinn)
previsions_pinn = previsions_pinn.cpu().detach().numpy()
fft_result_full = np.concatenate((previsions_pinn, np.conj(np.fliplr(previsions_pinn[:, 1:]))), axis=1)
previsions_time = np.real(np.fft.ifft(fft_result_full))

#Compute NMSE in time for each channel and NMSE in frequency
drir_ref = data_handler.drir
drir_ref = torch.tensor(drir_ref,dtype=torch.float32)
drir_prev = torch.tensor(previsions_time,dtype=torch.float32)
mean_nmse_time,nmse_time,std_dev= lf.NMSE(drir_ref,drir_prev)


print("NMSE TIME:",mean_nmse_time,std_dev)
print("NMSE FREQ:",mean_nmse_fre)

#Plot NMSE in time for each channel
plt.figure(1)
plt.plot(10*torch.log10(nmse_time.cpu()))
plt.title('NMSE in Time for Each Channel')

plt.xlabel('Time')
plt.ylabel('NMSE (dB)')
plt.show()

#Plot signal in time
plt.figure(2)
plt.plot(drir_ref[30,:])
plt.plot(drir_prev[30,:],'--')
plt.legend(['DRIR Reference', 'Estimated DRIR'])

plt.xlabel('Time')
plt.ylabel('Amplitude)')
plt.show()

#Plot NMSE in frequency
plt.figure(3)
sarita_nmse_freq = np.array(sarita_nmse_freq)
plt.plot(data_handler.frequencies,10*torch.log10(nmse_freq.cpu()))
plt.plot(data_handler.frequencies,10*np.log10(sarita_nmse_freq.T))

plt.title('NMSE for each frequency')
plt.legend(['SARITA', 'PINN'])

plt.xlabel('Frequency')
plt.ylabel('NMSE (dB)')
plt.show()

##Plot average NMSE for each channel
plt.figure(4)

az = data_handler.azimuth
col = data_handler.colatitude

fig, (ax,ax1) = plt.subplots(1, 2, figsize=(12, 5))
# Creare lo scatter plot

sc = ax.scatter(az, col, c=10*np.log10(nmse_time.cpu()), cmap='coolwarm',vmax=1,vmin=-15)
sc1 = ax1.scatter(az, col, c=10*np.log10(sarita_nmse), cmap='coolwarm',vmax=1,vmin=-15)

azimuth_sampled,colatitude_sampled,_ = ut.cart2sph((data_handler.INPUT_SAMPLED[:,0],data_handler.INPUT_SAMPLED[:,1],
                                                       data_handler.INPUT_SAMPLED[:,2]))
microphone_positions = np.column_stack((azimuth_sampled, colatitude_sampled))

ax.scatter(microphone_positions[:,0],microphone_positions[:,1],facecolors='none',
            color='black', marker='o', label='Microphones')
ax1.scatter(microphone_positions[:,0],microphone_positions[:,1],facecolors='none',
            color='black', marker='o', label='Microphones')

# Impostare manualmente i segnaposti e le etichette sugli assi x e y
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
ax.set_xlabel('Azimuth')
ax.set_ylabel('Colatitude')
ax.set_title('Scatter Plot NMSE')
ax1.set_xlabel('Azimuth')
ax1.set_ylabel('Colatitude')
ax1.set_title('Scatter Plot SARITA NMSE')

plt.show()

#%%
## Plot 3D and Pnm estimation

fftData_prev = previsions_pinn
fftData_ref = data_handler.NORMALIZED_OUTPUT
f = data_handler.frequencies

mean_nmse_fre,nmse_freq = lf.NMSE_freq(torch.tensor(fftData_ref,dtype=torch.cfloat),torch.tensor(fftData_prev,dtype=torch.cfloat))
print(mean_nmse_fre)
NFFT = fftData_ref.shape[1] * 2 - 1
order = 4
Pnm_prev = process.spatFT(fftData_prev, data_handler.grid, order_max=order)
Pnm_ref = process.spatFT(fftData_ref, data_handler.grid, order_max=order)

radial_filter_prev = gen.radial_filter_fullspec(
    order,
    NFFT=NFFT,
    fs=16000,
    array_configuration=data_handler.configuration
)

radial_filter_ref = gen.radial_filter_fullspec(
    order,
    NFFT=NFFT,
    fs=16000,
    array_configuration=data_handler.configuration
)

vizMTX1 = plot.makeMTX(
    Pnm_prev, radial_filter_prev, kr_IDX=ut.nearest_to_value_IDX(f, 100)
)
vizMTX2 = plot.makeMTX(
    Pnm_prev, radial_filter_prev, kr_IDX=ut.nearest_to_value_IDX(f, 1000)
)
vizMTX3 = plot.makeMTX(
    Pnm_prev, radial_filter_prev, kr_IDX=ut.nearest_to_value_IDX(f, 3000)
)
vizMTX4 = plot.makeMTX(
    Pnm_prev, radial_filter_prev, kr_IDX=ut.nearest_to_value_IDX(f, 6000)
)

plot.plot3Dgrid(2, 2, [vizMTX1, vizMTX2, vizMTX3, vizMTX4], "shape",title="PREV")


vizMTX1 = plot.makeMTX(
    Pnm_ref, radial_filter_ref, kr_IDX=ut.nearest_to_value_IDX(f, 100)
)
vizMTX2 = plot.makeMTX(
    Pnm_ref, radial_filter_ref, kr_IDX=ut.nearest_to_value_IDX(f, 1000)
)
vizMTX3 = plot.makeMTX(
    Pnm_ref, radial_filter_ref, kr_IDX=ut.nearest_to_value_IDX(f, 3000)
)
vizMTX4 = plot.makeMTX(
    Pnm_ref, radial_filter_ref, kr_IDX=ut.nearest_to_value_IDX(f, 6000)
)

plot.plot3Dgrid(2, 2, [vizMTX1, vizMTX2, vizMTX3, vizMTX4], "shape",title="REF")


#%%
# Parte reale
import numpy as np
import matplotlib.pyplot as plt

# Creare una matrice di esempio 25x1025
matrice = np.random.rand(25, 1025)

# Plot
plt.figure()

# Calcolare la differenza tra Pnm_prev e Pnm_ref (manca la definizione di queste variabili nel tuo codice originale)
diff_real = np.abs((Pnm_prev - Pnm_ref))
max_err = np.max(diff_real)
# Plot immagine
plt.imshow(diff_real/max_err, cmap='coolwarm', aspect='auto', origin='lower')

# Aggiungere una barra dei colori
plt.colorbar(label='Error')

# Etichette e titoli
plt.xlabel('FFT bins')
plt.ylabel('nm coeff')
plt.title('Error Spherical coefficients')

plt.show()
# %%
