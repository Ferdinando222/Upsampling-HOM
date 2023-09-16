#%%
from sound_field_analysis import io
import matplotlib.pyplot as plt
import numpy as np
import fnn
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
import loss_functions
from sklearn.preprocessing import MinMaxScaler
import random
# %%
# Hyperparameters
input_dim = 3 
output_dim = 1 
hidden_size = 2
epochs = 20000
batch_size = 64
numero_punti_da_campionare = 1202

#Prepare Data
DRIR = io.read_SOFA_file("../Spherical-microphone-array-upsampling/dataset/DRIR_CR1_VSA_1202RS_R.sofa")
grid = DRIR.grid
azimuth = grid.azimuth
colatitude = grid.colatitude
radius = grid.radius


input_data = list(zip(azimuth, colatitude, radius))
output_data = np.fft.fft(DRIR.signal.signal)[:, 25]


#Sampling points
input_sampled = []
output_sampled = []
index_sampling = random.sample(range(len(input_data)), numero_punti_da_campionare)
for index in index_sampling:
    input_sampled.append(input_data[index])
    output_sampled.append(output_data[index])

# Separate the real and imaginary parts
real_part = np.real(output_data)
imaginary_part = np.imag(output_data)
real_part_sampled = np.real(output_sampled)
imaginary_part_sampled = np.imag(output_sampled)

real_scaler = MinMaxScaler()
normalized_real_part = real_scaler.fit_transform(real_part.reshape(-1, 1))
real_scaler_sampled = MinMaxScaler()
normalized_real_part_sampled = real_scaler_sampled.fit_transform(real_part_sampled.reshape(-1, 1))

imaginary_scaler = MinMaxScaler()
normalized_imaginary_part = imaginary_scaler.fit_transform(imaginary_part.reshape(-1, 1))
imaginary_scaler_sampled = MinMaxScaler()
normalized_imaginary_part_sampled = imaginary_scaler_sampled.fit_transform(imaginary_part_sampled.reshape(-1, 1))

# Combine the normalized real and imaginary parts
normalized_output_data = normalized_real_part + 1j * normalized_imaginary_part
normalized_output_sampled = normalized_real_part_sampled + 1j * normalized_imaginary_part_sampled
# Prepare the input data as tensors
X_data = torch.tensor(input_sampled, dtype=torch.float32)

Y_data = torch.tensor(normalized_output_sampled,dtype=torch.complex32)




#%%
# Start Training
for size in range(64):
    print("Hidden_size:",hidden_size)
    writer = SummaryWriter(f"runs/hidden_size_{hidden_size}")
    writer_nmse = SummaryWriter("result/nmse-size")
    model = fnn.PINN(input_dim, output_dim,hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):

        predictions = model(X_data)

        loss = loss_functions.CombinedLoss(1,1)(
            predictions,Y_data)
        
        writer.add_scalar("Loss/train", loss, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item()}')
    writer.close()

    previsions =[]

    for data in input_data:
        with torch.no_grad():
            temp = np.array(model(torch.tensor(data, dtype=torch.float32)))
        previsions.append(temp)
    previsions = np.array(previsions).flatten()

    nmse = loss_functions.NMSE(normalized_output_data,previsions)
    writer_nmse.add_scalar("NMSE",nmse,hidden_size)
    hidden_size = hidden_size+1

# %%

# plot and NMSE of the model

# Crea un grafico 2D della pressione in funzione di azimuth e colatitude
fig, (ax,ax1) = plt.subplots(1, 2, figsize=(12, 5))
sc = ax.scatter(azimuth, colatitude, c=np.abs(previsions), cmap='viridis')
sc1 = ax1.scatter(azimuth, colatitude, c=np.abs(normalized_output_data), cmap='viridis')

# Imposta manualmente i segnaposti e le etichette sugli assi x e y
azimuth_ticks = [0, np.pi / 2, np.pi, 2 * np.pi]
colatitude_ticks = [0, np.pi / 2, np.pi]

ax.set_xticks(azimuth_ticks)
ax.set_yticks(colatitude_ticks)
ax1.set_xticks(azimuth_ticks)
ax1.set_yticks(colatitude_ticks)
# Imposta manualmente le etichette degli assi
ax.set_xticklabels(['0', 'π/2', 'π', '2π'])
ax.set_yticklabels(['0', 'π/2', 'π'])
ax1.set_xticklabels(['0', 'π/2', 'π', '2π'])
ax1.set_yticklabels(['0', 'π/2', 'π'])


ax.set_xlabel('Azimuth (radians)')
ax.set_ylabel('Colatitude (radians)')
ax.set_title('Previsions')
ax.grid(True)

ax1.set_xlabel('Azimuth (radians)')
ax1.set_ylabel('Colatitude (radians)')
ax1.set_title('Ground Truth')
ax1.grid(True)

plt.colorbar(sc1,label='Pressure')
plt.colorbar(sc, label='Pressure')
plt.show()


#TODO vari plot con tensorboard:
# 1) Cambiare numeri di microfoni
# 2) Cambiare la frequenza
# 3) come tracciare andamento nmse con tensorboard
# %%
