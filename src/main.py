#%%
from sound_field_analysis import io
import matplotlib.pyplot as plt
import numpy as np
import fnn
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
import loss_functions
import data as dt
# %%
# Hyperparameters
frequency = 1000
input_dim = 3 
output_dim = 1 
hidden_size = 2
epochs = 20000
batch_size = 64
points_sampled = 10

# EXTRACT DATA

input_data,output_data,azimuth,colatitude = dt.extract_data(frequency)


# Sampling points
input_sampled,output_sampled = dt.sampling_points(points_sampled)

# Normalization
normalized_output_data,normalized_output_sampled = dt.normalize_data(output_sampled)

# Create Tensors
X_data,X_data_not_sampled,Y_data = dt.create_tensors(input_sampled,normalized_output_sampled)




#%%
# Start Training
for size in range(10):
    print("Hidden_size:",hidden_size)
    writer = SummaryWriter(f"runs_pde/hidden_size_{hidden_size}_{points_sampled}")
    writer_nmse = SummaryWriter(f"result_pde/nmse-size_{points_sampled}")
    model = fnn.PINN(input_dim, output_dim,hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):

        predictions = model(X_data)

        loss = loss_functions.CombinedLoss(1,1,frequency=frequency)(
            predictions,Y_data,X_data_not_sampled)
        
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
plt.savefig(f"../src/image/prevPinn_{points_sampled}.png")  

plt.show()
# Save



#TODO vari plot con tensorboard:
# 2) Cambiare la frequenza
# %%
