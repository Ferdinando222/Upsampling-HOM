#%%
import data as dt
import global_variables as gb
import fnn
import torch.optim as optim

path_data = "../dataset/DRIR_CR1_VSA_1202RS_R.sofa"
data_handler = dt.DataHandler(path_data,gb.frequency)
points_sampled =14
data_handler.remove_points(2)
train_dataset,val_dataset = data_handler.data_loader(128)
inputs_not_sampled= data_handler.X_data

# %%
 #CREATE_NETWORK
learning_rate = 0.01
model = fnn.PINN(gb.input_dim,gb.output_dim,22)
model = model.to(gb.device)
#CREATE OPTIMIZER
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pinn= True

best_val_loss = float('inf')
patience = 400
counter = 0

data_weights = 1
pde_weights = 0.000000000001
bc_weights = 0.00001

print(gb.device)
for epoch in range(10000):
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
        if(counter %50 == 0):
            learning_rate = learning_rate/2
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
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
input_data = input_data.to(gb.device)
previsions_pinn = best_model.make_previsions(input_data)
previsions_pinn = previsions_pinn.cpu().detach().numpy().flatten()


#%%
previsions_pinn = np.squeeze(previsions_pinn)
utils.plot_model(data_handler,previsions_pinn,points_sampled,pinn)

# %%

import matplotlib.pyplot as plt
import numpy as np

# Estrai la parte reale e la parte immaginaria
real_parts = [z.real for z in data_handler.NORMALIZED_OUTPUT]
imaginary_parts = [z.imag for z in data_handler.NORMALIZED_OUTPUT]


real_parts_sampled = [z.real for z in data_handler.NORMALIZED_OUTPUT_SAMPLED]
imaginary_parts_sampled= [z.imag for z in data_handler.NORMALIZED_OUTPUT_SAMPLED]
# Crea un diagramma sul piano complesso

real_parts_previsions = [z.real for z in previsions_pinn]
imaginary_parts_previsions = [z.imag for z in previsions_pinn]

plt.figure(figsize=(8, 8))
plt.scatter(real_parts, imaginary_parts, color='blue', marker='o')
plt.scatter(real_parts_previsions, imaginary_parts_previsions, color='red', marker='x', label='Previsioni')
plt.scatter(real_parts_sampled, imaginary_parts_sampled, color='black', marker='o', label='SAMPLING')

# Etichette degli assi
plt.xlabel('Parte Reale')
plt.ylabel('Parte Immaginaria')

# Titolo del grafico
plt.title('Numeri Complessi sul Piano Complesso')

# Visualizza il grafico
plt.grid()
plt.axhline(0, color='black', lw=0.5)
plt.axvline(0, color='black', lw=0.5)
plt.show()







# %%