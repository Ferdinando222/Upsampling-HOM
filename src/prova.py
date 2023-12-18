#%%
import data as dt
import numpy as np
import global_variables as gb
import loss_functions as lf
import fnn
import torch
import torch.optim as optim

path_data = "../dataset/DRIR_CR1_VSA_1202RS_R.sofa"
#DOWNSAMPLING FACTOR
M = 3
NFFT = int(np.round(17000/M))
data_handler = dt.DataHandler(path_data,M)
data_handler.remove_points(4)
points_sampled =len(data_handler.INPUT_SAMPLED)
gb.points_sampled = points_sampled
print(points_sampled)
train_dataset,val_dataset = data_handler.data_loader(128)
inputs_not_sampled= data_handler.X_data

# %%
 #CREATE_NETWORK
learning_rate = 0.0001
model = fnn.PINN(gb.input_dim,gb.output_dim,256,5,2,15,1)
model = model.to(gb.device)
#CREATE OPTIMIZER
optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-3)

pinn= True

best_val_loss = float('inf')
patience = 200
counter = 0

data_weights = 1
pde_weights = 1e-9
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

previsions_pinn = previsions_pinn.cpu().detach().numpy()
fft_result_full = np.concatenate((previsions_pinn, np.conj(np.fliplr(previsions_pinn[:, 1:]))), axis=1)
previsions_time = np.real(np.fft.ifft(fft_result_full))

#COMPUTE NMSE
drir_ref = data_handler.drir
drir_ref = torch.tensor(drir_ref,dtype=torch.float32)
drir_prev = torch.tensor(previsions_time,dtype=torch.float32)
mean_nmse_time,nmse_time= lf.NMSE(drir_ref,drir_prev)

print(mean_nmse_time)

import matplotlib.pyplot as plt
plt.plot(10*torch.log10(nmse_time))

#%%

plt.plot(drir_ref[200,:])
plt.plot(drir_prev[200,:],'--')
plt.show()



#%%


import plotly.graph_objects as go
import numpy as np


# Creazione del grafico interattivo
fig = go.Figure()
idx = 600

# Aggiunta delle tracce al grafico
fig.add_trace(go.Scatter(y=drir_ref[idx,:], mode='lines', name='drir_ref'))
fig.add_trace(go.Scatter(y=drir_prev[idx,:], mode='lines', name='drir_prev', line=dict(dash='dash')))

# Aggiornamento del layout del grafico
fig.update_layout(
    title='Grafico Zoomabile',
    xaxis_title='Indice',
    yaxis_title='Valore',
)

# Abilitazione della funzione di zoom
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True),
        type="linear"
    )
)

# Mostra il grafico
fig.show()




# %%
import utils
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
