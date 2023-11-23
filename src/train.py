#%%
import data as dt
import utils
import pandas as pd
import matplotlib.pyplot as plt
import fnn
import numpy as np
import torch.optim as optim
import wandb
import torch
import global_variables as gb


# Define sweep config
sweep_configuration = {
    "method": "bayes",
    "name": "Training pinn with different hidden size",
    "metric": {"goal": "minimize", "name": "nmse"},
    "parameters": {
        #"batch_size": {"values": [16,32,64,128]},
        #"learning_rate": {"values":[0.01,0.001,0.0001]},
        #"hidden_size":{"values":[1024,512,256,128,64,32,22,12]},
        #"layer":{"max":10,"min":1},
        #"pde_weights":{"max":0.1,"min":0.000000001},
        #"data_weights":{"max":2.0,"min":0.001},
        "c":{"max":10,"min":1},
        "w0":{"max":10,"min":1},
        "w0_initial":{"max":10,"min":1}
    },
}

# Initialize sweep by passing in config.
sweep_id = wandb.sweep(sweep=sweep_configuration, project=f"UPSAMPLING-pinn-{gb.points_sampled}-pinn_ef")


#%%

def train():
    epochs = 100000
    with wandb.init():

        #HYPERPARAMETERS
        hidden_size = 256
        layers = 2
        batch_size = 32
        learning_rate = 0.0001
        data_weights = 1
        pde_weights = 1
        bc_weights=0
        c=wandb.config.c
        w0=wandb.config.w0
        w0_initial = wandb.config.w0_initial
        M = 1

        #CREATE DATASET
        path_data = "../dataset/DRIR_CR1_VSA_1202RS_R.sofa"
        data_handler = dt.DataHandler(path_data,M)
        data_handler.remove_points(4)
        points_sampled =len(data_handler.INPUT_SAMPLED)
        gb.points_sampled = points_sampled
        train_dataset,val_dataset = data_handler.data_loader(batch_size)

        #CREATE_NETWORK
        model = fnn.PINN(gb.input_dim,gb.output_dim,hidden_size,layers,w0,w0_initial,c)
        model = model.to(gb.device)

        #CREATE OPTIMIZER
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        inputs_not_sampled= data_handler.X_data
        
        #TRAINING

        pinn=False
        if pinn:
            print("Start Training PINN")
            wandb.run.name = f"hidden_size_{hidden_size}_{points_sampled}_pinn"
        else:
            print("Start Training NO PINN")
            wandb.run.name = f"hidden_size_{hidden_size}_{points_sampled}_no_pinn"

        # Set up early stopping parameters.
        best_val_loss = float('inf')
        patience = 250
        counter = 0
        
        for epoch in range(epochs):
            loss,loss_data,loss_pde,loss_bc = model.train_epoch(train_dataset,inputs_not_sampled,optimizer,data_weights,pde_weights,bc_weights,points_sampled,pinn=pinn)
            val_loss = model.test_epoch(val_dataset)

            wandb.log({
                "epoch":epoch,
                "loss":loss,
                "loss_data":loss_data,
                "loss_pde":torch.mean(loss_pde),
                "loss_bc":loss_bc,
                "val_loss":val_loss
            }
            )
            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item()},Val_Loss:{val_loss.item()}')

            # Check for early stopping criterion
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                counter = 0
            else:
                counter += 1
                #if(counter %25 == 0 and learning_rate>0.00001):
                #    learning_rate = learning_rate/10
                #    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                #    print(f"Decrease learning rate {learning_rate} epochs.")

            if counter >= patience:
                print(f"Early stopping after {epoch + 1} epochs.")
                counter=0
                break
    

        nmse = best_model.test(data_handler)
        wandb.log({"nmse": np.mean(np.array(nmse))})


    path_sarita = '../dataset/nmse_db_sarita.csv'
    nmse_sarita = pd.read_csv(path_sarita).to_numpy().transpose().flatten()

    # Creazione del grafico con legende personalizzate
    plt.plot(gb.frequency,nmse, label='NMSE Pinn')
    plt.plot(gb.frequency,nmse_sarita[1:1417], label='Sarita')

    # Aggiungi etichette agli assi
    plt.xlabel('FREQUENCY')
    plt.ylabel('NMSE (DB)')

    # Aggiungi una legenda
    plt.legend()

    # Mostra il grafico
    plt.show()
    print('FINISHED')

if __name__=="__main__":
    #TRAINING 
    wandb.agent(sweep_id= sweep_id, project=f"UPSAMPLING-pinn-{gb.points_sampled}-pinn_ef",function=train)

# %%
