#%%
import data as dt
import loss_functions as lf
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
        #"hidden_size":{"values":[256,128,64]},
        #"layer":{"max":4,"min":2},
        "pde_weights":{"max":5e-9,"min":1e-9},
        #"data_weights":{"max":2.0,"min":0.001},
        #"c":{"max":20,"min":1},
        #"w0":{"max":20,"min":1},
        #"w0_initial":{"max":20,"min":1}                                          
    },
}

# Initialize sweep by passing in config.
sweep_id = wandb.sweep(sweep=sweep_configuration, project=f"UPSAMPLING-nopinn-{gb.points_sampled}-pinn_ef")


#%%

def train():
    epochs = 100000
    with wandb.init():

        #HYPERPARAMETERS
        hidden_size = 256
        layers = 5
        batch_size = 128
        learning_rate = 0.0001
        data_weights = 1
        pde_weights = wandb.config.pde_weights
        bc_weights=0
        c=1
        w0=2
        w0_initial = 15
        M = 3

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
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-3)
        inputs_not_sampled= data_handler.X_data
        
        #TRAINING

        pinn=True

        print(pinn)
        if pinn:
            print("Start Training PINN")
            wandb.run.name = f"hidden_size_{hidden_size}_{points_sampled}_pinn"
        else:
            print("Start Training NO PINN")
            wandb.run.name = f"hidden_size_{hidden_size}_{points_sampled}_no_pinn"

        # Set up early stopping parameters.
        best_val_loss = float('inf')
        patience = 300
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

            if counter >= patience:
                print(f"Early stopping after {epoch + 1} epochs.")
                counter=0
                break

        input_data = data_handler.X_data
        input_data = input_data.to(gb.device)
        previsions_pinn = model.make_previsions(input_data)

        previsions_pinn = previsions_pinn.cpu().detach().numpy()
        fft_result_full = np.concatenate((previsions_pinn, np.conj(np.fliplr(previsions_pinn[:, 1:]))), axis=1)
        previsions_time = np.real(np.fft.ifft(fft_result_full))

        # COMPUTE NMSE
        drir_ref = data_handler.drir
        drir_ref = torch.tensor(drir_ref, dtype=torch.float32)
        drir_prev = torch.tensor(previsions_time, dtype=torch.float32)
        mean_nmse_time, nmse_time = lf.NMSE(drir_ref, drir_prev)

        wandb.log({"nmse": mean_nmse_time})

        print('FINISHED')

if __name__=="__main__":
    #TRAINING 
    wandb.agent(sweep_id= sweep_id, project=f"UPSAMPLING-nopinn-{gb.points_sampled}-pinn_ef",function=train)

# %%
