#%%
import data as dt
import loss_functions as lf
import fnn
import numpy as np
import torch.optim as optim
import wandb
import torch
import global_variables as gb
import loss_functions


# Define sweep config
sweep_configuration = {
    "method": "bayes",
    "name": "Training pinn with different hidden size",
    "metric": {"goal": "minimize", "name": "nmse"},
    "parameters": {
        #"batch_size": {"values": [8,16,128]},
        ##"learning_rate": {"values":[0.01,0.001,0.0001]},
        #"hidden_size":{"values":[512,256,128]},
        #"layer":{"max":6,"min":3},
        "pde_weights":{"values":[1e-6,5e-6,1e-5]},
        #"data_weights":{"max":2.0,"min":0.0001},
        #"bc_weights":{"max":5e-5,"min":1e-25},
        #"c":{"max":10,"min":1},
        #"w0":{"max":10,"min":1},
        #"w0_initial":{"max":15,"min":1},
        #"weight_decay":{"values":[1e-3,1e-4]}
    },
}

# Initialize sweep by passing in config.
sweep_id = wandb.sweep(sweep=sweep_configuration, project=f"UPSAMPLING-nopinn-{gb.points_sampled}-pinn_ef")


#%%

def train():
    epochs = 10000
    with wandb.init():

        #HYPERPARAMETERS
        hidden_size = 512
        layers = 4
        batch_size = 128
        learning_rate = 0.001
        data_weights = 1
        bc_weights=0
        c=5
        w0=1
        w0_initial = 5
        M = 3
        weight_decay = 1e-3

        #CREATE DATASET
        path_data = "../dataset/dataset_daga/Pos1_DRIR_LS_0.sofa"
        data_handler = dt.DataHandler(path_data,M)
        data_handler.remove_points(2)
        points_sampled =len(data_handler.INPUT_SAMPLED)
        gb.points_sampled = points_sampled
        train_dataset,val_dataset = data_handler.data_loader(batch_size)


        #CREATE_NETWORK
        model = fnn.PINN(gb.input_dim,gb.output_dim,hidden_size,layers,w0,w0_initial,c)
        model = model.to(gb.device)

        #CREATE OPTIMIZER

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4000, factor=0.1, verbose=True,min_lr=1e-5,cooldown=1500)
        inputs_not_sampled= data_handler.X_data

        
        #TRAINING

        pinn=True
        loss_comb = loss_functions.CombinedLoss(pinn)


        print(pinn)
        if pinn:
            print("Start Training PINN")
            wandb.run.name = f"hidden_size_{hidden_size}_{points_sampled}_pinn"
        else:
            print("Start Training NO PINN")
            wandb.run.name = f"hidden_size_{hidden_size}_{points_sampled}_no_pinn"

        # Set up early stopping parameters.
        best_val_loss = float('inf')
        patience = 200
        counter = 0
        
        for epoch in range(epochs):
            loss,loss_data,loss_pde,loss_bc = model.train_epoch(train_dataset,inputs_not_sampled,optimizer,loss_comb,
                                                                points_sampled,pinn=pinn)
            val_loss = model.test_epoch(val_dataset)
            #scheduler.step(val_loss)

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
            mean_nmse_time, nmse_time,_ = lf.NMSE(drir_ref, drir_prev)

            wandb.log({
                "epoch":epoch,
                "loss":loss,
                "loss_data":loss_data,
                "loss_pde":torch.mean(loss_pde),
                "loss_bc":loss_bc,
                "val_loss":val_loss,
                "e_f":gb.e_f,
                "e_d":gb.e_d,
                "e_b":gb.e_b,
                "nmse":mean_nmse_time,
                "learning_rate":optimizer.param_groups[0]['lr']
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
        print('FINISHED')
    # Path to save model
    path_saving = f"../src/models/models_{len(inputs_not_sampled)}_{gb.output_dim}_{points_sampled}.pth"
    torch.save(model.state_dict(), path_saving)

if __name__=="__main__":
    #TRAINING 
    wandb.agent(sweep_id= sweep_id, project=f"UPSAMPLING-nopinn-{gb.points_sampled}-pinn_ef",function=train)

# %%
