#%%
import data as dt
import utils
import fnn
import torch
import numpy as np
import torch.optim as optim
import wandb

# Hyperparameters
frequency = 1000
input_dim = 3 
output_dim = 1 
points_sampled =1202

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define sweep config
sweep_configuration = {
    "method": "bayes",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "nmse"},
    "parameters": {
        "batch_size": {"values": [16, 32, 64]},
        "learning_rate": {"values":[0.001]},
        "hidden_size":{"values":[12,22,30]},
        "pde_weights":{"max":100.0,"min":0.1},
        "data_weights":{"max":2.0,"min":0.001}
    },
}

# Initialize sweep by passing in config.
# (Optional) Provide a name of the project.
sweep_id = wandb.sweep(sweep=sweep_configuration, project="UPSAMPLING-HOM-10")


def train():
    epochs = 5000
    with wandb.init():
    
        points_sampled =1202
        #CREATE DATASET
        path_data = "../dataset/DRIR_CR1_VSA_1202RS_R.sofa"
        data_handler = dt.DataHandler(path_data,frequency)
        points_sampled = 10
        data_handler.remove_points(points_sampled)
        print(data_handler.X_data.size())
        assert data_handler.X_data.size(0) <= points_sampled+1
        train_dataset = data_handler.data_loader(wandb.config.batch_size)

        #CREATE_NETWORK
        model = fnn.PINN(input_dim,output_dim,wandb.config.hidden_size)
        model = model.to(device)

        #CREATE OPTIMIZER
        optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
        _,inputs_not_sampled,_ = data_handler.create_tensors()
        
        #TRAINING
        print("Start Training PINN")
        pinn=True
        if pinn:
            wandb.run.name = f"hidden_size_{wandb.config.hidden_size}_{points_sampled}_pinn"
        else:
            wandb.run.name = f"hidden_size_{wandb.config.hidden_size}_{points_sampled}_no_pinn"

        for epoch in range(epochs):
            loss,loss_data,loss_pde = model.train_epoch(train_dataset,inputs_not_sampled,optimizer,wandb.config.data_weights,wandb.config.pde_weights,frequency,points_sampled,pinn=pinn)
            wandb.log({
                "epoch":epoch,
                "loss":loss,
                "loss_data":loss_data,
                "loss_pde":loss_pde
            }
            )
            if epoch % 50 == 0:
                print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item()}')

        nmse = model.test(data_handler)
        wandb.log({"nmse": nmse})

    print(nmse)
    # # plot and NMSE of the model
    # _,input_data,_ = data_handler.create_tensors()
    # input_data = input_data.to(device)
    # azimuth = data_handler.azimuth
    # colatitude = data_handler.colatitude
    # normalized_output_data = data_handler.NORMALIZED_OUTPUT_DATA
    # previsions_pinn = model.make_previsions(input_data)
    # previsions_pinn = previsions_pinn.cpu().detach().numpy()
    # utils.plot_model(azimuth,colatitude,normalized_output_data,previsions_pinn,points_sampled,True)
    print('FINISHED')

if __name__=="__main__":
    #TRAINING PINN
    wandb.agent(sweep_id=sweep_id,function=train)
    #TRAINING NO PINN

# %%
