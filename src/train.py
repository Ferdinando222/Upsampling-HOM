#%%
import data as dt
import utils
import fnn
import torch.optim as optim
import wandb
import global_variables as gb


# Define sweep config
sweep_configuration = {
    "method": "bayes",
    "name": "Training pinn with different hidden size",
    "metric": {"goal": "minimize", "name": "nmse"},
    "parameters": {
        "batch_size": {"values": [16,32,64,128]},
        "learning_rate": {"values":[0.01,0.001,0.0001]},
        "hidden_size":{"values":[256,128,64,32,22,12]},
        "layer":{"values":[5,3,2,1]},
        "pde_weights":{"max":0.1,"min":0.00001},
        #"data_weights":{"max":2.0,"min":0.001}
        "c":{"values":[20,10,5,1]},
        "w0":{"values":[50,25,12,5,1]},
        "w0_initial":{"values":[50,25,12,5,1]}
    },
}

# Initialize sweep by passing in config.
sweep_id = wandb.sweep(sweep=sweep_configuration, project=f"UPSAMPLING-pinn-siren-{gb.points_sampled}-{gb.frequency}")

#TODO: 
# 1)TRAINING IN DIFFERENT AMBIENT
# 2) Try to implement loss with boundary condition
#%%

def train():
    epochs = 10000000
    with wandb.init():

        #HYPERPARAMETERS
        hidden_size = wandb.config.hidden_size
        layers =wandb.config.layer
        batch_size =wandb.config.batch_size
        learning_rate = wandb.config.learning_rate
        c=wandb.config.c
        w0=wandb.config.w0
        w0_initial = wandb.config.w0_initial
        data_weights = 1
        bc_weights=0
        pde_weights = wandb.config.pde_weights
        frequency =400

        #CREATE DATASET
        path_data = "../dataset/DRIR_CR1_VSA_1202RS_R.sofa"
        data_handler = dt.DataHandler(path_data,frequency)
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

        pinn=True
        if pinn:
            print("Start Training PINN")
            wandb.run.name = f"hidden_size_{hidden_size}_{points_sampled}_{gb.frequency}_pinn"
        else:
            print("Start Training NO PINN")
            wandb.run.name = f"hidden_size_{hidden_size}_{points_sampled}_{gb.frequency}_no_pinn"

        # Set up early stopping parameters.
        best_val_loss = float('inf')
        patience = 8000
        counter = 0
        
        for epoch in range(epochs):
            loss,loss_data,loss_pde,loss_bc = model.train_epoch(train_dataset,inputs_not_sampled,optimizer,data_weights,pde_weights,bc_weights,points_sampled,pinn=pinn)
            val_loss = model.test_epoch(val_dataset)

            wandb.log({
                "epoch":epoch,
                "loss":loss,
                "loss_data":loss_data,
                "loss_pde":loss_pde,
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
                if(counter %1000 == 0 and learning_rate> 0.00001):
                    learning_rate = learning_rate/10
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                    print(f"Decrease learning rate {learning_rate} epochs.")

            if counter >= patience:
                print(f"Early stopping after {epoch + 1} epochs.")
                counter=0
                break
    

        nmse = best_model.test(data_handler)
        wandb.log({"nmse": nmse})

    print(nmse)

    # plot the model
    input_data = data_handler.X_data
    input_data = input_data.to(gb.device)
    previsions_pinn = best_model.make_previsions(input_data)
    previsions_pinn = previsions_pinn.cpu().detach().numpy()
    previsions_pinn = previsions_pinn
    utils.plot_model(data_handler,previsions_pinn,points_sampled,pinn)
    print('FINISHED')

if __name__=="__main__":
    #TRAINING 
    wandb.agent(sweep_id=sweep_id,project=f"UPSAMPLING-pinn-siren-{gb.points_sampled}-{gb.frequency}",function=train)

# %%
