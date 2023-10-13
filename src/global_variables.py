import torch

#GLOBAL VARIABLES
frequency = 1000
input_dim = 3 
output_dim = 1 
points_sampled =14
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")