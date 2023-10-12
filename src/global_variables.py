import torch

#GLOBAL VARIABLES
frequency = 12000
input_dim = 4 
output_dim = 1 
points_sampled =14
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")