import torch

#GLOBAL VARIABLES
frequency = []
input_dim = 3 
output_dim = 513
e_f,e_d,e_b = 1e-6,1,0.01
points_sampled =38
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")