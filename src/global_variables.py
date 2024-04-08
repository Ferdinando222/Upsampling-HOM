import torch
from sound_field_analysis import io

#GLOBAL VARIABLES
frequency = []
input_dim = 3 
output_dim = 513
e_f,e_d,e_b = 1e-7,1,0.01
spherical_grid = None
spherical_grid_up = None
points_sampled = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")