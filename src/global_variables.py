import torch
from sklearn.preprocessing import MinMaxScaler

#GLOBAL VARIABLES
frequency = []
input_dim = 3 
output_dim = 513
points_sampled =38
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")