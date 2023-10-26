import torch
from sklearn.preprocessing import MinMaxScaler

#GLOBAL VARIABLES
frequency = 23000
input_dim = 3
scaler_r_s = MinMaxScaler()
scaler_r_ns= MinMaxScaler()
scaler_r_d = MinMaxScaler()

scaler_i_s = MinMaxScaler()
scaler_i_ns= MinMaxScaler()
scaler_i_d = MinMaxScaler()
output_dim = 1 
points_sampled =14
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")