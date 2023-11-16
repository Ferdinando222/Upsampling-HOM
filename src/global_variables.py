import torch
from sklearn.preprocessing import MinMaxScaler

#GLOBAL VARIABLES
frequency = []
input_dim = 3 
scaler_r_s = MinMaxScaler()
scaler_r_ns= MinMaxScaler()
scaler_r_d = MinMaxScaler()

scaler_i_s = MinMaxScaler()
scaler_i_ns= MinMaxScaler()
scaler_i_d = MinMaxScaler()
output_dim = 8500
points_sampled =38
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")