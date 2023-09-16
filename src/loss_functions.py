import torch
import torch.nn as nn
import numpy as np
import math

class DataTermLoss(nn.Module):
    def __init__(self):
        super(DataTermLoss, self).__init__()

    def forward(self, predictions, target):
        mse = torch.mean(torch.abs(predictions - target) ** 2)
        return mse


class CombinedLoss(nn.Module):
    def __init__(self, data_weight=1.0, pde_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.data_weight = data_weight
        self.pde_weight = pde_weight
        self.data_loss_fn = DataTermLoss()


    def forward(self, predictions, target):
        # Calculate individual loss terms
        data_loss = self.data_loss_fn(predictions, target)

        return data_loss
    
def NMSE(normalized_output,previsions):
    normalized_output = normalized_output.flatten()
    mse = np.sum(np.abs(normalized_output-previsions)**2)
    normalization= np.sum(np.abs(normalized_output)**2)

    nmse = mse/normalization
    nmse_db = 10 * math.log10(nmse)
    return nmse_db


