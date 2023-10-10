#%%
from sound_field_analysis import io, utils, gen
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import global_variables as gb


path_data = "../dataset/DRIR_CR1_VSA_1202RS_R.sofa"
DRIR = io.read_SOFA_file(path_data)
grid = DRIR.grid
# %%
# Calculate the index based on the given frequency

# Extract the output data (FFT at the specified index)
OUTPUT_DATA = np.fft.fft(DRIR.signal.signal)
n = len(OUTPUT_DATA[0])
frequencies = np.fft.fftfreq(n, 1.0 / DRIR.signal.fs)
frequencies = frequencies[:n//2]
OUTPUT_DATA = OUTPUT_DATA[:n//2]


# Extract spherical coordinates
azimuth = grid.azimuth
colatitude = grid.colatitude
radius = grid.radius
# Convert spherical coordinates to Cartesian coordinates
x, y, z = utils.sph2cart((azimuth, colatitude,radius))
INPUT_DATA = np.array(zip(x, y, z))
# %%
