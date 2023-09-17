from sound_field_analysis import io
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import random

INPUT_DATA = []
OUTPUT_DATA = []

def extract_data(frequency):
    global INPUT_DATA
    global OUTPUT_DATA
    DRIR = io.read_SOFA_file("../dataset/DRIR_CR1_VSA_1202RS_R.sofa")
    grid = DRIR.grid
    azimuth = grid.azimuth
    colatitude = grid.colatitude
    radius = grid.radius
    index = int(np.ceil((frequency/DRIR.signal.fs)*len(DRIR.signal[0])))
    global INPUT_DATA
    global OUTPUT_DATA
    INPUT_DATA = list(zip(azimuth, colatitude, radius))
    OUTPUT_DATA = np.fft.fft(DRIR.signal.signal)[:, index]
    return INPUT_DATA,OUTPUT_DATA,azimuth,colatitude

def sampling_points(points_sampled):
    #Sampling points
    global INPUT_DATA
    input_sampled = []
    output_sampled = []
    index_sampling = random.sample(range(len(INPUT_DATA)), points_sampled)
    for index in index_sampling:
        input_sampled.append(INPUT_DATA[index])
        output_sampled.append(OUTPUT_DATA[index])

    return input_sampled,output_sampled

def normalize_data(output_sampled):

    # Separate the real and imaginary parts
    real_part = np.real(OUTPUT_DATA)
    imaginary_part = np.imag(OUTPUT_DATA)
    real_part_sampled = np.real(output_sampled)
    imaginary_part_sampled = np.imag(output_sampled)

    real_scaler = MinMaxScaler()
    normalized_real_part = real_scaler.fit_transform(real_part.reshape(-1, 1))
    real_scaler_sampled = MinMaxScaler()
    normalized_real_part_sampled = real_scaler_sampled.fit_transform(real_part_sampled.reshape(-1, 1))

    imaginary_scaler = MinMaxScaler()
    normalized_imaginary_part = imaginary_scaler.fit_transform(imaginary_part.reshape(-1, 1))
    imaginary_scaler_sampled = MinMaxScaler()
    normalized_imaginary_part_sampled = imaginary_scaler_sampled.fit_transform(imaginary_part_sampled.reshape(-1, 1))

    # Combine the normalized real and imaginary parts
    normalized_output_data = normalized_real_part + 1j * normalized_imaginary_part
    normalized_output_sampled = normalized_real_part_sampled + 1j * normalized_imaginary_part_sampled

    return normalized_output_data,normalized_output_sampled
    
def create_tensors(input_sampled,normalized_output_sampled):
    # Prepare the input data as tensors
    X_data = torch.tensor(input_sampled, dtype=torch.float32)
    X_data_not_sampled = torch.tensor(INPUT_DATA,dtype=torch.float32)

    Y_data = torch.tensor(normalized_output_sampled,dtype=torch.complex32)

    return X_data,X_data_not_sampled,Y_data
