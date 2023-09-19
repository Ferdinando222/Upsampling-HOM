from sound_field_analysis import io,utils
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import random

class DataHandler:
    def __init__(self, sofa_file_path,frequency,points_sampled):
        #INPUT AND OUTPUT DATA
        self.INPUT_DATA = None
        self.OUTPUT_DATA = None
        #SAMPLING INPUT AND OUTPUT DATA
        self.OUTPUT_SAMPLED = None
        self.INPUT_SAMPLED = None
        #NORAMALIZE OUTPUT DATA
        self.NOMRALIZED_OUTPUT_DATA = None
        self.NORMALIZED_OUTPUT_SAMPLED = None
        #EXTRACT POINTS IN SPHERICAL AND CARTESIAN COORDINATES
        self.x = None 
        self.y = None 
        self.z = None
        self.azimuth = None
        self.colatitude = None
        self.radius = None

        self.extract_data(frequency,sofa_file_path)
        self.sampling_points(points_sampled)
        self.normalize_data()


    def extract_data(self, frequency,sofa_file_path):
        DRIR = io.read_SOFA_file(sofa_file_path)
        grid = DRIR.grid
        index = int(np.ceil((frequency / DRIR.signal.fs) * len(DRIR.signal[0])))
        self.OUTPUT_DATA = np.fft.fft(DRIR.signal.signal)[:, index]
        self.azimuth = grid.azimuth
        self.colatitude = grid.colatitude
        self.radius = grid.radius
        self.x, self.y, self.z = utils.sph2cart((self.azimuth, self.colatitude, self.radius))
        self.INPUT_DATA = list(zip(self.x, self.y, self.z))

    def sampling_points(self, points_sampled):
        index_sampling = random.sample(range(len(self.INPUT_DATA)), points_sampled)
        self.INPUT_SAMPLED= [self.INPUT_DATA[index] for index in index_sampling]
        self.OUTPUT_SAMPLED = [self.OUTPUT_DATA[index] for index in index_sampling]
        

    def normalize_data(self):
        def normalize(part):
            scaler = MinMaxScaler()
            return scaler.fit_transform(part.reshape(-1, 1))

        real_part_sampled, real_part_data = np.real(self.OUTPUT_SAMPLED), np.real(self.OUTPUT_DATA)
        imaginary_part_sampled, imaginary_part_data = np.imag(self.OUTPUT_SAMPLED), np.imag(self.OUTPUT_DATA)

        normalized_real_sampled = normalize(real_part_sampled)
        normalized_real_data = normalize(real_part_data)

        normalized_imaginary_sampled = normalize(imaginary_part_sampled)
        normalized_imaginary_data = normalize(imaginary_part_data)

        self.NORMALIZED_OUTPUT_SAMPLED = normalized_real_sampled + 1j * normalized_imaginary_sampled
        self.NORMALIZED_OUTPUT_DATA = normalized_real_data + 1j * normalized_imaginary_data


    def create_tensors(self):
        X_data = torch.tensor(self.INPUT_SAMPLED, dtype=torch.float32)
        X_data_not_sampled = torch.tensor(self.INPUT_DATA,dtype=torch.float32)
        Y_data = torch.tensor(self.NORMALIZED_OUTPUT_SAMPLED, dtype=torch.complex32)
        return X_data,X_data_not_sampled, Y_data

    @staticmethod
    def convert_cartesian(azimuth, colatitude, radius):
        x, y, z = utils.sph2cart((azimuth, colatitude, radius))
        return x, y, z