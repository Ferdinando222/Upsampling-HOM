from sound_field_analysis import io, utils, gen,process
import sound_field_analysis as sfa
import numpy as np
import torch
from torch.utils.data import DataLoader
import global_variables as gb
import math
from scipy import signal
from torch.utils.data import TensorDataset

import matplotlib.pyplot as plt


process.spatFT

class DataHandler:
    """
    DataHandler class for processing and managing acoustic data.

    This class is designed to handle acoustic data from a SOFA file, preprocess it,
    and create tensors for training neural network models.

    Args:
        sofa_file_path (str): Path to the SOFA file containing acoustic data.
        frequency (float): The frequency of the acoustic data.
        M (flaot): Factor for undersampling

    Attributes:
        INPUT_DATA (list of tuples): Input data extracted from the SOFA file.
        OUTPUT_DATA (numpy.ndarray): Output data extracted from the SOFA file.
        NORMALIZED_OUTPUT_DATA (numpy.ndarray): Normalized output data.
        NORMALIZED_OUTPUT_SAMPLED (numpy.ndarray): Normalized output data for sampled points.
        x (numpy.ndarray): X-coordinates of the data points.
        y (numpy.ndarray): Y-coordinates of the data points.
        z (numpy.ndarray): Z-coordinates of the data points.
        azimuth (numpy.ndarray): Azimuth angles of the data points.
        colatitude (numpy.ndarray): Colatitude angles of the data points.
        radius (numpy.ndarray): Radii of the data points.
        X_data (torch.Tensor): Tensor containing the input data for sampled points.
        X_data_not_sampled (torch.Tensor): Tensor containing the input data for all points.
        Y_data_not_sampled (torch.Tensor): Tensor containing the output data for all points.
        Y_data (torch.Tensor): Tensor containing the normalized output data for sampled points.
    """

    def __init__(self, sofa_file_path,M):

        #SIGNAL IN TIME
        self.drir = None
        # INPUT AND OUTPUT DATA
        self.INPUT_DATA = None
        self.OUTPUT_DATA = None

        #INPUT AND OUTPUT NOT SAMPLED
        self.INPUT_NOT_SAMPLED = None
        self.OUTPUT_NOT_SAMPLED = None

        # NORMALIZE OUTPUT DATA
        self.NORMALIZED_OUTPUT_NOT_SAMPLED = None
        self.NORMALIZED_OUTPUT_SAMPLED = None
        self.NORMALIZED_OUTPUT = None

        # EXTRACT POINTS IN SPHERICAL AND CARTESIAN COORDINATES
        self.x = None
        self.y = None
        self.z = None
        self.azimuth = None
        self.colatitude = None
        self.radius = None
        self.frequencies = None
        #TENSORS
        self.X_sampled = None
        self.X_not_sampled = None
        self.Y_sampled = None
        self.Y_not_sampled = None
        self.X_data = None
        self.Y_data = None

        self.M = M

        # Extract data from the SOFA file
        self.extract_data(sofa_file_path,self.M)

        # Initialize sampled data to be the same as the full data
        self.INPUT_SAMPLED = self.INPUT_DATA
        self.OUTPUT_SAMPLED = self.OUTPUT_DATA
        self.INPUT_NOT_SAMPLED = self.INPUT_DATA
        self.OUTPUT_NOT_SAMPLED = self.OUTPUT_DATA

        # Normalize the data
        self.normalize_data()

        # Create tensors for training
        self.create_tensors()

    def extract_data(self, sofa_file_path,M):
        """
        Extract acoustic data from a SOFA file.

        Args:
            sofa_file_path (str): Path to the SOFA file containing acoustic data.
            M (float): Down sampling factor
        """
        # Read the SOFA file
        DRIR = io.read_SOFA_file(sofa_file_path)
        drir = DRIR.signal.signal

        # Control if the drir signal has the following shape [signal,mic]
        if(len(drir) != len(DRIR.grid.azimuth)):
            drir = drir.T

        grid = DRIR.grid
        fs = int(DRIR.signal.fs/M)
        NFFT = len(drir[0,:])
        self.NFFT_down = int(np.round(NFFT/M))
        print(self.NFFT_down)

        #DOWNSAMPLING
        print("Undersampling...")

        drir_down = np.zeros((len(drir), self.NFFT_down))

        for i in range(len(drir)):
            drir_down[i, :] = signal.resample_poly(drir[i,:],1,self.M)

        self.NFFT_down = 1025
        self.drir = np.array(drir_down)


        print("Shape signal:",self.drir.shape)
        self.drir = self.drir[:,:self.NFFT_down]
        print("Shape signal undersampled",self.drir.shape)
        # Extract the output data (FFT at the specified index)

        print("Computing FFT...")
        self.OUTPUT_DATA = np.fft.fft(self.drir,self.NFFT_down)

        n = len(self.OUTPUT_DATA[0,:])

        print("Shape FFT:",len(self.OUTPUT_DATA),len(self.OUTPUT_DATA[0,:]))
        self.frequencies = np.fft.fftfreq(n, 1.0 / fs)
        self.frequencies= self.frequencies[:int(np.ceil(n/2))]
        gb.frequency = self.frequencies

        # Calculate the index based on the given frequency

        self.OUTPUT_DATA = self.OUTPUT_DATA[:,:int(np.ceil(n/2))]

        print("Shape FFT:",len(self.OUTPUT_DATA),len(self.OUTPUT_DATA[0,:]))

        # Extract spherical coordinates
        # DELET COMMENT FOR DATASET DAGA
        self.colatitude = np.mod(grid.azimuth,2*np.pi)
        self.azimuth = np.mod(grid.colatitude,2*np.pi)
        #self.colatitude =grid.colatitude
        #self.azimuth = grid.azimuth
        self.radius = grid.radius

        # Convert spherical coordinates to Cartesian coordinates
        self.x, self.y, self.z = utils.sph2cart((self.azimuth, self.colatitude, self.radius))

        self.INPUT_DATA = list(zip(self.x, self.y, self.z))


        # Check on INPUT_DATA AND OUTPUT_DATA
        assert len(self.INPUT_DATA) == len(drir)
        assert len(self.INPUT_DATA) == len(self.OUTPUT_DATA)


    def normalize_data(self):
        """
        Normalize the output data.

        The real and imaginary parts of the output data are normalized separately.
        """


        max_out_s_p = np.max(np.abs(self.OUTPUT_SAMPLED))
        max_out_ns_p = np.max(np.abs(self.OUTPUT_NOT_SAMPLED))
        max_out = np.max(np.abs(self.OUTPUT_DATA))

        # self.NORMALIZED_OUTPUT_SAMPLED = np.abs(self.OUTPUT_SAMPLED)/(max_out_s_p) * np.exp(1j * np.angle(self.OUTPUT_SAMPLED))
        # self.NORMALIZED_OUTPUT_NOT_SAMPLED = np.abs(self.OUTPUT_NOT_SAMPLED)/(max_out_ns_p) * np.exp(1j * np.angle(self.OUTPUT_NOT_SAMPLED))
        # self.NORMALIZED_OUTPUT =np.abs(self.OUTPUT_DATA)/(max_out) * np.exp(1j * np.angle(self.OUTPUT_DATA))

        
        self.NORMALIZED_OUTPUT_SAMPLED = self.OUTPUT_SAMPLED
        self.NORMALIZED_OUTPUT_NOT_SAMPLED = self.OUTPUT_NOT_SAMPLED
        self.NORMALIZED_OUTPUT  = self.OUTPUT_DATA



    def create_tensors(self):
        """
        Create PyTorch tensors from the data.

        This method creates tensors for input and output data.
        """
        self.X_sampled = torch.tensor(self.INPUT_SAMPLED, dtype=torch.float32)
        self.X_not_sampled = torch.tensor(self.INPUT_NOT_SAMPLED, dtype=torch.float32)
        self.X_data = torch.tensor(self.INPUT_DATA, dtype=torch.float32)
        self.Y_sampled = torch.tensor(self.NORMALIZED_OUTPUT_SAMPLED, dtype=torch.cfloat)
        self.Y_not_sampled = torch.tensor(self.NORMALIZED_OUTPUT_NOT_SAMPLED,dtype=torch.cfloat)
        self.Y_data = torch.tensor(self.NORMALIZED_OUTPUT, dtype=torch.cfloat)

        return self.X_sampled, self.X_not_sampled, self.Y_sampled,self.Y_not_sampled

    def remove_points(self, order):
        """
        Remove points from the sampled data.

        Args:
            points_sampled (int): Number of sampled points to keep.
        """

        def distance(point1, point2, radius=1.0):
            # Convert spherical coordinates to radians
            phi1, theta1 = point1
            phi2, theta2 = point2

            # Compute spherical distance
            d = math.acos(np.sin(phi1) * np.sin(phi2) + np.cos(phi1) * np.cos(phi2) * np.cos(theta1 - theta2)) * radius

            return d

        mic = np.column_stack((self.azimuth, self.colatitude))

        self.grid_under = gen.lebedev(order)
        az_und = self.grid_under.azimuth
        col_und = self.grid_under.colatitude
        mic_under = np.column_stack((az_und, col_und))

        matching_indices = []
        
    
        for item in mic_under:
            min_distance = np.infty
            for i,item1 in enumerate(mic):
                dist = distance(item,item1)
                if dist < min_distance:
                    min_distance = dist
                    number = i

            matching_indices.append(number)
        
        matching_indices = np.array(matching_indices)
        print(matching_indices)
        azimuth = self.azimuth[matching_indices]
        col = self.colatitude[matching_indices]

        # Conversione in radianti per il plot
        azimuth_rad = self.azimuth
        colatitude_rad = self.colatitude
        azimuth_un = azimuth
        colatitude_un = col

        # Plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111,projection='polar')

        # Plot dei punti
        print(azimuth_rad,colatitude_rad)
        ax.scatter(azimuth_rad, colatitude_rad, marker='o', color='blue')
        ax.scatter(azimuth_un, colatitude_un, marker='o', color='red')
        ax.scatter(az_und,col_und,marker='x',color='green')

        ax.set_theta_zero_location('N')  # Imposta zero a nord
        ax.set_theta_direction(-1)  # Orientazione antioraria
        # Personalizzazione
        ax.set_title('Plot delle posizioni in coordinate sferiche', va='bottom')
        ax.grid(True)

        plt.show()
        self.INPUT_SAMPLED = np.array(self.INPUT_SAMPLED)[matching_indices,:]
        self.OUTPUT_SAMPLED = np.array(self.drir)[matching_indices,:]
        self.drir_sampled = self.OUTPUT_SAMPLED
        self.OUTPUT_SAMPLED = np.fft.fft(self.OUTPUT_SAMPLED,self.NFFT_down)


        self.INPUT_NOT_SAMPLED  = np.delete(self.INPUT_DATA,matching_indices,axis=0 )
        self.OUTPUT_NOT_SAMPLED = np.delete(self.drir, matching_indices, axis=0)
        self.OUTPUT_NOT_SAMPLED= np.fft.fft(self.OUTPUT_NOT_SAMPLED,self.NFFT_down)
        self.OUTPUT_SAMPLED = self.OUTPUT_SAMPLED[:,:int(np.ceil(self.NFFT_down/2))]
        self.OUTPUT_NOT_SAMPLED = self.OUTPUT_NOT_SAMPLED[:,:int(np.ceil(self.NFFT_down/2))]

        # Check input and ouput have the same size of points_sampled
        print(len(mic_under))
        print(len(self.INPUT_DATA))
        assert len(self.INPUT_SAMPLED) == len(self.OUTPUT_SAMPLED) == len(mic_under) 
        print(len(self.INPUT_NOT_SAMPLED),len(self.OUTPUT_NOT_SAMPLED),len(self.INPUT_DATA)-len(mic_under) )
        assert len(self.INPUT_NOT_SAMPLED) == len(self.OUTPUT_NOT_SAMPLED) == (len(self.INPUT_DATA)-len(mic_under))

        # Normalize and create tensors
        self.normalize_data()
        self.create_tensors()

    def data_loader(self, batch_size=32):
        """
        Create a DataLoader for training data.

        Args:
            batch_size (int): Batch size for the DataLoader.

        Returns:
            train_dataloader (DataLoader): DataLoader for training data.
            test_dataloader (DataLoader): DataLoader for validation data.
        """

        train_dataset = TensorDataset(torch.tensor(self.X_sampled), torch.tensor(self.Y_sampled))
        val_dataset = TensorDataset(torch.tensor(self.X_not_sampled), torch.tensor(self.Y_not_sampled))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

        return train_dataloader,test_dataloader
