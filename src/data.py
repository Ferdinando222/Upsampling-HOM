from sound_field_analysis import io, utils, gen
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import global_variables as gb

class DataHandler:
    """
    DataHandler class for processing and managing acoustic data.

    This class is designed to handle acoustic data from a SOFA file, preprocess it,
    and create tensors for training machine learning models, such as Physics-Informed Neural Networks (PINNs).

    Args:
        sofa_file_path (str): Path to the SOFA file containing acoustic data.
        frequency (float): The frequency of the acoustic data.

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

    def __init__(self, sofa_file_path, frequency):
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
        #TENSORS
        self.X_sampled = None
        self.X_not_sampled = None
        self.Y_sampled = None
        self.Y_not_sampled = None
        self.X_data = None
        self.Y_data = None

        # Extract data from the SOFA file
        self.extract_data(frequency, sofa_file_path)

        # Initialize sampled data to be the same as the full data
        self.INPUT_SAMPLED = self.INPUT_DATA
        self.OUTPUT_SAMPLED = self.OUTPUT_DATA
        self.INPUT_NOT_SAMPLED = self.INPUT_DATA
        self.OUTPUT_NOT_SAMPLED = self.OUTPUT_DATA

        # Normalize the data
        self.normalize_data()

        # Create tensors for training
        self.create_tensors()

    def extract_data(self, frequency, sofa_file_path):
        """
        Extract acoustic data from a SOFA file.

        Args:
            frequency (float): The frequency of the acoustic data.
            sofa_file_path (str): Path to the SOFA file containing acoustic data.
        """
        # Read the SOFA file
        DRIR = io.read_SOFA_file(sofa_file_path)
        grid = DRIR.grid

        # Calculate the index based on the given frequency
        index = int(np.ceil((frequency / DRIR.signal.fs) * len(DRIR.signal[0])))

        # Extract the output data (FFT at the specified index)
        self.OUTPUT_DATA = np.fft.fft(DRIR.signal.signal)[:, index]

        # Extract spherical coordinates
        self.azimuth = grid.azimuth
        self.colatitude = grid.colatitude
        self.radius = grid.radius

        # Convert spherical coordinates to Cartesian coordinates
        self.x, self.y, self.z = utils.sph2cart((self.azimuth, self.colatitude, self.radius))

        self.INPUT_DATA = list(zip(self.x, self.y, self.z))

        # Check on INPUT_DATA AND OUTPUT_DATA
        assert len(self.INPUT_DATA) == len(DRIR.signal.signal)
        assert len(self.INPUT_DATA) == len(self.OUTPUT_DATA)


    def normalize_data(self):
        """
        Normalize the output data.

        The real and imaginary parts of the output data are normalized separately.
        """
        def normalize(part,scaler):
            scale = scaler.fit_transform(part.reshape(-1, 1))
            return scale

        real_part_sampled, real_part_data,real_part = np.real(self.OUTPUT_SAMPLED), np.real(self.OUTPUT_NOT_SAMPLED),np.real(self.OUTPUT_DATA)
        imaginary_part_sampled, imaginary_part_data,imaginary_part = np.imag(self.OUTPUT_SAMPLED), np.imag(self.OUTPUT_NOT_SAMPLED),np.imag(self.OUTPUT_DATA)

        normalized_real_sampled = normalize(real_part_sampled,gb.scaler_r_s)
        normalized_real_not_sampled = normalize(real_part_data,gb.scaler_r_ns)
        normalized_real_data = normalize(real_part,gb.scaler_r_d)

        normalized_imaginary_sampled = normalize(imaginary_part_sampled,gb.scaler_i_s)
        normalized_imaginary_not_sampled= normalize(imaginary_part_data,gb.scaler_i_ns)
        normalized_imaginary_data = normalize(imaginary_part,gb.scaler_i_d)

        self.NORMALIZED_OUTPUT_SAMPLED = normalized_real_sampled + 1j * normalized_imaginary_sampled
        self.NORMALIZED_OUTPUT_NOT_SAMPLED = normalized_real_not_sampled + 1j * normalized_imaginary_not_sampled
        self.NORMALIZED_OUTPUT = normalized_real_data + 1j * normalized_imaginary_data

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
        mic = np.column_stack((self.azimuth, self.colatitude))

        grid_under = gen.lebedev(order)
        az_und = grid_under.azimuth
        col_und = grid_under.colatitude
        mic_under = np.column_stack((az_und, col_und))

        matching_indices = []

        # Scansiona gli elementi di mic_under
        for i,item in enumerate(mic):
            item = np.round(item,6)
            for position in mic_under:
                position = np.round(position,6)
                
                if np.array_equal(position,item):
                    matching_indices.append(i)
                    break

        self.INPUT_SAMPLED = [item for i, item in enumerate(self.INPUT_DATA) if i  in matching_indices]
        self.OUTPUT_SAMPLED =[item for i, item in enumerate(self.OUTPUT_DATA) if i  in matching_indices]

        self.INPUT_NOT_SAMPLED  = np.delete(self.INPUT_DATA,matching_indices,axis=0 )
        self.OUTPUT_NOT_SAMPLED = np.delete(self.OUTPUT_DATA, matching_indices, axis=0)

        # Check input and ouput have the same size of points_sampled
        assert len(self.INPUT_SAMPLED) == len(self.OUTPUT_SAMPLED) == 14, "Input and output sizes do not match points_sampled"
        assert len(self.INPUT_NOT_SAMPLED) == len(self.OUTPUT_NOT_SAMPLED) == (len(self.INPUT_DATA)-14)

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
        dataset_sampled = list(zip(self.X_sampled, self.Y_sampled))
        dataset_not_sampled = list(zip(self.X_not_sampled,self.Y_not_sampled))
        train_dataloader = DataLoader(dataset_sampled, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(dataset_not_sampled,batch_size=batch_size,shuffle=True)

        return train_dataloader,test_dataloader
