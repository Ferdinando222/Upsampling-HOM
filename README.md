# Upsampling-HOM: Physics-Informed Neural Networks for Spatial Upsampling of Spherical Microphone Arrays

This repository contains the official PyTorch implementation and resources for the research paper:  
**"A Physics-Informed Neural Network-Based Approach for the Spatial Upsampling of Spherical Microphone Arrays"**  
You can read the full paper [here](https://ieeexplore.ieee.org/document/10694489).

## Overview

Spatial upsampling for spherical microphone arrays is the task of reconstructing a high-resolution sound field from a limited number of microphone measurements. This is a crucial problem in audio signal processing, as it enables enhanced spatial sound reproduction and analysis, benefiting applications such as immersive audio, virtual reality, and sound source localization. This project proposes a novel approach using Physics-Informed Neural Networks (PINNs) to address this challenge by leveraging the underlying wave equation governing sound propagation.

## Project Structure

The repository is organized as follows:

-   `src/`: Contains all the Python source code for the project.
    -   `fnn.py`, `siren.py`: Implementations of the neural network models.
    -   `train.py`: Script for training the models.
    -   `test.py`: Script for evaluating the trained models.
    -   `data.py`: Handles data loading and preprocessing.
    -   `loss_functions.py`: Defines the loss functions used for training.
    -   `utils.py`, `global_variables.py`: Contain utility functions and global variables used across the project.
    -   `models/`: This directory is used to save the trained model checkpoints (as `.pth` files).
-   `dataset/`: Holds the datasets required for training and evaluating the models.
    -   `dataset_daga/`, `dataset_sarita/`: Subdirectories containing `.sofa` files, which are used to store spatially oriented acoustic data.
-   `results/`: Stores output files generated during or after experiments. This can include text files or CSVs with metrics like loss values and Normalized Mean Square Error (NMSE) for different configurations (e.g., `loss_siren+pde_16.csv`, `mean_nmse_time.txt`).

## Setup and Installation

To set up the project environment, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AcousticOdyssey/Upsampling-HOM.git
    cd Upsampling-HOM
    ```

2.  **Create a Python Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    The project uses a `requirements.txt` file to manage dependencies.
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The project utilizes two primary datasets for training, evaluation, and benchmarking:

-   **DAGA Dataset:** This dataset is used as the primary source for training and evaluating the spatial upsampling models. The `.sofa` files for the DAGA dataset are located in the `dataset/dataset_daga/` directory.
-   **SARITA Benchmark:** The SARITA dataset is used as a benchmark for comparison purposes, as indicated by its use in the testing scripts (e.g., `src/test.py`).

**SOFA File Format:**
The datasets are provided in the `.sofa` (Spatially Oriented Format for Acoustics) file format. SOFA is a standardized format designed to store acoustic data such as Head-Related Transfer Functions (HRTFs), Binaural Room Impulse Responses (BRIRs), and other spatially oriented acoustic measurements. This format is crucial for ensuring consistency and compatibility in acoustic research and applications.

**Data Availability:**
The `dataset/` directory within this repository includes sample `.sofa` files from the DAGA dataset. These samples are sufficient for running the provided training and testing scripts and demonstrating the functionality of the models. Currently, external download of larger dataset portions is not required to use this repository.

## Running the Code

This section describes how to run the training and testing scripts.

### Training (`src/train.py`)

The main script for training the Physics-Informed Neural Network (PINN) models is `src/train.py`.

-   **Experiment Tracking:** The project uses `wandb` (Weights & Biases) for experiment tracking and visualization. You may need to create a free `wandb` account at [https://wandb.ai](https://wandb.ai) and log in using `wandb login` in your terminal before running the training script for the first time.
-   **Hyperparameter Optimization:** The `sweep_configuration` dictionary within `src/train.py` is set up for hyperparameter optimization using `wandb` sweeps. You can launch sweeps to explore different hyperparameter combinations automatically.
-   **Manual Hyperparameter Configuration:** For single training runs, hyperparameters such as hidden layer size, number of layers, learning rate, and the weights for different parts of the loss function (e.g., PDE loss) can be modified directly within the `train()` function in `src/train.py`.
-   **Model Saving:** Trained models are saved in the `src/models/` directory. The filenames typically encode the configuration, for example, `models_32_True_513_9.pth` might indicate a model trained with 32 hidden units, PDE loss enabled, a certain number of FFT points, and for 9 input microphone points.

To start training, navigate to the `src` directory and run:
```bash
cd src
python train.py
```
(Adjust parameters within the script or use `wandb` sweeps for different configurations).

### Testing/Evaluation (`src/test.py`)

The `src/test.py` script is used to evaluate the performance of trained models.

-   **Interactive Model Selection:** When you run the script, it provides an interactive command-line interface that will prompt you to:
    1.  Select the number of points for which the model was trained (e.g., 4, 9, 16, 25). This refers to the number of microphone points in the low-resolution input.
    2.  Choose the model type to test (e.g., `Siren`, `Siren+Pde`, `Siren+Pde+Rowdy`).
-   **Model Loading:** Based on your selections, the script loads the corresponding pre-trained model from the `src/models/` directory.
-   **Outputs and Metrics:** The script provides comprehensive evaluation results:
    -   **Console Output:** Normalized Mean Square Error (NMSE) values for both time and frequency domains are printed to the console.
    -   **Plots:** Several plots are generated and displayed to visualize the model's performance:
        -   NMSE in the time domain for each microphone channel.
        -   Time-domain signal comparison: PINN prediction vs. Ground Truth, and SARITA benchmark vs. Ground Truth.
        -   NMSE in the frequency domain: PINN vs. SARITA benchmark.
        -   Magnitude of the signal for each channel at a fixed frequency.
        -   Average NMSE for each channel visualized on a spherical plot, showing spatial performance.
        -   NMSE for Spherical Harmonics (SH) coefficients, indicating accuracy in the spherical harmonics domain.
    -   **Saved Results:** Key results, such as mean NMSE values and frequency-domain NMSE, are also saved to files in the `results/` directory (e.g., `results/mean_nmse_time.txt`, `results/nmse_freq_{models}_{points}.txt`).

To run the evaluation, navigate to the `src` directory and execute:
```bash
cd src
python test.py
```
Follow the interactive prompts to select the model and configuration you wish to evaluate.

## Results

This section outlines where to find the outputs of the training and evaluation processes and how to interpret key performance metrics.

**Location of Results:**

-   **Numerical Results:** Quantitative results from training and evaluation are saved in the `results/` directory. These include:
    -   Loss histories from training, typically saved as CSV files (e.g., `loss_siren+pde_16.csv`).
    -   Normalized Mean Square Error (NMSE) values from testing, such as mean NMSE over time (`mean_nmse_time.txt`) and frequency-domain NMSE for specific configurations (e.g., `nmse_freq_{models}_{points}.txt`).
-   **Visual Results (Plots):** The `src/test.py` script generates several plots during its execution to provide a visual assessment of the model's performance. As described in the "Testing/Evaluation" section, these plots include time-domain signal comparisons, frequency-domain NMSE, spatial distribution of errors, and SH coefficient accuracy. Currently, these plots are primarily displayed on-screen during the script's execution.

**Interpretation of Key Metrics:**

The primary metric used to evaluate the performance of the upsampling models is the **Normalized Mean Square Error (NMSE)**.

-   **What it measures:** NMSE quantifies the difference between the sound field predicted by the model and the ground truth (actual) sound field. It is a normalized measure, which helps in comparing performance across different scales or conditions.
-   **Desired Value:** A **lower NMSE value indicates better performance**. This means that the model's predictions are closer to the actual sound field measurements.
-   **Reporting in Decibels (dB):** NMSE is often reported in decibels (dB). When expressed in dB, a **more negative value signifies better performance**. For example, an NMSE of -20 dB is better than -10 dB.

## License

This project is licensed under the terms of the `LICENSE` file.

## Citation

If you use the code from this repository or find our work relevant to your research, please consider citing the associated paper:

```bibtex
@INPROCEEDINGS{10694489,
  author={Miotello, Federico and Terminiello, Ferdinando and Pezzoli, Mirco and Bernardini, Alberto and Antonacci, Fabio and Sarti, Augusto},
  booktitle={2024 18th International Workshop on Acoustic Signal Enhancement (IWAENC)}, 
  title={A Physics-Informed Neural Network-Based Approach for the Spatial Upsampling of Spherical Microphone Arrays}, 
  year={2024},
  volume={},
  number={},
  pages={215-219},
  keywords={Array signal processing;Conferences;Neural networks;Parallel processing;Linear programming;Microphone arrays;Machine listening;Spatial resolution;Testing;Convergence;physics-informed neural network;spherical microphone array;space-time audio signal processing},
  doi={10.1109/IWAENC61483.2024.10694489}}
  pages={},
  doi={10.1109/document/10694489}
}
```


