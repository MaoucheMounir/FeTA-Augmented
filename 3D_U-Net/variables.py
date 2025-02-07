import sys
import torch
from torchvision.transforms.v2 import GaussianNoise, RandomInvert, ColorJitter
from torchio.transforms import RandomBiasField, RandomGamma, RandomNoise, RandomAnisotropy

########################################################################################################################
#                                                 VARIABLES                                                            #
########################################################################################################################

config = {
        # Paths
        'data_path': '/home/imag2/Desktop/BabySynthSeg-main/Data/FeTA_2022_test/mri',   # Path to the MRI data
        'labels_path': '/home/imag2/Desktop/BabySynthSeg-main/Data/FeTA_2022_test/segmentation',   # Path to the segmentations
        'weights_path': '/home/imag2/Desktop/Segmentation-Mounir/BabySynthSeg copie/weights_mounir',   # Path to save the model weights

        # Model
        'input_ch': 1,
        'output_ch': 2,   # ventricles + background
        # 'output_ch': 8,  # 7 structures + background
        'nb_dataloaders': 1,

        # Parameters for the patches
        'patch_size': (96,96,96), #(128,128,128)
        'stride': 32,

        # Hyper-parameters
        'lr': 0.01,
        'test_split': 0.2,
        'val_split': 0.1,
        'smooth': 1e-5,
        'ee': sys.float_info.epsilon,

        # Training settings
        'num_epochs': 100, #500,
        'batch_size': 2,
        'device': "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        "note": ""
}

transformation_dict = {
        "gaussian_noise": GaussianNoise(),
        "random_invert": RandomInvert(),
        "color_jitter": ColorJitter(0.2, 0.2, 0.2, 0.2),
        "random_bias_field": RandomBiasField(coefficients=1),
        "random_gamma": RandomGamma(),
        "random_noise": RandomNoise(),
        "random_anisotropy": RandomAnisotropy()
    }