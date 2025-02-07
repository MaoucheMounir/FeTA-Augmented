import torch
import torch.utils.data as data
import torchio as tio
import numpy as np

from variables import transformation_dict

########################################################################################################################
#                                                 DATA FUNCTIONS & CLASSES                                                  #
########################################################################################################################


# class CustomDataset(data.Dataset):
#     def __init__(self, data, transform=None):
#         self.data = data
#         self.transform = transform

#     def __getitem__(self, index):
#         sample = self.data[index]
#         if self.transform:
#             sample = self.transform(sample)
#         return sample

#     def __len__(self):
#         return len(self.data)

def convert_to_binary_mask(segmentation, target_label=4): #M: la segmentation est supposee avoir plusieurs labels, donc on donne un label et on transforme la carte de segmentation en une image binaire contenant le masque de ce label-la en particulier
    """Converts a segmentation to a binary mask."""
    binary_mask = (segmentation == target_label).astype(np.float32)
    return binary_mask


def get_loader(input_data, batch_size):
    """Builds and returns Dataloader."""
    #dataset = CustomDataset(input_data, transform)
    data_loader = data.DataLoader(dataset=input_data,
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last=True)
    return data_loader

def create_data_augmentations(transformation_names:list[str]):
    """
    Creates torchIO compose object that applies data augmentation like torch.compose
    args: The list of the names of the data augmentations
    returns: The data augmentation composition object from torchIO
    """
    
    transformations = []
    for transf in transformation_names:
        if transf not in transformation_dict:
            raise ValueError(f"Nom de transformation invalide : '{transf}'")
        else:
            transformations.append(transformation_dict[transf])
    
    return tio.Compose(transformations)

def create_subjects_from_patches(list_patches:list[tuple]):
    """
    Converts patches and segmentationsinto tio.Subject objects 
    args: List of tuples (patch,segmentation)
    patch and segmentation are numpy arrays

    returns: list of torchIO Subjects. 
    each torchIO Subject will be an object that contains a pytorch tensor and a segmentation
    """
    subjects_list = []
    for patch, segmentation in list_patches:
        subject = tio.Subject(
            t1=tio.ScalarImage(tensor=torch.tensor(patch, dtype=torch.float).unsqueeze(0)),
            label=tio.LabelMap(tensor=torch.tensor(segmentation, dtype=torch.float).unsqueeze(0)),
        )
        subjects_list.append(subject)
    return subjects_list

def get_loader_tio(subjects_list, batch_size, transforms):
    """Builds and returns TorchIO SubjectsLoader while associating the transforms (the transformations are applied online when the model asks for a specific patch.).
    args: list of torchIO Subjects, batch size and the transforms compose object to apply data augmentation
    returns: subjects loader object similar to pytorch data loader
    """
    
    # SubjectsDataset is a subclass of torch.data.utils.Dataset
    subjects_dataset = tio.SubjectsDataset(subjects_list, transform=transforms)

    # Images are processed in parallel thanks to a SubjectsLoader
    # (which inherits from torch.utils.data.DataLoader)
    subjects_loader = tio.SubjectsLoader(
        subjects_dataset,
        batch_size=batch_size,
        num_workers=2,
        shuffle=True,
    )
    return subjects_loader


    ########################################################################################################################
    #                                                  PATCHS                                                              #
    ########################################################################################################################

def get_patch(image, patch_size, stride):
    """
    Extract patches from a 3D image.

    Args:
        image (np.array): 3D image.
        patch_size (tuple): Size of the patch to extract.
        stride (int): Stride to use when extracting the patches.

    Returns:
        list: List of patches.
    """

    patches = []
    patch_size = (patch_size[0], patch_size[1], patch_size[2])

    for i in range(0, image.shape[0] - patch_size[0] + 1, stride):
        for j in range(0, image.shape[1] - patch_size[1] + 1, stride):
            for k in range(0, image.shape[2] - patch_size[2] + 1, stride):
                patch = image[i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]]
                patches.append(patch)

    # print("Number of patches extracted:", len(patches))

    return patches


