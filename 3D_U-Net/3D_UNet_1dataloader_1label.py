########################################################################################################################
#                                                  IMPORTS                                                             #
########################################################################################################################

from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import os
import warnings

import torch
#import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torchio as tio

import nibabel as nib
import sys
from icecream import ic
import torch

from data import *
from modelClasses import *
from loggingFunctions import *
from variables import config
from evaluation import CombinedLoss, dice_score, evaluate_tio


########################################################################################################################
#                                                   DATA                                                               #
########################################################################################################################

if __name__ == '__main__':
    # Find all the files in the data and segmentation folders which end with .nii.gz
    data_files = [file for file in os.listdir(config['data_path']) if file.endswith('.nii.gz')]
    segmentation_files = [f for f in os.listdir(config['labels_path']) if f.endswith('.nii.gz')]

    # print("Number of MRI files:", len(data_files))
    # print("Number of segmentation files:", len(segmentation_files))

    # Load all the data and labels
    # Each element of the list is a tuple containing the data/label and the file name
    mri = [((nib.as_closest_canonical(nib.load(os.path.join(config['data_path'], file))).get_fdata(), nib.load(os.path.join(config['data_path'], file)).affine), file) for file in data_files]
    segmentation = [((nib.as_closest_canonical(nib.load(os.path.join(config['labels_path'], seg))).get_fdata(), nib.load(os.path.join(config['labels_path'], seg)).affine), seg) for seg in segmentation_files]
    # M: as closest canonnical est utilisee pour rectifier l'image vers la coupe canonique la plus proche. 
    # .affine est utilisee pour garder la matrice contenant la transformation entre les coordonnees matricielles des voxels et leurs coordonnees dans l'espace physique du scanner

    print("Number of MRI files loaded:", len(mri))
    print("Number of segmentation files loaded:", len(segmentation))


    # Split the data into training, validation and test sets
    data_train, data_val, labels_train, label_val = train_test_split(mri, segmentation, test_size=config['val_split'], random_state=42)


    # Create train, val and test datasets of tuples containing the data and the associated labels
    train_dataset = []
    val_dataset = []

    for mri_train in data_train:
        mri_name = mri_train[1]
        for seg in segmentation:
            seg_name = seg[1]
            if mri_name[3:] == seg_name[12:]:   # To modify depending on the file names # M: on fixe une image, ensuite on parcourt tous les labels et on cherche celui qui lui correspond via leur nom
                train_dataset.append((mri_train, seg))

    for mri_val in data_val:
        mri_name = mri_val[1]
        for seg in segmentation:
            seg_name = seg[1]
            if mri_name[3:] == seg_name[12:]:   # To modify depending on the file names
                val_dataset.append((mri_val, seg))

    print("Number of pairs in train dataset :", len(train_dataset))
    print("Number of pairs in val dataset :", len(val_dataset))


    ########################################################################################################################
    #                                                 PREPROCESSING                                                        #
    ########################################################################################################################

    # Only keep one label
    for i in range(len(train_dataset)):
        mri, seg = train_dataset[i]
        seg_data = seg[0][0]
        binary_mask = convert_to_binary_mask(seg_data)
        seg_data = (binary_mask, seg[0][1])
        train_dataset[i] = (mri, (seg_data, seg[1]))  # Keep the affine matrix and filename

    for i in range(len(val_dataset)):
        mri, seg = val_dataset[i]
        seg_data = seg[0][0]
        binary_mask = convert_to_binary_mask(seg_data)
        seg_data = (binary_mask, seg[0][1])
        val_dataset[i] = (mri, (seg_data, seg[1]))

    print("Train dataset 1 length:", len(train_dataset))
    print("Validation dataset 1 length:", len(val_dataset))


    ########################################################################################################################
    #                                                  PATCHS                                                              #
    ########################################################################################################################

    # Parameters for the patches
    patch_size = config['patch_size']
    stride = config['stride']

    # Get patches for the training
    train_patchs_dataset = [None] * len(train_dataset)

    for i in range(len(train_dataset)):
        mri = train_dataset[i][0][0][0]
        seg = train_dataset[i][1][0][0]
        mri_patches = get_patch(mri, patch_size, stride)
        seg_patches = get_patch(seg, patch_size, stride)
        train_patchs_dataset[i] = (mri_patches, seg_patches)

    # Get patches for the validation
    val_patchs_dataset = [None] * len(val_dataset)

    for i in range(len(val_dataset)):
        mri = val_dataset[i][0][0][0]
        seg = val_dataset[i][1][0][0]
        mri_patches = get_patch(mri, patch_size, stride)
        seg_patches = get_patch(seg, patch_size, stride)
        val_patchs_dataset[i] = (mri_patches, seg_patches)

    print("Train dataset length:", len(train_patchs_dataset))
    print("Validation dataset length:", len(val_patchs_dataset))

    # Make a unique list of tuple (mri_patch, segmentation_patch) for all the patches of all the patients
    list_patches_train = []
    for train in train_patchs_dataset:
        for mri, seg in zip(*train):
            list_patches_train.append((mri, seg))

    list_patches_val = []
    for val in val_patchs_dataset:
        for mri, seg in zip(*val):
            list_patches_val.append((mri, seg))

    print("Train dataset length after linearization:", len(list_patches_train))
    print("Validation dataset length after linearization:", len(list_patches_val))


########################################################################################################################
#                                                   MAIN                                                               #
########################################################################################################################
    # Device
    device = torch.device(config['device'])
    print(f"----- Device: {device} -----")
    
    # Initialize TensorBoard writer
    writer = SummaryWriter()

    # Model instantiation
    model = U_Net(config['input_ch'], config['output_ch']).to(device)

    # Loss function
    loss_fn = CombinedLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    ############################ DATA AUGMENTATION ###################################################################################
    
    data_augmentations:list[str] = ["random_anisotropy", "random_noise"]
    transforms_tio = create_data_augmentations(data_augmentations) 

    ## Convertir les patches et segmentations en objets tio.Subject
    train_subjects_list = create_subjects_from_patches(list_patches_train)
    val_subjects_list = create_subjects_from_patches(list_patches_val)

    ## Creer les data loaders (TorchIO Subject Loaders plus precisement)
    train_loader = get_loader_tio(train_subjects_list, config['batch_size'], transforms_tio)
    val_loader = get_loader_tio(val_subjects_list, config['batch_size'], tio.Compose([]))
    
    del list_patches_train, list_patches_val, train_subjects_list, val_subjects_list, 
    
    ################################################################################################################
    
    # Pour logger dans un fichier et à l'écran
    start_timestamp = get_timestamp()
    log_file = open(name_log_file(start_timestamp), "a")
    log_file.write(f"Data Augmentation:{data_augmentations}\n")   
    log_file.write(str(config))

    # Rediriger les prints et erreurs du terminal vers le fichier de log et l'ecran
    sys.stdout = MultiStream(sys.stdout, log_file)  
    sys.stderr = MultiStream(sys.stderr, log_file)  

    # TRAINING LOOP #

    #train_dice_scores = []
    #train_losses = []

    val_dice_scores = []
    val_losses = []

    for epoch in tqdm(range(config['num_epochs'])):
        losses = []
        scores = []
        step = 0
        
        #with warnings.catch_warnings():
        #    warnings.simplefilter("ignore") #torchio lance un warning qui me demande d'utiliser la classe subject loader alors que je l'utilise (on dirait qu'elle est lancée dans la fonction evaluate_tio)
        for batch in train_loader:
            step += 1
            
            if step <= 500/int(config['nb_dataloaders']):   # in 1 epoch, the model will see 250*batch_size patches randomly selected
                model.train()
                optimizer.zero_grad()
                x, y = batch['t1'][tio.DATA], batch['label'][tio.DATA]
                
                x = x.to(device)
                y = y.squeeze().long().to(device)

                pred = model(x)
                loss = loss_fn(pred, y)

                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                scores.append(dice:=dice_score(pred, y).cpu().detach().numpy())

                
                del loss, x, y, pred
            else:
                break

        print("\n Train loss: ", np.mean(losses))
        print("\n Train dice score: ", np.mean(scores))

        eval_loss, eval_dice = evaluate_tio(val_loader, model, loss_fn, device) 
        print("\n Val loss: ", eval_loss)
        print("\n Val dice score: ", eval_dice)
        val_losses.append(eval_loss)
        val_dice_scores.append(eval_dice)

        # Write train and validation losses and accuracies to TensorBoard
        writer.add_scalar('Loss/Train', np.mean(losses), epoch)
        writer.add_scalar('Loss/Validation', eval_loss, epoch)
        writer.add_scalar('Dice_Score/Train', np.mean(scores), epoch)
        writer.add_scalar('Dice_Score/Validation', eval_dice, epoch)

        scheduler.step()
        
    # Close TensorBoard writer
    writer.close()
    log_file.close()
    save_model_weights(model.state_dict(), start_timestamp)
    sys.stdout = sys.__stdout__random_gamma
    sys.stderr = sys.__stderr__
