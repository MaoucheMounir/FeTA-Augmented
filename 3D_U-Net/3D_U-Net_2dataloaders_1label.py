########################################################################################################################
#                                                  IMPORTS                                                             #
########################################################################################################################

from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import os

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import nibabel as nib
import sys
import time

from scipy import ndimage


########################################################################################################################
#                                                 VARIABLES                                                            #
########################################################################################################################

config = {
        # Paths
        'data_path': '',   # Path to the MRI data
        'labels_path': '',   # Path to the segmentations
        'weights_path': '',   # Path to save the model weights

        # Model
        'input_ch': 1,
        'output_ch': 2,   # ventricles + background
        # 'output_ch': 8,
        'nb_dataloaders': 2,

        # Hyper-parameters
        'lr': 0.01,
        'test_split': 0.2,
        'val_split': 0.1,
        'smooth': 1e-5,
        'ee': sys.float_info.epsilon,

        # Training settings
        'num_epochs': 500,
        'batch_size': 2,
        'device': "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

}


########################################################################################################################
#                                                 FUNCTIONS & CLASSES                                                  #
########################################################################################################################

def convert_to_binary_mask(segmentation, target_label=4):
    """Converts a segmentation to a binary mask."""
    binary_mask = (segmentation == target_label).astype(np.float32)
    return binary_mask


def get_loader(dataset, batch_size):
    """Builds and returns Dataloader."""

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True)
    return data_loader


def evaluate(dataloader, model, criterion, threshold=0.5):
    model.eval()
    losses = []
    scores = []

    for batch in dataloader:
        x1, y1 = batch[0]
        x2, y2 = batch[1]

        # concatenate the tensors
        x = torch.concat((x1, x2), dim=0)
        y = torch.concat((y1, y2), dim=0)

        x = x.unsqueeze(1).float().to(config['device'])   # adding a channel dimension to the x tensor
        y = y.long().to(config['device'])

        pred = model(x)

        loss = criterion(pred, y)
        losses.append(loss.item())
        scores.append(dice_score(pred, y).cpu().detach().numpy())

        del x, y, pred, loss

    return np.mean(losses), np.mean(scores)


class DiceLoss(nn.Module):
    """ Computes the Dice Loss. """
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        target = F.one_hot(target, num_classes=config['output_ch']).permute(0, 4, 1, 2, 3)
        target = target.float()

        intersection = (pred * target).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = dice.mean(dim=1)  # to have the mean per structure
        dice_loss = 1 - dice_loss.mean()

        return dice_loss


class CombinedLoss(nn.Module):
    """Combined loss of crossentropy and dice. """
    def __init__(self, weight_ce=1.0, weight_dice=1.0):
        super(CombinedLoss, self).__init__()
        # we already apply a softmax at the end of the network, thus we have to do the crossentropy loss "by hand"
        self.cross_entropy = nn.NLLLoss()
        self.dice_loss = DiceLoss()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

    def forward(self, pred, target):
        ce_loss = self.cross_entropy(torch.log(pred + config['ee']), target)
        dice_loss = self.dice_loss(pred, target)
        return self.weight_ce * ce_loss + self.weight_dice * dice_loss


def dice_score(pred, target, smooth=1e-5):
    """Computes the dice score."""
    target = F.one_hot(target, num_classes=config['output_ch']).permute(0, 4, 1, 2, 3)
    target = target.float()

    intersection = (pred * target).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))

    dice = (2. * intersection + smooth) / (union + smooth)
    dice = dice.mean(dim=1)

    return dice.mean()


########################################################################################################################
#                                                   DATA                                                               #
########################################################################################################################

# Find all the files in the data and segmentation folders which end with .nii.gz
data_files = [file for file in os.listdir(config['data_path']) if file.endswith('.nii.gz')]
segmentation_files = [f for f in os.listdir(config['labels_path']) if f.endswith('.nii.gz')]

# print("Number of MRI files:", len(data_files))
# print("Number of segmentation files:", len(segmentation_files))

# Load all the data and labels
# Each element of the list is a tuple containing the data/label and the file name
mri = [((nib.as_closest_canonical(nib.load(config['data_path'] + file)).get_fdata(), nib.load(config['data_path'] + file).affine), file) for file in data_files]
segmentation = [((nib.as_closest_canonical(nib.load(config['labels_path'] + seg)).get_fdata(), nib.load(config['labels_path'] + seg).affine), seg) for seg in segmentation_files]

mri_number = len(mri)
print("Number of MRI files loaded:", len(mri))
print("Number of segmentation files loaded:", len(segmentation))

# Retrieve the segmentation file associated with the mri file
data_list = []
for i in range(mri_number):
    mri_name = mri[i][1][4:7]   # To modify depending on the file names
    for j in range(len(segmentation)):
        seg_name = segmentation[j][1][-10:-7]   # To modify depending on the file names
        if mri_name == seg_name:
            data_list.append((mri[i], segmentation[j]))

print("Number of couples MRI/segmentation:", len(data_list))


########################################################################################################################
#                                                 PREPROCESSING                                                        #
########################################################################################################################

# Only keep one label
for i in range(len(data_list)):
    mri, seg = data_list[i]
    seg_data = seg[0][0]
    binary_mask = convert_to_binary_mask(seg_data, target_label=4)
    seg_data = (binary_mask, seg[0][1])
    data_list[i] = (mri, (seg_data, seg[1]))

print("Number of couples MRI/segmentation after preprocessing:", len(data_list))


########################################################################################################################
#                                                    MODEL                                                             #
########################################################################################################################

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv3d(64, output_ch, kernel_size=1, stride=1, padding=0)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        d0 = self.softmax(d1)

        return d0


########################################################################################################################
#                                                  PATCHS                                                              #
########################################################################################################################

def get_patch(image, seg, patch_size, stride, pourcentage=0.05):
    """
    Extract patches from a 3D image.

    Args:
        image (np.array): 3D image.
        patch_size (tuple): Size of the patch to extract.
        stride (int): Stride to use when extracting the patches.

    Returns:
        list: List of patches.
    """

    non_V_patches = []
    V_patches = []
    patch_size = (patch_size[0], patch_size[1], patch_size[2])

    for i in range(0, image.shape[0] - patch_size[0] + 1, stride):
        for j in range(0, image.shape[1] - patch_size[1] + 1, stride):
            for k in range(0, image.shape[2] - patch_size[2] + 1, stride):
                mri_patch = image[i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]]
                seg_patch = seg[i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]]
                # if seg_patch contains at least x% of ventricles, we consider it as a V patch
                if np.count_nonzero(seg_patch == 1) >= pourcentage * np.prod(patch_size):   # 1 is the label of the ventricles after preprocessing
                    V_patches.append((mri_patch, seg_patch))
                else:
                    non_V_patches.append((mri_patch, seg_patch))

    # print("Number of V patches extracted:", len(V_patches))
    # print("Number of non-V patches extracted:", len(non_V_patches))

    return V_patches, non_V_patches


########################################################################################################################

# Parameters for the patches
patch_size = (128, 128, 128)
stride = 32
pourcentage = 0.05
dataset_non_V = []
dataset_V = []


# Get the patches for the MRI and segmentation of each patient
for i in range(len(data_list)):
    V_patches, non_V_patches = get_patch(data_list[i][0][0][0], data_list[i][1][0][0], patch_size, stride, pourcentage)
    dataset_V.append(V_patches)
    dataset_non_V.append(non_V_patches)

# Save the segmentation patches for the first patient
# patches_to_save = dataset_V[:20]
# for i in range(len(patches_to_save)):
#     seg_patch = patches_to_save[i][1]
#     nib.save(nib.Nifti1Image(seg_patch, np.eye(4)), f"path/seg_patch_{i}.nii.gz")

# Make a unique list of tuple (mri_patch, segmentation_patch) for all the patches of all the patients
list_patches_V = []
for subject in dataset_V:
    for patch in subject:
        list_patches_V.append(patch)

list_patches_non_V = []
for subject in dataset_non_V:
    for patch in subject:
        list_patches_non_V.append(patch)

print("Number of patches with {}% of ventricles extracted:".format(pourcentage), len(list_patches_V))
print("Number of patches without enough ventricles extracted:", len(list_patches_non_V))

# Split the dataset into training and validation sets
train_dataset_V, val_dataset_V = train_test_split(list_patches_V, test_size=config['test_split'], random_state=42)
train_dataset_non_V, val_dataset_non_V = train_test_split(list_patches_non_V, test_size=config['test_split'], random_state=42)

print("Number of patches in training dataset V:", len(train_dataset_V))
print("Number of patches in validation dataset V:", len(val_dataset_V))
print("Number of patches in training dataset non V:", len(train_dataset_non_V))
print("Number of patches in validation dataset non V:", len(val_dataset_non_V))


########################################################################################################################
#                                                   MAIN                                                               #
########################################################################################################################

if __name__ == '__main__':
    # Device
    device = torch.device(config['device'])
    print(f"----- Device: {device} -----")

    # Initialize TensorBoard writer
    writer = SummaryWriter()

    # Model instantiation
    model = U_Net(config['input_ch'], config['output_ch']).to(device)

    # Loss function
    # loss_fn = nn.CrossEntropyLoss()
    # loss_fn = DiceLoss()
    loss_fn = CombinedLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    # Data loaders
    # Need batch size to be a mutiple of 2
    if config['batch_size'] < 2:
        print('Batch size must be at least 2')
        config['batch_size'] = 2
    train_loader1 = get_loader(train_dataset_V, int(config['batch_size']/2))
    val_loader1 = get_loader(val_dataset_V, int(config['batch_size']/2))
    train_loader2 = get_loader(train_dataset_non_V, int(config['batch_size']/2))
    val_loader2 = get_loader(val_dataset_non_V, int(config['batch_size']/2))


    ################################################################################################################

    # TRAINING LOOP #

    train_dice_scores = []
    train_losses = []

    val_dice_scores = []
    val_losses = []

    for epoch in tqdm(range(config['num_epochs'])):

        losses = []
        scores = []
        step = 0

        for batch in zip(train_loader1, train_loader2):
            step += 1

            if step <= 500/int(config['nb_dataloaders']):   # in 1 epoch, the model will see 250*batch_size patches randomly selected
                model.train()
                optimizer.zero_grad()
                x1, y1 = batch[0]
                x2, y2 = batch[1]

                # concatenate the tensors
                x = torch.concat((x1, x2), dim=0)
                y = torch.concat((y1, y2), dim=0)

                x = x.unsqueeze(1).float().to(config['device'])   # adding a channel dimension to the x tensor
                y = y.long().to(config['device'])

                pred = model(x)

                loss = loss_fn(pred, y)
                loss.backward()

                optimizer.step()

                losses.append(loss.item())
                scores.append(dice_score(pred, y).cpu().detach().numpy())

                del loss, x, y, pred
            else:
                break

        print("\n Train loss: ", np.mean(losses))
        print("\n Train dice score: ", np.mean(scores))

        eval_loss, eval_dice = evaluate(zip(val_loader1, val_loader2), model, loss_fn)
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

    # Save model weights
    weights_name = f"model_weights_{time.strftime('%Y%m%d-%H%M%S')}.pth"
    weights_path = os.path.join(config['weights_path'], weights_name)
    torch.save(model.state_dict(), weights_path)