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
        # 'output_ch': 2,   # ventricles + background
        'output_ch': 8,   # 7 structures + background
        'nb_dataloaders': 1,

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

def get_loader(dataset, batch_size):
    """Builds and returns Dataloader."""

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True)
    return data_loader


def evaluate(dataloader, model, criterion, threshold=0.5):
    """ Evaluate the model on the validation set. """
    model.eval()
    losses = []
    scores = []

    for batch in dataloader:
        x, y = batch
        
        x = x.unsqueeze(1).float().to(device)   # adding a channel dimension to the x tensor
        y = y.long().to(device)

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
    """Combined loss of crossentropy and dice."""
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
        if mri_name[3:] == seg_name[12:]:   # To modify depending on the file names
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


########################################################################################################################

# Parameters for the patches
patch_size = (128, 128, 128)
stride = 32

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
    train_loader = get_loader(list_patches_train, config['batch_size'])
    val_loader = get_loader(list_patches_val, config['batch_size'])


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

        for batch in train_loader:
            step += 1

            if step <= 500/int(config['nb_dataloaders']):   # in 1 epoch, the model will see 250*batch_size patches randomly selected
                model.train()
                optimizer.zero_grad()
                x, y = batch

                x = x.unsqueeze(1).float().to(device)   # adding a channel dimension to x
                y = y.long().to(device)

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

        eval_loss, eval_dice = evaluate(val_loader, model, loss_fn)
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