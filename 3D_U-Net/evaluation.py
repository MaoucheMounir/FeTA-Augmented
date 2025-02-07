import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
import numpy as np

from variables import config


def evaluate(dataloader, model, criterion, device, threshold=0.5):
    """Evaluate the model on the validation set."""
    model.eval()
    losses = []
    scores = []

    for batch in dataloader:
        x, y = batch
        # adding a channel dimension to the x and y tensors
        x = x.unsqueeze(1).float().to(device)
        y = y.long().to(device)

        pred = model(x)

        loss = criterion(pred, y)
        losses.append(loss.item())
        scores.append(dice_score(pred, y).cpu().detach().numpy())

        del x, y, pred, loss

    return np.mean(losses), np.mean(scores)

def evaluate_tio(dataloader, model, criterion, device, threshold=0.5):
    """Evaluate the model on the validation set."""
    model.eval()
    losses = []
    scores = []

    for batch in dataloader:
        x, y = batch['t1'][tio.DATA], batch['label'][tio.DATA]
        
        # adding a channel dimension to the x and y tensors
        x = x.float().to(device)
        y = y.squeeze().long().to(device)

        pred = model(x)

        loss = criterion(pred, y)
        losses.append(loss.item())
        scores.append(dice_score(pred, y).cpu().detach().numpy())

        del x, y, pred, loss

    return np.mean(losses), np.mean(scores)


class DiceLoss(nn.Module):
    """Dice loss."""
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

def dice_score_multiclass(prediction, ground_truth, num_classes, smooth=1e-6, class_=None):
    """
    Calculate the Dice score for multi-class segmentation.

    Args:
    - prediction (numpy array): The predicted segmentation mask.
    - ground_truth (numpy array): The ground truth segmentation mask.
    - num_classes (int): The number of classes in the segmentation.
    - smooth (float): Smoothing constant to avoid division by zero.

    Returns:
    - float: Average Dice score across all classes.
    """
    dice_scores = []

    for c in range(1, num_classes + 1):
        prediction_c = (prediction == c).astype(int)
        if class_ is None:
            ground_truth_c = (ground_truth == c).astype(int)
        else:
            ground_truth_c = (ground_truth == class_).astype(int)

        intersection = np.sum(prediction_c * ground_truth_c)
        union = np.sum(prediction_c) + np.sum(ground_truth_c)

        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice)

        # print("Dice score for class", c, ":", dice)

    return dice_scores, np.mean(dice_scores)


def precision_score_multiclass(prediction, ground_truth, num_classes, smooth=1e-6, class_=None):
    """
    Calculate the precision score for multi-class segmentation.

    Args:
    - prediction (numpy array): The predicted segmentation mask.
    - ground_truth (numpy array): The ground truth segmentation mask.
    - num_classes (int): The number of classes in the segmentation.
    - smooth (float): Smoothing constant to avoid division by zero.

    Returns:
    - float: Average precision score across all classes.
    """
    precision_scores = []

    for c in range(1, num_classes + 1):
        prediction_c = (prediction == c).astype(int)
        if class_ is None:
            ground_truth_c = (ground_truth == c).astype(int)
        else:
            ground_truth_c = (ground_truth == class_).astype(int)

        true_positives = np.sum(prediction_c * ground_truth_c)
        predicted_positives = np.sum(prediction_c)

        precision = (true_positives + smooth) / (predicted_positives + smooth)
        precision_scores.append(precision)

    return np.mean(precision_scores)


def recall_score_multiclass(prediction, ground_truth, num_classes, smooth=1e-6, class_=None):
    """
    Calculate the recall score for multi-class segmentation.

    Args:
    - prediction (numpy array): The predicted segmentation mask.
    - ground_truth (numpy array): The ground truth segmentation mask.
    - num_classes (int): The number of classes in the segmentation.
    - smooth (float): Smoothing constant to avoid division by zero.

    Returns:
    - float: Average recall score across all classes.
    """
    recall_scores = []

    for c in range(1, num_classes + 1):
        prediction_c = (prediction == c).astype(int)
        if class_ is None:
            ground_truth_c = (ground_truth == c).astype(int)
        else:
            ground_truth_c = (ground_truth == class_).astype(int)

        true_positives = np.sum(prediction_c * ground_truth_c)
        actual_positives = np.sum(ground_truth_c)

        recall = (true_positives + smooth) / (actual_positives + smooth)
        recall_scores.append(recall)

    return np.mean(recall_scores)




def volume_ventricles(segmentation, segmentation_gt, v_label=1):
    """
    Calculate the volume of the Ventricles for a given segmentation.

    Returns:
    - float: Volume of the Ventricles
    - float: Error (difference between the volume of the Ventricles and the expected volume)
    """
    # count the number of voxels that have the value class_
    ventricles = (segmentation == v_label).astype(int)
    ventricles_volume = np.sum(ventricles)

    # count the number of voxels that have the value class_ in the ground truth segmentation
    ventricles_gt = (segmentation_gt == v_label).astype(int)
    ventricles_volume_gt = np.sum(ventricles_gt)

    # calculate the error
    error = (ventricles_volume - ventricles_volume_gt)/1000  # divide by 1000 to convert from mm^3 to cm^3

    return ventricles_volume, error
